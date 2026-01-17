import torch
from src.model.model_io import load_model_from_path
from src.data.transforms import AddCurriculumNoise
from src.training.training_utils import DenoiseDropoutScheduler
from src.training.weights_scheduler import WeightsScheduler
from src.training.training_utils import ScheduledSampling, WarmupStableDecayScheduler
from src.model.loss_functions import EntireRegLossFunction, SeparatedRegLossFunction, CombinedLossFunction, DenoiseRegLoss, PenaltyReducedFocalLoss,EndBinaryCrossEntropy, EndSoftMax
from torch.utils.data import DataLoader, random_split, Subset
from  torchvision.transforms import v2
import numpy as np
from src.data.datasets import FreeViewInMemory, seq2seq_padded_collate_fn
from src.data.parsers import CocoFreeView
from src.data.transforms import (ExtractRandomPeriod, Normalize, StandarizeTime, LogNormalizeDuration,
                                 AddRandomCenterCorrelatedRadialNoise, DiscretizationNoise, SaveCleanX, QuantileNormalizeDuration, AddGaussianNoiseToFixations, AddHeatmaps)
from src.model.path_model import PathModel
from src.model.mixer_model import MixerModel
from src.model.dino_wrapper import DinoV3Wrapper
from src.data.datasets import FreeViewImgDataset, CoupledDataloader, DeduplicatedMemoryDataset

STR_TO_LOSS_FUNC = {
    'bce_with_logits': torch.nn.functional.binary_cross_entropy_with_logits,
    'end_bce_with_logits': EndBinaryCrossEntropy(),
    'end_softmax': EndSoftMax(),
    'mse': torch.nn.functional.mse_loss,
    'l1': torch.nn.functional.l1_loss
}

def build_extract_random_period(config):
    return  ExtractRandomPeriod(
            start_index=config.get('start_index', 2),
            period_duration=config.get('period_duration', 2600),
            sampling_rate=config.get('sampling_rate', 60),
            downsample_period=config.get('downsample_period', 200),
            random_offset=config.get('random_offset', True)
        )
def build_add_random_center_correlated_radial_noise(config):
    return AddRandomCenterCorrelatedRadialNoise(
            initial_center=[config.get('initial_center_i', 320//2), 
                            config.get('initial_center_j', 512//2)],
            ptoa=1/config.get('angle_to_pixels',16),
            radial_corr=config.get('radial_corr',0.5),
            radial_avg_norm=config.get('radial_avg_norm',4.13),
            radial_std=config.get('radial_std',5.5),
            center_noise_std=config.get('center_noise_std',300),
            center_corr=config.get('center_corr',0.9),
            center_delta_norm=config.get('center_delta_norm',800),
            center_delta_r=config.get('center_delta_r',0.2)
        )

def build_discretization_noise(config):
    return DiscretizationNoise((config.get('image_H',320),
                                config.get('image_W',512)))

def build_normalize_coords(config, key = None):
    max_value = torch.tensor([config.image_W,config.image_H])
    if not hasattr(config, 'mode'):
        return Normalize(key='x', mode=config.key, max_value=max_value) 
    elif key is not None:
        return Normalize(key = key, mode=config.mode, max_value=max_value)
    else:       
        return Normalize(key=config.key, mode=config.mode, max_value=max_value)

def build_normalize_time(config, key = None):
    if not hasattr(config, 'mode'):
        return Normalize(key='x', mode=config.key, max_value=config.period_duration)
    elif key is not None:
        return Normalize(key = key, mode=config.mode, max_value=config.period_duration)
    else:
        return Normalize(key=config.key, mode=config.mode, max_value=config.period_duration)

def build_log_normalize_duration(config, key = None):
    if key is not None:
        return LogNormalizeDuration(mean=config.mean, std=config.std, scale=config.scale, use_tan=config.get('use_tan', False), key = key)
    else:
        return LogNormalizeDuration(mean=config.mean, std=config.std, scale=config.scale, use_tan=config.get('use_tan', False), key = 'y')

def build_quantile_normalize_duration(config, key = None):
    if key is not None:
        return QuantileNormalizeDuration(key = key, pkl_path = config.get('pkl_path', 'quantile_transformer.pkl'))
    else:
        return QuantileNormalizeDuration(key = 'y', pkl_path = config.get('pkl_path', 'quantile_transformer.pkl'))
    
def build_add_curriculum_noise(config):
    return AddCurriculumNoise(num_steps = config.get('num_steps', 10), s = config.get('s', 0.002), start_alpha = config.get('start_alpha', 0.4), n_updates = config.get('n_updates', -1))

class PipelineBuilder:
    def __init__(self, config):
        # dataset parameters
        self.config = config
        self.PathDataset = None
        self.img_dataset = None
        self.data = None
        self.curriculum_noise = None
        if self.config.model.device.startswith('cuda') and torch.cuda.is_available():
            self.device = torch.device(self.config.model.device)
        else:
            if self.config.model.device.startswith('cuda'):
                print("CUDA not available, using CPU.")
            self.device = torch.device('cpu')
            
    def load_dataset(self):
        transforms = []
        # check if self.config.data.transforms exits
        if hasattr(self.config.data, 'transforms'):
            has_save_clean_x = False
            has_in_tgt = False
            for transform_str in self.config.data.transforms.transform_list:
                transform_config = self.config.data.transforms.get(transform_str)
                if transform_str == 'ExtractRandomPeriod':
                    transforms.append(build_extract_random_period(transform_config))
                elif transform_str == 'AddRandomCenterCorrelatedRadialNoise':
                    transforms.append(build_add_random_center_correlated_radial_noise(transform_config))
                elif transform_str == 'DiscretizationNoise':
                    transforms.append(build_discretization_noise(transform_config))
                elif transform_str == 'NormalizeCoords':
                    transforms.append(build_normalize_coords(transform_config))
                    if has_save_clean_x:
                        transforms.append(build_normalize_coords(transform_config, key = 'clean_x'))
                elif transform_str == 'NormalizeFixationCoords':
                    transforms.append(build_normalize_coords(transform_config, key = 'y'))
                    if has_in_tgt:
                        transforms.append(build_normalize_coords(transform_config, key = 'in_tgt'))
                elif transform_str == 'NormalizeTime':
                    transforms.append(build_normalize_time(transform_config))
                    if has_save_clean_x:
                        transforms.append(build_normalize_time(transform_config, key = 'clean_x'))
                elif transform_str == 'StandarizeTime':
                    transforms.append(StandarizeTime())
                elif transform_str == 'LogNormalizeDuration':
                    transforms.append(build_log_normalize_duration(transform_config))
                    if has_in_tgt:
                        transforms.append(build_log_normalize_duration(transform_config, key = 'in_tgt'))
                elif transform_str == 'NormalizeDuration':
                    transforms.append(build_normalize_time(transform_config, key = 'y'))
                    if has_in_tgt:
                        transforms.append(build_normalize_time(transform_config, key = 'in_tgt'))
                elif transform_str == 'SaveCleanX':
                    transforms.append(SaveCleanX())
                    has_save_clean_x = True
                elif transform_str == 'QuantileNormalizeDuration':
                    transforms.append(build_quantile_normalize_duration(transform_config))
                    if has_in_tgt:
                        transforms.append(build_quantile_normalize_duration(transform_config, key = 'in_tgt'))
                elif transform_str == 'AddGaussianNoiseToFixations':
                    transforms.append(AddGaussianNoiseToFixations(transform_config.get('std', 0)))
                    has_in_tgt = True
                elif transform_str == 'GenerateHeatmaps':
                    transforms.append(AddHeatmaps(image_size = (transform_config.get('image_H', 32), transform_config.get('image_W', 32)),                                                
                                                  sigma = transform_config.get('sigma', 1.5),
                                                  device = self.device,
                                                  dtype = torch.float32))
                elif transform_str == 'AddCurriculumNoise':
                    self.curriculum_noise = build_add_curriculum_noise(transform_config)
                    transforms.append(self.curriculum_noise)
                else:
                    raise ValueError(f"Transform {transform_str} not supported.")
        else:
            transforms = [build_extract_random_period(self.config.data),
                          build_add_random_center_correlated_radial_noise(self.config.data),
                          build_discretization_noise(self.config.data),
                          StandarizeTime()]
        if self.PathDataset is None:
            
            # Check if 'load' attribute exists in self.config.data; if not, use it directly from data
            log = getattr(self.config.data, 'load', None)
            if log is not None and hasattr(self.config.data.load, 'log'):
                log_value = self.config.data.load.log
            elif hasattr(self.config.data, 'log'):
                log_value = self.config.data.log
            else:
                raise AttributeError("Neither 'load.log' nor 'log' is present in self.config.data")
            self.PathDataset = FreeViewInMemory(transforms=transforms, log=log_value)
        else:
            self.PathDataset.transforms = transforms
        # Check 'use_img_dataset' in 'load', if not, look in data directly
        load_config = None
        if hasattr(self.config.data, 'load'):
            load_config = self.config.data.load
        else:
            load_config = self.config.data
        self.img_size = load_config.img_size
        if hasattr(load_config, 'use_img_dataset') and load_config.use_img_dataset:
            if self.data is None:
                self.data = CocoFreeView()
                self.data.filter_by_idx(self.PathDataset.data_store['filtered_idx'])
            
            transform = PipelineBuilder.make_transform(resize_size= load_config.img_size)
            if self.img_dataset is None:
                self.img_dataset = DeduplicatedMemoryDataset(self.data, resize_size= load_config.img_size, transform=transform)
            else:
                self.img_dataset.resize_size = load_config.img_size
                self.img_dataset.runtime_transform = transform

    def log_split(self,train_subjects,val_subjects,test_subjects,train_stimuli,
                  val_stimuli,test_stimuli,train_idx,val_idx,test_idx,stimuli):
        if train_subjects is not None:
            print(f""" 
                    Train subjects:  {train_subjects})
                    'Validation subjects:  {val_subjects}
                    'Test subjects:  {test_subjects}
                    """)
        else:    
            print(f""" 
                'Stimuli Train set size:  {len(train_stimuli)}, ' percentage:  {len(train_stimuli)/len(stimuli)*100}
                'Stimuli Validation set size:  {len(val_stimuli)}, ' percentage:  {len(val_stimuli)/len(stimuli)*100}
                'Stimuli Test set size:  {len(test_stimuli)}, ' percentage:  {len(test_stimuli)/len(stimuli)*100}
                'Scanpath Train set size:  {len(train_idx)}, ' percentage:  {len(train_idx)/len(self.PathDataset)*100}
                'Scanpath Validation set size:  {len(val_idx)}, ' percentage:  {len(val_idx)/len(self.PathDataset)*100}
                'Scanpath Test set size:  {len(test_idx)}, ' percentage:  {len(test_idx)/len(self.PathDataset)*100}
                """)
        
    def split_data(array, ratios):
        indices = np.arange(len(array))
        sub_count = len(array)
        np.random.shuffle(indices)
        train_array = array[indices[:int(ratios[0]*sub_count)]]
        val_array = array[indices[int(ratios[0]*sub_count):int((ratios[0]+ratios[1])*sub_count)]]
        test_array = array[indices[int((ratios[0]+ratios[1])*sub_count):]]
        return train_array, val_array, test_array
    


    def make_splits(self):
        if hasattr(self.config.data, 'split_strategy') and self.config.data.split_strategy.name == 'disjoint':
            subjects = self.data.get_all_subjects()
            train_subjects , val_subjects , test_subjects  = PipelineBuilder.split_data(subjects, 
                                                                                      [self.config.data.split_strategy.train_subjects_per, 
                                                                                       self.config.data.split_strategy.val_subjects_per])
            stimuli = self.data.get_all_stimuli()
            train_stimuli, val_stimuli, test_stimuli = PipelineBuilder.split_data(stimuli, 
                                                                                  [self.config.data.split_strategy.train_stimuli_per, 
                                                                                   self.config.data.split_strategy.val_stimuli_per])
            train_idx, val_idx, test_idx = self.data.get_disjoint_splits(train_subjects, val_subjects, test_subjects,
                                                                train_stimuli, val_stimuli, test_stimuli)
            train_idx , val_idx , test_idx= torch.tensor(train_idx), torch.tensor(val_idx), torch.tensor(test_idx) 
            if self.config.training.log:
                self.log_split(train_subjects, val_subjects, test_subjects, train_stimuli, val_stimuli, 
                               test_stimuli, train_idx, val_idx, test_idx, stimuli)
        elif hasattr(self.config.data, 'split_strategy') and self.config.data.split_strategy.name == 'stimuly_disjoint':
            stimuli = self.data.get_all_stimuli()
            train_stimuli, val_stimuli, test_stimuli = PipelineBuilder.split_data(stimuli, 
                                                                                  [self.config.data.split_strategy.train_stimuli_per, 
                                                                                   self.config.data.split_strategy.val_stimuli_per])
            train_idx, val_idx, test_idx = self.data.get_stimuly_disjoint_splits(train_stimuli, val_stimuli, test_stimuli)
            train_idx , val_idx , test_idx= torch.tensor(train_idx), torch.tensor(val_idx), torch.tensor(test_idx) 
            if self.config.training.log:
                self.log_split(None, None, None, train_stimuli, val_stimuli, test_stimuli, train_idx, val_idx, test_idx, stimuli)
        else:
            total_size = len(self.PathDataset)
            train_size = int(self.config.data.train_per * total_size)
            val_size = int(self.config.data.val_per * total_size)
            idx = torch.randperm(total_size)
            train_idx = idx[:train_size]
            val_idx = idx[train_size:train_size+val_size]
            test_idx = idx[train_size+val_size:]
        return train_idx, val_idx, test_idx

    @staticmethod
    def make_transform(resize_size: int = 256):
        to_tensor = v2.ToImage()
        resize = v2.Resize((resize_size, resize_size), antialias=True)
        to_float = v2.ToDtype(torch.float32, scale=True)
        normalize = v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        return v2.Compose([to_tensor, resize, to_float, normalize])

    def build_dataloader(self, train_idx, val_idx, test_idx) -> DataLoader:
        if hasattr(self.config.data, 'load'):
            dataloader_config = self.config.data.load
        else:
            dataloader_config = self.config.data
        if hasattr(dataloader_config, 'use_img_dataset') and dataloader_config.use_img_dataset:
            train_set = Subset(self.img_dataset, train_idx)
            val_set = Subset(self.img_dataset, val_idx)
            test_set = Subset(self.img_dataset, test_idx)
            train_dataloader = CoupledDataloader(self.PathDataset,
                                                    train_set,
                                                    shuffle = True,
                                                    batch_size=dataloader_config.batch_size,
                                                    num_workers = dataloader_config.num_workers,
                                                    persistent_workers = dataloader_config.persistent_workers,
                                                    prefetch_factor = dataloader_config.prefetch_factor,
                                                    pin_memory = dataloader_config.pin_memory,
                                                    drop_last_batch = False)
            val_dataloader = CoupledDataloader(self.PathDataset,
                                                    val_set,
                                                    shuffle = False,
                                                    batch_size=dataloader_config.batch_size,
                                                    num_workers = dataloader_config.num_workers,
                                                    persistent_workers = dataloader_config.persistent_workers,
                                                    prefetch_factor = dataloader_config.prefetch_factor,
                                                    pin_memory = dataloader_config.pin_memory,
                                                    drop_last_batch = False)
            test_dataloader = CoupledDataloader(self.PathDataset,
                                                    test_set,
                                                    shuffle = False,
                                                    batch_size=dataloader_config.batch_size,
                                                    num_workers = dataloader_config.num_workers,
                                                    persistent_workers = dataloader_config.persistent_workers,
                                                    prefetch_factor = dataloader_config.prefetch_factor,
                                                    pin_memory = dataloader_config.pin_memory,
                                                    drop_last_batch = False)
            return train_dataloader, val_dataloader, test_dataloader
        else:
            batch_size = 128
            if hasattr(dataloader_config, 'batch_size'):
                batch_size = dataloader_config.batch_size
            else:
                batch_size = self.config.training.batch_size
            train_set = Subset(self.PathDataset, train_idx)
            val_set = Subset(self.PathDataset, val_idx)
            test_set = Subset(self.PathDataset, test_idx)
            train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn= seq2seq_padded_collate_fn)
            val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn= seq2seq_padded_collate_fn)
            test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn= seq2seq_padded_collate_fn)
            return train_dataloader, val_dataloader, test_dataloader
    
    def clear_dataframe(self):
        del self.data
        self.data = None

    def build_model(self) -> torch.nn.Module:
        if self.config.training.pretrained_model is not None:
            model = load_model_from_path(self.config.training.pretrained_model)
            return model
        activation = None
        if self.config.model.activation == "relu":
            activation = torch.nn.ReLU()
        elif self.config.model.activation == "gelu":
            activation = torch.nn.GELU()

        model_name = self.config.model.get('name', 'PathModel')

        if model_name == 'MixerModel':
            image_encoder = None
            image_dim = None
            if hasattr(self.config.model, 'image_encoder') and self.config.model.image_encoder.enabled:
                image_encoder = DinoV3Wrapper(
                    repo_path=self.config.model.image_encoder.repo_path,
                    model_name=self.config.model.image_encoder.name,
                    freeze=self.config.model.image_encoder.freeze,
                    regularization=self.config.model.image_encoder.get('regularization', False),
                    device=self.device,
                    weights=self.config.model.image_encoder.weights
                )
                image_dim = self.config.model.image_encoder.image_dim
            if (hasattr(self.config.data, 'transforms') and 
                hasattr(self.config.data.transforms, 'NormalizeCoords') 
                and not hasattr(self.config.data.transforms.NormalizeCoords,'mode')
                and self.config.model.input_encoder == 'Fourier'):
                    input_encoder = 'nerf_fourier'
            else:
                input_encoder = self.config.model.get('input_encoder', 'linear')


            model = MixerModel(input_dim = self.config.model.input_dim,
                              output_dim = self.config.model.output_dim,
                              img_size = self.img_size,
                              n_encoder = self.config.model.n_encoder,
                              n_decoder = self.config.model.n_decoder,
                              model_dim = self.config.model.model_dim,
                              total_dim = self.config.model.total_dim,
                              denoise_head_hidden_dropout = self.config.model.get('denoise_head_hidden_dropout', 0),
                              denoise_head_output_dropout = self.config.model.get('denoise_head_output_dropout', 0),
                              eye_decoder_dropout = self.config.model.get('eye_decoder_dropout', 0),
                              reg_head_output_dropout = self.config.model.get('reg_head_output_dropout', 0),
                              use_deformable_fixation_decoder = self.config.model.get('use_deformable_fixation_decoder', False),
                              use_denoised_coordinates = self.config.model.get('use_denoised_coordinates', False),
                              adapter_dropout = self.config.model.get('adapter_dropout', 0),
                              n_heads = self.config.model.n_heads,
                              ff_dim = self.config.model.ff_dim,
                              image_features_dropout = self.config.model.get('image_features_dropout', 0),
                              max_pos_enc = self.config.model.max_pos_enc,
                              max_pos_dec = self.config.model.max_pos_dec,
                              input_encoder = input_encoder,
                              num_freq_bands = self.config.model.get('num_freq_bands', None),
                              use_enh_img_features = self.config.model.get('use_enh_img_features', False),
                              pos_enc_hidden_dim = self.config.model.get('pos_enc_hidden_dim', None),
                              norm_first = self.config.model.norm_first,
                              dropout_p= self.config.model.dropout_p,
                              head_type = self.config.model.get('head_type', 'linear'),
                              mlp_head_hidden_dim = self.config.model.get('mlp_head_hidden_dim', None),
                              pos_enc_sigma = self.config.model.get('pos_enc_sigma', None),
                              use_rope = self.config.model.get('use_rope', False),
                              word_dropout_prob = self.config.model.get('word_dropout_prob', 0),
                              phases = self.config.training.Phases if hasattr(self.config.training, 'Phases') else [],
                              activation = activation,
                              device = self.device,
                              enh_features_dropout = self.config.model.get('enh_features_dropout', 0),
                              image_encoder = image_encoder,
                              n_feature_enhancer = self.config.model.n_feature_enhancer,
                              image_dim = image_dim,
                              src_word_dropout_prob = self.config.model.get('src_word_dropout_prob', 0),
                              mixed_image_features = self.config.model.get('mixed_image_features', False),
                              n_eye_decoder = self.config.model.get('n_eye_decoder', 0),
                              src_dropout = self.config.model.get('src_dropout', 0),
                              reg_head_dropout = self.config.model.get('reg_head_dropout', 0),
                              tgt_dropout = self.config.model.get('tgt_dropout', 0),
                              dur_head_dropout = self.config.model.get('dur_head_dropout', 0),
                              n_adapter = self.config.model.get('n_adapter', 0),
                              decoder_dropout = self.config.model.get('decoder_dropout', 0),
                              mixer_dropout = self.config.model.get('mixer_dropout', 0),
                              use_deformable_eye_decoder = self.config.model.get('use_deformable_eye_decoder', False),
                              eye_encoder_dropout = self.config.model.get('eye_encoder_dropout', 0),
                              end_dropout = self.config.model.get('end_dropout', 0),
                              add_denoise_head = self.config.model.get('add_denoise_head', True),
                              use_kv_cache = self.config.model.get('use_kv_cache', False),
                              geometric_sigma = self.config.model.get('geometric_sigma', 0),
                              adapter_hidden_dims = self.config.model.image_encoder.get('adapter_hidden_dims', self.config.model.get('mlp_head_hidden_dim', None)))
            
        elif model_name == 'PathModel':
            model = PathModel(input_dim = self.config.model.input_dim,
                              output_dim = self.config.model.output_dim,
                              n_encoder = self.config.model.n_encoder,
                              n_decoder = self.config.model.n_decoder,
                              model_dim = self.config.model.model_dim,
                              total_dim = self.config.model.total_dim,
                              n_heads = self.config.model.n_heads,
                              ff_dim = self.config.model.ff_dim,
                              max_pos_enc = self.config.model.max_pos_enc,
                              max_pos_dec = self.config.model.max_pos_dec,
                              norm_first = self.config.model.norm_first,
                              dropout_p= self.config.model.dropout_p,
                              input_encoder = self.config.model.get('input_encoder', 'linear'),
                              head_type = self.config.model.get('head_type', 'linear'),
                              mlp_head_hidden_dim = self.config.model.get('mlp_head_hidden_dim', None),
                              src_dropout = self.config.model.get('src_dropout', 0),
                              tgt_dropout = self.config.model.get('tgt_dropout', 0),
                              activation = activation,
                              device = self.device)
        else:
            raise ValueError(f"Model name {model_name} not supported.")
        return model
    
    def build_optimizer(self, model: PathModel):
        param_dicts = model.get_parameter_groups(self.config.training.learning_rate)
        optimizer = torch.optim.AdamW(param_dicts, 
                                     weight_decay = self.config.training.get('weight_decay', 0))
        return optimizer
    
    def build_scheduler(self, optimizer: torch.optim.Optimizer, train_dataloader: DataLoader):
        steps_per_epoch = len(train_dataloader)
        if self.config.scheduler.type == 'one_cycle':
            if self.config.training.log:
                print("Using One Cycle Learning Rate Scheduler")
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.config.scheduler.max_lr,
                total_steps=None,
                epochs=self.config.training.num_epochs,
                steps_per_epoch= steps_per_epoch,
                pct_start=self.config.scheduler.pct_start,
                div_factor=self.config.scheduler.div_factor,
                final_div_factor=self.config.scheduler.final_div_factor,
            )
        elif self.config.scheduler.type == 'multistep_lr':
            if self.config.training.log:
                print("Using MultiStep Learning Rate Scheduler")
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.config.scheduler.milestones,
                gamma=self.config.scheduler.gamma
            )
        elif self.config.scheduler.type == 'warmup_stable_decay':
            if self.config.training.log:
                print("Using Warmup Stable Decay Learning Rate Scheduler")
            scheduler = WarmupStableDecayScheduler(
                optimizer,
                warmup_steps=self.config.scheduler.warmup_steps*steps_per_epoch,
                stable_steps=self.config.scheduler.stable_steps*steps_per_epoch,
                decay_steps=self.config.scheduler.decay_steps*steps_per_epoch,
                min_lr=self.config.scheduler.min_lr,
            )
        else:
            raise ValueError(f"Scheduler type {self.config.scheduler.type} not supported.")
        return scheduler

    def build_loss_fn(self, primary_loss = None):
        if primary_loss is not None:
            loss_type = primary_loss
        elif hasattr(self.config, 'loss'):
            if hasattr(self.config.loss, 'complex_type') and self.config.loss.complex_type is not None:
                loss_type = self.config.loss.complex_type
            else:
                loss_type = self.config.loss.type 
        else:
            loss_type = 'entire_reg'
            
        loss_fn = None
        if loss_type == 'entire_reg':
            loss_fn = EntireRegLossFunction(cls_weight = self.config.loss.cls_weight,
                                         cls_func = STR_TO_LOSS_FUNC[self.config.loss.cls_func],
                                         reg_func = STR_TO_LOSS_FUNC[self.config.loss.reg_func])
        elif loss_type == 'separated_reg':
            loss_fn = SeparatedRegLossFunction(cls_weight = self.config.loss.cls_weight,
                                         cls_func = STR_TO_LOSS_FUNC[self.config.loss.cls_func],
                                         coord_func = STR_TO_LOSS_FUNC[self.config.loss.coord_func],
                                         dur_func = STR_TO_LOSS_FUNC[self.config.loss.dur_func],
                                         dur_weight = self.config.loss.dur_weight)
        elif loss_type == 'combined':
            loss_fn = CombinedLossFunction(denoise_loss = DenoiseRegLoss(STR_TO_LOSS_FUNC[self.config.loss.denoise_loss_type]),
                                         fixation_loss = self.build_loss_fn(primary_loss = self.config.loss.fixation_loss_type),
                                         denoise_weight = 0)
        elif loss_type == 'focal_loss':
            loss_fn = PenaltyReducedFocalLoss(alpha = self.config.loss.alpha,
                                         beta = self.config.loss.beta,
                                         cls_func = STR_TO_LOSS_FUNC[self.config.loss.cls_func],
                                         cls_weight = self.config.loss.cls_weight,
                                         dur_func = STR_TO_LOSS_FUNC[self.config.loss.dur_func],
                                         dur_weight = self.config.loss.dur_weight)
        else:
            raise ValueError(f"Loss type {loss_type} not supported.")
        return loss_fn
        
    def build_phases(self):
        if hasattr(self.config.training, 'Phases'):
            output = []
            for phase in self.config.training.Phases:
                phase = self.config.training.get(phase)
                output.append((phase.name, phase.denoise_weight, phase.decisive_metric, phase.epochs))
            return output
        else:
            return [('Combined', .3 ,self.config.training.decisive_metric, self.config.training.num_epochs)]
        
        
    def build_weights_scheduler(self, loss_fn: torch.nn.Module):
        if hasattr(self.config.training, 'weights_scheduler') and self.config.training.weights_scheduler.enabled:
            return WeightsScheduler(init_b = self.config.training.weights_scheduler.init_b,
                                    end_b = self.config.training.weights_scheduler.end_b,
                                    epochs = self.config.training.weights_scheduler.epochs,
                                    alpha = self.config.training.weights_scheduler.alpha,
                                    loss_function = loss_fn,
                                    device = self.device)
        else:
            return None
        
    def build_scheduled_sampling(self, steps_per_epoch: int):
        print("Scheduled Sampling:")
        print("steps per epoch: ", steps_per_epoch)
        if hasattr(self.config.training, 'use_scheduled_sampling') and self.config.training.use_scheduled_sampling:
            return ScheduledSampling(active_epochs = self.config.scheduled_sampling.active_epochs,
                                     device = self.device,
                                     steps_per_epoch = steps_per_epoch,
                                     warmup_epochs = self.config.scheduled_sampling.warmup_epochs,
                                     n_updates = self.config.scheduled_sampling.get('n_updates', -1),
                                     min_prob = self.config.scheduled_sampling.get('min_prob', 0),
                                     use_kv_cache = self.config.model.get('use_kv_cache', False))
        else:
            return None

    def build_denoise_dropout_scheduler(self, model: torch.nn.Module, steps_per_epoch: int):
        if hasattr(self.config.training, 'use_denoise_dropout_scheduler') and self.config.training.use_denoise_dropout_scheduler:
            return DenoiseDropoutScheduler(model = model,
                                       base_prob = self.config.denoise_dropout_scheduler.base_prob,
                                       active_epochs = self.config.denoise_dropout_scheduler.active_epochs,
                                       warmup_epochs = self.config.denoise_dropout_scheduler.warmup_epochs,
                                       steps_per_epoch = steps_per_epoch)
        else:
            return None

    def training_summary(self, n_samples):
        print(""" Traning Summary:
                Number of epochs: {}
                Classification Loss Weight: {}
                Validation every {} epochs
                Training Samples: {}
                """.format(self.config.training.num_epochs, 
                           self.config.loss.cls_weight, 
                           self.config.training.val_interval, 
                           n_samples))

    
