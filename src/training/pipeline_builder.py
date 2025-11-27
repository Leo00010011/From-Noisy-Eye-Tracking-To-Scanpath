import torch
from torch.utils.data import DataLoader, random_split, Subset
from  torchvision.transforms import v2
import numpy as np
from src.data.datasets import FreeViewInMemory, seq2seq_padded_collate_fn
from src.data.parsers import CocoFreeView
from src.model.path_model import PathModel
from src.model.mixer_model import MixerModel
from src.model.dino_wrapper import DinoV3Wrapper
from src.data.datasets import FreeViewImgDataset, CoupledDataloader, DeduplicatedMemoryDataset



class PipelineBuilder:
    def __init__(self, config):
        # dataset parameters
        self.config = config
        if self.config.model.device.startswith('cuda') and torch.cuda.is_available():
            self.device = torch.device(self.config.model.device)
        else:
            if self.config.model.device.startswith('cuda'):
                print("CUDA not available, using CPU.")
            self.device = torch.device('cpu')
            
    def load_dataset(self):
        self.PathDataset = FreeViewInMemory(sample_size= self.config.data.sample_size,
                                     log = self.config.data.log, 
                                     start_index=self.config.data.start_index)
        if self.config.data.use_img_dataset:
            self.data = CocoFreeView()
            self.data.filter_by_idx(self.PathDataset.data_store['filtered_idx'])

    def split_data(array, ratios):
        indices = np.arange(len(array))
        sub_count = len(array)
        np.random.shuffle(indices)
        train_array = array[indices[:int(ratios[0]*sub_count)]]
        val_array = array[indices[int(ratios[0]*sub_count):int((ratios[0]+ratios[1])*sub_count)]]
        test_array = array[indices[int((ratios[0]+ratios[1])*sub_count):]]
        return train_array, val_array, test_array
    
    def log_split(self,train_subjects,val_subjects,test_subjects,train_stimuli,
                  val_stimuli,test_stimuli,train_idx,val_idx,test_idx,stimuli):
        print(f""" 
                Train subjects:  {train_subjects})
                'Validation subjects:  {val_subjects}
                'Test subjects:  {test_subjects}
                'Stimuli Train set size:  {len(train_stimuli)}, ' percentage:  {len(train_stimuli)/len(stimuli)*100}
                'Stimuli Validation set size:  {len(val_stimuli)}, ' percentage:  {len(val_stimuli)/len(stimuli)*100}
                'Stimuli Test set size:  {len(test_stimuli)}, ' percentage:  {len(test_stimuli)/len(stimuli)*100}
                'Scanpath Train set size:  {len(train_idx)}, ' percentage:  {len(train_idx)/len(self.PathDataset)*100}
                'Scanpath Validation set size:  {len(val_idx)}, ' percentage:  {len(val_idx)/len(self.PathDataset)*100}
                'Scanpath Test set size:  {len(test_idx)}, ' percentage:  {len(test_idx)/len(self.PathDataset)*100}
                """)


    def make_splits(self):
        if self.config.data.split_strategy.name == "disjoint":
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
        if not self.config.data.use_img_dataset:
            train_set = Subset(self.PathDataset, train_idx)
            val_set = Subset(self.PathDataset, val_idx)
            test_set = Subset(self.PathDataset, test_idx)
            train_dataloader = DataLoader(train_set, batch_size=self.config.data.batch_size, shuffle=True, num_workers=0, collate_fn= seq2seq_padded_collate_fn)
            val_dataloader = DataLoader(val_set, batch_size=self.config.data.batch_size, shuffle=False, num_workers=0, collate_fn= seq2seq_padded_collate_fn)
            test_dataloader = DataLoader(test_set, batch_size=self.config.data.batch_size, shuffle=False, num_workers=0, collate_fn= seq2seq_padded_collate_fn)
            return train_dataloader, val_dataloader, test_dataloader
        else:
            transform = PipelineBuilder.make_transform(resize_size= self.config.data.get('img_size', 256))
            if self.config.data.use_cached_img_dataset:
                dataset = DeduplicatedMemoryDataset(self.data, resize_size= self.config.data.get('img_size', 256), transform=transform)
            else:
                dataset = FreeViewImgDataset(self.data, transform=transform)
            train_set = Subset(dataset, train_idx)
            val_set = Subset(dataset, val_idx)
            test_set = Subset(dataset, test_idx)
            train_dataloader = CoupledDataloader(self.PathDataset,
                                                 train_set,
                                                 shuffle = True,
                                                 batch_size=self.config.data.batch_size,
                                                 num_workers = self.config.data.num_workers,
                                                 persistent_workers = self.config.data.persistent_workers,
                                                 prefetch_factor = self.config.data.prefetch_factor,
                                                 pin_memory = self.config.data.pin_memory)
            val_dataloader = CoupledDataloader(self.PathDataset,
                                                 val_set,
                                                 shuffle = False,
                                                 batch_size=self.config.data.batch_size,
                                                 num_workers = self.config.data.num_workers,
                                                 persistent_workers = self.config.data.persistent_workers,
                                                 prefetch_factor = self.config.data.prefetch_factor,
                                                 pin_memory = self.config.data.pin_memory)
            test_dataloader = CoupledDataloader(self.PathDataset,
                                                 test_set,
                                                 shuffle = False,
                                                 batch_size=self.config.data.batch_size,
                                                 num_workers = self.config.data.num_workers,
                                                 persistent_workers = self.config.data.persistent_workers,
                                                 prefetch_factor = self.config.data.prefetch_factor,
                                                 pin_memory = self.config.data.pin_memory)
            return train_dataloader, val_dataloader, test_dataloader
    
    def clear_dataframe(self):
        del self.data

    def build_model(self) -> torch.nn.Module:
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
                    device=self.device,
                    weights=self.config.model.image_encoder.weights
                )
                image_dim = self.config.model.image_encoder.image_dim

            model = MixerModel(input_dim = self.config.model.input_dim,
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
                              head_type = self.config.model.get('head_type', 'linear'),
                              mlp_head_hidden_dim = self.config.model.get('mlp_head_hidden_dim', None),
                              activation = activation,
                              device = self.device,
                              image_encoder = image_encoder,
                              n_feature_enhancer = self.config.model.n_feature_enhancer,
                              image_dim = image_dim)
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
                              head_type = self.config.model.get('head_type', 'linear'),
                              mlp_head_hidden_dim = self.config.model.get('mlp_head_hidden_dim', None),
                              activation = activation,
                              device = self.device)
        else:
            raise ValueError(f"Model name {model_name} not supported.")
        return model
    
    def build_optimizer(self, model: PathModel):
        optimizer = torch.optim.Adam(model.parameters(), 
                                     lr = self.config.training.learning_rate)
        return optimizer
    
    def build_scheduler(self, optimizer: torch.optim.Optimizer, train_dataloader: DataLoader):
        if self.config.scheduler.type == 'one_cycle':
            if self.config.training.log:
                print("Using One Cycle Learning Rate Scheduler")
            sample_count = len(train_dataloader.dataset)
            steps_per_epoch = int(np.ceil(sample_count / train_dataloader.batch_size))
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
        else:
            raise ValueError(f"Scheduler type {self.config.scheduler.type} not supported.")
        return scheduler

    def build_loss_fn(self):
        # TODO allow to configurate different loss functions
        # The duration, coordinates and end_token losses should have independent losses
        pass
        


    def training_summary(self, n_samples):
        print(""" Traning Summary:
                Number of epochs: {}
                Classification Loss Weight: {}
                Validation every {} epochs
                Training Samples: {}
                """.format(self.config.training.num_epochs, 
                           self.config.training.cls_weight, 
                           self.config.training.val_interval, 
                           n_samples))

    
