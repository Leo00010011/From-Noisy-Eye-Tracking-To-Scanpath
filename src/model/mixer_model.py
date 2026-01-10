import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from src.model.blocks import (TransformerEncoder, DoubleInputDecoder, MLP, FeatureEnhancer, ArgMaxRegressor,
                              LearnableCoordinateDropout, ResidualRegressor, GatedFusion, TransformerDecoder,
                              TrajectoryHeatmapGenerator)
from src.model.pos_encoders import PositionalEncoding, GaussianFourierPosEncoder, FourierPosEncoder
from src.model.rope_positional_embeddings import RopePositionEmbedding

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MixerModel(nn.Module):
    def __init__(self, n_encoder,
                       n_decoder,
                       img_size = 256,
                       input_dim = 3,
                       output_dim = 3,
                       model_dim = 1024,
                       total_dim = 1024,
                       n_heads = 8,
                       ff_dim = 2048,
                       dropout_p = 0.1,
                       input_encoder = None,
                       max_pos_enc = 8,
                       max_pos_dec = 4,
                       image_features_dropout = 0.3,
                       num_freq_bands = 15,
                       use_enh_img_features = False,
                       pos_enc_hidden_dim = None,
                       activation = F.relu,
                       norm_first = False ,
                       head_type = None,
                       mlp_head_hidden_dim = None,
                       image_encoder = None,
                       n_feature_enhancer = 1,
                       use_denoised_coordinates = False,
                       image_dim = None,
                       pos_enc_sigma = 1.0,
                       use_rope = False,
                       word_dropout_prob = 0.3,
                       src_word_dropout_prob = 0,
                       dur_head_dropout = 0,
                       reg_head_dropout = 0,
                       mixed_image_features = False,
                       mixer_dropout = 0,
                       end_dropout = 0,
                       phases = None,
                       src_dropout = 0,
                       adapter_dropout = 0,
                       tgt_dropout = 0,
                       eye_encoder_dropout = 0,
                       enh_features_dropout = 0,
                       denoise_head_hidden_dropout = 0,
                       denoise_head_output_dropout = 0,
                       n_adapter = 0,
                       n_eye_decoder = 0,
                       use_kv_cache = False,
                       dtype = torch.float32,
                       device = 'cpu'):
        super().__init__()
        factory_mode = {'device':device, 'dtype': dtype}
        self.name = 'MixerModel'
        self.model_dim = model_dim
        self.n_heads = n_heads
        self.image_features_dropout = image_features_dropout
        self.ff_dim = ff_dim
        self.dropout_p = dropout_p
        self.n_encoder = n_encoder
        self.n_decoder = n_decoder
        self.factory_mode = factory_mode
        self.norm_first = norm_first
        self.head_type = head_type
        self.mlp_head_hidden_dim = mlp_head_hidden_dim
        self.image_encoder = image_encoder
        self.n_feature_enhancer = n_feature_enhancer
        self.img_size = img_size
        self.num_freq_bands = num_freq_bands
        self.pos_enc_hidden_dim = pos_enc_hidden_dim
        self.input_encoder = input_encoder
        self.pos_enc_sigma = pos_enc_sigma
        self.use_rope = use_rope
        self.use_enh_img_features = use_enh_img_features and not mixed_image_features
        self.enh_features_dropout = enh_features_dropout
        self.word_dropout_prob = word_dropout_prob
        self.phase = None
        self.src_dropout = src_dropout
        self.tgt_dropout = tgt_dropout
        self.eye_encoder_dropout = eye_encoder_dropout
        self.dur_head_dropout = dur_head_dropout
        self.end_dropout = end_dropout
        self.reg_head_dropout = reg_head_dropout
        self.mixed_image_features = mixed_image_features
        self.mixer_dropout = mixer_dropout
        self.n_adapter = n_adapter
        self.denoise_head_hidden_dropout = denoise_head_hidden_dropout
        self.denoise_head_output_dropout = denoise_head_output_dropout
        self.denoise_modules = []
        self.fixation_modules = []
        self.use_denoised_coordinates = use_denoised_coordinates
        self.n_eye_decoder = n_eye_decoder
        self.scheduled_sampling = None
        self.src_word_dropout_prob = src_word_dropout_prob
        self.adapter_dropout = adapter_dropout
        self.use_kv_cache = use_kv_cache
        # SPECIAL TOKENS
        if mixed_image_features:
            self.mix_enh_image_features = GatedFusion(model_dim, dropout_p = mixer_dropout, **factory_mode)
            self.denoise_modules.append(self.mix_enh_image_features)
        self.start_token = nn.Parameter(torch.randn(1,1,model_dim,**factory_mode))
        self.fixation_modules.append(self.start_token)
        if word_dropout_prob > 0:
            self.word_dropout = LearnableCoordinateDropout(model_dim=model_dim, dropout_prob=word_dropout_prob, **factory_mode)
            self.fixation_modules.append(self.word_dropout)
        if src_word_dropout_prob > 0:
            self.src_word_dropout = LearnableCoordinateDropout(model_dim=model_dim, dropout_prob=src_word_dropout_prob, **factory_mode)
            self.denoise_modules.append(self.src_word_dropout)
        if src_dropout > 0:
            self.src_dropout_nn = nn.Dropout(src_dropout)
            self.denoise_modules.append(self.src_dropout_nn)
        if tgt_dropout > 0:
            self.tgt_dropout_nn = nn.Dropout(tgt_dropout)
            self.fixation_modules.append(self.tgt_dropout_nn)
        if enh_features_dropout > 0:
            self.enh_features_dropout_nn = nn.Dropout(enh_features_dropout)
            self.denoise_modules.append(self.enh_features_dropout_nn)
        # INPUT PROCESSING
        self.time_dec_pe = PositionalEncoding(max_pos_dec, model_dim,**factory_mode)
        self.time_enc_pe = PositionalEncoding(max_pos_enc, model_dim,**factory_mode)
        if input_encoder == 'linear':
            self.enc_input_proj = nn.Linear(input_dim, model_dim, **factory_mode)
            self.denoise_modules.append(self.enc_input_proj)
            self.dec_input_proj = nn.Linear(input_dim, model_dim, **factory_mode)
            self.fixation_modules.append(self.dec_input_proj)
        elif input_encoder == 'fourier' or input_encoder == 'fourier_sum':
            self.enc_coords_pe = GaussianFourierPosEncoder(2, num_freq_bands, pos_enc_hidden_dim, model_dim, pos_enc_sigma, **factory_mode)
            self.enc_time_pe = GaussianFourierPosEncoder(1, num_freq_bands, pos_enc_hidden_dim, model_dim, pos_enc_sigma, **factory_mode)
            self.denoise_modules.append(self.enc_coords_pe)
            self.denoise_modules.append(self.enc_time_pe)
            self.dec_coords_pe = GaussianFourierPosEncoder(2, num_freq_bands, pos_enc_hidden_dim, model_dim, pos_enc_sigma, **factory_mode)
            self.dec_time_pe = GaussianFourierPosEncoder(1, num_freq_bands, pos_enc_hidden_dim, model_dim, pos_enc_sigma, **factory_mode)
            self.fixation_modules.append(self.dec_coords_pe)
            self.fixation_modules.append(self.dec_time_pe)
        elif input_encoder == 'nerf_fourier':
            self.enc_coords_pe = FourierPosEncoder(2, num_freq_bands, pos_enc_hidden_dim, model_dim, **factory_mode)
            self.enc_time_pe = FourierPosEncoder(1, num_freq_bands, pos_enc_hidden_dim, model_dim, **factory_mode)
            self.denoise_modules.append(self.enc_coords_pe)
            self.denoise_modules.append(self.enc_time_pe)
            self.dec_coords_pe = FourierPosEncoder(2, num_freq_bands, pos_enc_hidden_dim, model_dim, **factory_mode)
            self.dec_time_pe = FourierPosEncoder(1, num_freq_bands, pos_enc_hidden_dim, model_dim, **factory_mode)
            self.fixation_modules.append(self.dec_coords_pe)
            self.fixation_modules.append(self.dec_time_pe)
        elif input_encoder == 'fourier_concat':
            self.enc_inputs_pe = GaussianFourierPosEncoder(3, num_freq_bands, pos_enc_hidden_dim, model_dim, pos_enc_sigma, **factory_mode)
            self.dec_inputs_pe = GaussianFourierPosEncoder(3, num_freq_bands, pos_enc_hidden_dim, model_dim, pos_enc_sigma, **factory_mode)
            self.denoise_modules.append(self.enc_inputs_pe)
            self.fixation_modules.append(self.dec_inputs_pe)
        elif input_encoder == 'image_features_concat':
            self.enc_input_proj = nn.Linear(input_dim, model_dim, **factory_mode)
            self.denoise_modules.append(self.enc_input_proj)
            self.dec_input_proj = nn.Linear(input_dim, model_dim, **factory_mode)
            self.mix_image_features = nn.Linear(model_dim*2, model_dim, **factory_mode)
            self.fixation_modules.append(self.dec_input_proj)
            self.fixation_modules.append(self.mix_image_features)
        elif input_encoder == "shared_gaussian":
            self.pos_proj = GaussianFourierPosEncoder(2, num_freq_bands, pos_enc_hidden_dim, model_dim, pos_enc_sigma,input_encoder = input_encoder, patch_size = 16 ,**factory_mode)
            self.time_proj = GaussianFourierPosEncoder(1, num_freq_bands, pos_enc_hidden_dim, model_dim, pos_enc_sigma,input_encoder = input_encoder, patch_size = 16 ,**factory_mode)
            self.dur_proj = GaussianFourierPosEncoder(1, num_freq_bands, pos_enc_hidden_dim, model_dim, pos_enc_sigma,input_encoder = input_encoder, patch_size = 16 ,**factory_mode)
            self.denoise_modules.append(self.pos_proj)
            self.denoise_modules.append(self.time_proj)
            self.fixation_modules.append(self.dur_proj)
        elif input_encoder == "shared_gaussian_base":
            self.eye_pos_proj = GaussianFourierPosEncoder(2, num_freq_bands, pos_enc_hidden_dim, model_dim, pos_enc_sigma,input_encoder = input_encoder, patch_size = 16 ,**factory_mode)
            self.fix_pos_proj = GaussianFourierPosEncoder(2, num_freq_bands, pos_enc_hidden_dim, model_dim, pos_enc_sigma,input_encoder = input_encoder, patch_size = 16 ,base = self.eye_pos_proj.B,**factory_mode)
            self.img_pos_proj = GaussianFourierPosEncoder(2, num_freq_bands, pos_enc_hidden_dim, model_dim, pos_enc_sigma,input_encoder = input_encoder, patch_size = 16 ,base = self.eye_pos_proj.B,**factory_mode)
            self.time_proj = GaussianFourierPosEncoder(1, num_freq_bands, pos_enc_hidden_dim, model_dim, pos_enc_sigma,input_encoder = input_encoder, patch_size = 16 ,**factory_mode)
            self.dur_proj = GaussianFourierPosEncoder(1, num_freq_bands, pos_enc_hidden_dim, model_dim, pos_enc_sigma,input_encoder = input_encoder, patch_size = 16 ,**factory_mode)
            self.denoise_modules.append(self.eye_pos_proj)
            self.denoise_modules.append(self.time_proj)
            self.fixation_modules.append(self.fix_pos_proj)
            self.fixation_modules.append(self.img_pos_proj)
            self.fixation_modules.append(self.dur_proj)
        else:
            raise ValueError(f"Unsupported input_encoder: {input_encoder}")
        
        if image_encoder is not None:
            img_embed_dim = image_encoder.embed_dim
            patch_resolution = int((self.img_size / image_encoder.model.patch_size))
            self.patch_resolution = (patch_resolution, patch_resolution)
            print(self.patch_resolution)
            if img_embed_dim == model_dim:
                self.img_input_proj = nn.Identity()
            else:
                # self.img_input_proj = nn.Linear(img_embed_dim, model_dim, **factory_mode)
                self.img_input_proj = MLP(img_embed_dim,
                                           mlp_head_hidden_dim,
                                           model_dim,
                                           hidden_dropout_p = image_features_dropout,
                                           **factory_mode)
            self.fixation_modules.append(self.img_input_proj)
            if use_rope:
                self.rope_pos = RopePositionEmbedding(embed_dim = self.model_dim,
                                                      num_heads = self.n_heads,
                                                      base = image_encoder.model.rope_embed.base,
                                                      min_period = image_encoder.model.rope_embed.min_period,
                                                      max_period  = image_encoder.model.rope_embed.max_period,
                                                      normalize_coords  = image_encoder.model.rope_embed.normalize_coords,
                                                      shift_coords  = image_encoder.model.rope_embed.shift_coords,
                                                      jitter_coords  = image_encoder.model.rope_embed.jitter_coords,
                                                      rescale_coords  = image_encoder.model.rope_embed.rescale_coords ,
                                                      **factory_mode)
        # ENCODER
        if n_encoder > 0:
            path_layer = TransformerEncoder(model_dim = model_dim,
                                            total_dim = total_dim,
                                            n_heads = n_heads,
                                            ff_dim = ff_dim,
                                            dropout_p = eye_encoder_dropout,
                                            activation= activation,
                                            norm_first= norm_first,
                                            **factory_mode)
            self.path_encoder = _get_clones(path_layer,n_encoder) 
            for mod in self.path_encoder:
                self.denoise_modules.append(mod)

        if n_feature_enhancer > 0 and not (self.n_eye_decoder > 0):
            feature_enhancer_layer = FeatureEnhancer(model_dim = model_dim,
                                            total_dim = total_dim,
                                            n_heads = n_heads,
                                            ff_dim = ff_dim,
                                            dropout_p = dropout_p,
                                            activation= activation,
                                            norm_first= norm_first,
                                            **factory_mode)
            self.feature_enhancer = _get_clones(feature_enhancer_layer,n_feature_enhancer)
            for mod in self.feature_enhancer:
                self.denoise_modules.append(mod)
                
        if self.n_eye_decoder > 0:
            eye_decoder_layer = TransformerDecoder(model_dim = model_dim,
                                            total_dim = total_dim,
                                            n_heads = n_heads,
                                            ff_dim = ff_dim,
                                            dropout_p = eye_encoder_dropout,
                                            activation= activation,
                                            norm_first= norm_first,
                                            **factory_mode)
            self.eye_decoder = _get_clones(eye_decoder_layer,n_eye_decoder)
            for mod in self.eye_decoder:
                self.denoise_modules.append(mod)
                
            
        if self.n_adapter > 0:
            adapter_layer = TransformerEncoder(model_dim = model_dim,
                                        total_dim = total_dim,
                                        n_heads = n_heads,
                                        ff_dim = ff_dim,
                                        dropout_p = adapter_dropout,
                                        activation= activation,
                                        norm_first= norm_first,
                                        **factory_mode)
            self.adapter = _get_clones(adapter_layer,n_adapter) 
            for mod in self.adapter:
                self.fixation_modules.append(mod)
            if self.norm_first:
                self.adapter_norm = nn.LayerNorm(model_dim, eps = 1e-5, **factory_mode)
                self.fixation_modules.append(self.adapter_norm)
                    
                
        
        
        # DECODER
        decoder_layer = DoubleInputDecoder(model_dim = model_dim,
                                           total_dim = total_dim,
                                           n_heads = n_heads,
                                           ff_dim = ff_dim,
                                           dropout_p = dropout_p,
                                           activation= activation,
                                           norm_first= norm_first,
                                           use_kv_cache = use_kv_cache,
                                           **factory_mode)
        self.decoder = _get_clones(decoder_layer,n_decoder)
        for mod in self.decoder:
            self.fixation_modules.append(mod)
        if  norm_first:
            self.final_dec_norm = nn.LayerNorm(model_dim, eps = 1e-5, **factory_mode)
            self.fixation_modules.append(self.final_dec_norm)
            self.final_enc_norm = nn.LayerNorm(model_dim, eps = 1e-5, **factory_mode)
            self.denoise_modules.append(self.final_enc_norm)
            self.final_fenh_norm_src = nn.LayerNorm(model_dim, eps = 1e-5, **factory_mode)
            self.denoise_modules.append(self.final_fenh_norm_src)
            self.final_fenh_norm_image = nn.LayerNorm(model_dim, eps = 1e-5, **factory_mode)
            self.denoise_modules.append(self.final_fenh_norm_image)
            if self.mixed_image_features:
                self.final_fsrc_norm_image = nn.LayerNorm(model_dim, eps = 1e-5, **factory_mode)
                self.denoise_modules.append(self.final_fsrc_norm_image)
        # HEADS
        if head_type == 'mlp':
            self.regression_head = MLP(model_dim,
                                           mlp_head_hidden_dim,
                                           output_dim,
                                           hidden_dropout_p = reg_head_dropout,
                                           **factory_mode)
            self.end_head = MLP(model_dim,
                                     mlp_head_hidden_dim,
                                     1,
                                     hidden_dropout_p = end_dropout,
                                     **factory_mode)
            self.fixation_modules.append(self.regression_head)
            self.fixation_modules.append(self.end_head)
        elif head_type == 'linear':
            self.regression_head = nn.Sequential(
                nn.Dropout(reg_head_dropout),
                nn.Linear(model_dim, output_dim, **factory_mode)
            )
            self.end_head = nn.Sequential(
                nn.Dropout(end_dropout),
                nn.Linear(model_dim, 1, **factory_mode)
            )
            self.fixation_modules.append(self.regression_head)
            self.fixation_modules.append(self.end_head)
        elif head_type == 'multi_mlp':
            self.coord_head = MLP(model_dim,
                                           mlp_head_hidden_dim,
                                           2,
                                           hidden_dropout_p = reg_head_dropout,
                                           **factory_mode)
            self.dur_head = MLP(model_dim,
                                           mlp_head_hidden_dim,
                                           1,
                                           hidden_dropout_p = dur_head_dropout,
                                           **factory_mode)
            
            self.end_head = MLP(model_dim,
                                     mlp_head_hidden_dim,
                                     1,
                                     hidden_dropout_p = end_dropout,
                                     **factory_mode)
            self.fixation_modules.append(self.coord_head)
            self.fixation_modules.append(self.dur_head)
            self.fixation_modules.append(self.end_head)
            
        elif head_type == 'start_head':
            self.coord_head = MLP(model_dim,
                                           mlp_head_hidden_dim,
                                           2,
                                           hidden_dropout_p = reg_head_dropout,
                                           **factory_mode)
            
            self.start_head = MLP(model_dim,
                                           mlp_head_hidden_dim,
                                           2,
                                           hidden_dropout_p = reg_head_dropout,
                                           **factory_mode)
            self.dur_head = MLP(model_dim,
                                           mlp_head_hidden_dim,
                                           1,
                                           hidden_dropout_p = dur_head_dropout,
                                           **factory_mode)
            
            self.end_head = MLP(model_dim,
                                     mlp_head_hidden_dim,
                                     1,
                                     hidden_dropout_p = end_dropout,
                                     **factory_mode)
            self.fixation_modules.append(self.coord_head)
            self.fixation_modules.append(self.dur_head)
            self.fixation_modules.append(self.end_head)
        elif head_type == 'argmax_regressor':
            if image_encoder is None:
                raise ValueError("Image encoder is required for argmax regressor")
            self.regressor_head = MLP(model_dim,
                                           mlp_head_hidden_dim,
                                           model_dim,
                                           **factory_mode)
            
            self.argmax_regressor = ArgMaxRegressor(H = self.patch_resolution[0], W = self.patch_resolution[1], **factory_mode)
            self.dur_head = MLP(model_dim,
                                           mlp_head_hidden_dim,
                                           1,
                                           **factory_mode)
            
            self.end_head = MLP(model_dim,
                                     mlp_head_hidden_dim,
                                     1,
                                     hidden_dropout_p = end_dropout,
                                     **factory_mode)
            self.fixation_modules.append(self.regressor_head)
            self.fixation_modules.append(self.argmax_regressor)
            self.fixation_modules.append(self.dur_head)
            self.fixation_modules.append(self.end_head)
        elif head_type == 'heatmap':
            self.trajectory_heatmap_generator = TrajectoryHeatmapGenerator(model_dim,
                                                                           self.patch_resolution[0],
                                                                           self.patch_resolution[1],
                                                                           **factory_mode)
            self.dur_head = MLP(model_dim,
                                           mlp_head_hidden_dim,
                                           1,
                                           **factory_mode)
            
            self.end_head = MLP(model_dim,
                                     mlp_head_hidden_dim,
                                     1,
                                     hidden_dropout_p = end_dropout,
                                     **factory_mode)
            self.fixation_modules.append(self.trajectory_heatmap_generator)
            self.fixation_modules.append(self.dur_head)
            self.fixation_modules.append(self.end_head)
        else:
            raise ValueError(f"Unsupported head_type: {head_type}")
        
        # DENOISE HEADS
        if phases is not None and ('Denoise' in phases or 'Combined' in phases):
            # self.denoise_head = ResidualRegressor(model_dim, hidden_dropout_p = self.denoise_head_hidden_dropout, output_dropout_p = self.denoise_head_output_dropout, **factory_mode)
            self.denoise_head =  MLP(model_dim,
                                        mlp_head_hidden_dim,
                                        2,
                                        hidden_dropout_p = self.denoise_head_hidden_dropout,
                                        output_dropout_p = self.denoise_head_output_dropout,
                                        **factory_mode)
            self.denoise_modules.append(self.denoise_head)

    def get_key_name(self, model, module_list):
        # 1. Create a set of object IDs for your target list for O(1) lookup
        target_ids = {id(m) for m in module_list}
        found_names = []

        # 2. Iterate through every named module in the model
        for name, module in model.named_modules():
            # 3. Check if the current module's ID is in our target list
            if id(module) in target_ids:
                found_names.append(name)
                
        return found_names
    
    def load_encoder(self, checkpoint_path):
        encoder_keys = self.get_key_name(self, self.denoise_modules)
        """
        Loads only the parameters specified in encoder_keys_list from the checkpoint.
        """
        full_checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" in full_checkpoint:
            full_checkpoint = full_checkpoint["model_state_dict"]
        encoder_state_dict = {
            k: v for k, v in full_checkpoint.items() 
            if any(block_name in k for block_name in encoder_keys)
        }

        missing_keys, unexpected_keys = self.load_state_dict(encoder_state_dict, strict=False)

        print(f"✅ Loaded {len(encoder_state_dict)} encoder layers.")
        
        for expected_key in encoder_keys:
            if any(expected_key in k for k in missing_keys):
                print(f"⚠️ Warning: Expected block '{expected_key}' was NOT loaded.")

    def clear_kv_cache(self):
        for mod in self.decoder:
            mod.clear_kv_cache()
            
            
    def disable_kv_cache(self):
        self.use_kv_cache = False
        for mod in self.decoder:
            mod.disable_kv_cache()

    def param_summary(self):
        summ = f"""MixerModel Summary:
        Number of Encoder Layers: {self.n_encoder}
        Number of Decoder Layers: {self.n_decoder}
        Number of Feature Enhancer Layers: {self.n_feature_enhancer}
        Model Dimension: {self.model_dim}
        Number of Heads: {self.n_heads}
        Feed Forward Dimension: {self.ff_dim}
        Dropout Probability: {self.dropout_p}
        Norm First: {self.norm_first}
        Head Type: {self.head_type}
        Image Encoder: {self.image_encoder is not None}
        Device: {self.factory_mode['device']}
        Dtype: {self.factory_mode['dtype']}
        """
        if self.head_type == 'mlp' and self.mlp_head_hidden_dim is not None:
            resolved_dims = [dim for dim in self.mlp_head_hidden_dim]
            summ += f"        MLP Head Hidden Dimension: {resolved_dims}\n"
        if self.head_type == 'multi_mlp' and self.mlp_head_hidden_dim is not None:
            resolved_dims = [dim for dim in self.mlp_head_hidden_dim]
            summ += f"        Multi MLP Head Hidden Dimension: {resolved_dims}\n"
        if self.input_encoder == 'fourier' or self.input_encoder == 'fourier_concat' or self.input_encoder == 'fourier_sum':
            summ += f"        Fourier Input Encoder: True\n Number of Frequency Bands: {self.num_freq_bands}\n Sigma: {self.pos_enc_sigma}\n"
        elif self.input_encoder == 'linear':
            summ += f"        Linear Input Encoder: True\n"
        elif self.input_encoder == 'nerf_fourier':
            summ += f"        NeRF Fourier Input Encoder: True\n"
        elif self.input_encoder == 'image_features_concat':
            summ += f"        Image Features Concat Input Encoder: True\n"
        elif self.input_encoder == 'shared_gaussian':
            summ += f"        Sharded Gaussian Input Encoder: True\n"
        elif self.input_encoder == 'shared_gaussian_base':
            summ += f"        Shared Gaussian Base Input Encoder: True\n"
        else:
            raise ValueError(f"Unsupported input_encoder: {self.input_encoder}")
        return summ
    
    def set_phase(self, phase):
        self.phase = phase
        if phase == 'Denoise':
            for mod in self.fixation_modules:
                mod.requires_grad_(False)
            for mod in self.denoise_modules:
                mod.requires_grad_(True)
        elif phase == 'Fixation':
            for mod in self.denoise_modules:
                mod.requires_grad_(False)
            for mod in self.fixation_modules:
                mod.requires_grad_(True)
        elif phase == 'Combined':
            for mod in self.denoise_modules:
                mod.requires_grad_(True)
            for mod in self.fixation_modules:
                mod.requires_grad_(True)
    
    def input_encoding(self, src):
        if self.input_encoder == 'fourier' or self.input_encoder == 'fourier_sum' or self.input_encoder == 'nerf_fourier':
            enc_coords = src[:,:,:2]
            enc_time = src[:,:,2]
            enc_coords = self.enc_coords_pe(enc_coords)
            enc_time = self.enc_time_pe(enc_time)
            src = enc_coords + enc_time
            # add the time and coords 
        elif self.input_encoder == 'linear' or self.input_encoder == 'image_features_concat':
            # apply the linear projections
            src = self.enc_input_proj(src)
        elif self.input_encoder == 'fourier_concat':
            # apply the fourier encodings
            src = self.enc_inputs_pe(src)
        elif self.input_encoder == 'shared_gaussian':
            enc_coords = src[:,:,:2]
            enc_time = src[:,:,2]
            enc_coords = self.pos_proj(enc_coords)
            if self.src_word_dropout_prob > 0:
                enc_coords = self.src_word_dropout(enc_coords)
            enc_time = self.time_proj(enc_time)
            src = enc_coords + enc_time
        elif self.input_encoder == 'shared_gaussian_base':
            enc_coords = src[:,:,:2]
            enc_time = src[:,:,2]
            enc_coords = self.eye_pos_proj(enc_coords)
            if self.src_word_dropout_prob > 0:
                enc_coords = self.src_word_dropout(enc_coords)
            enc_time = self.time_proj(enc_time)
            src = enc_coords + enc_time
        else:
            raise ValueError(f"Unsupported input_encoder: {self.input_encoder}")
        return src
    
    def encode(self, src, image_src, src_mask, **kwargs):
        src_coords = src[:,:,:2].clone()
        
        src = self.input_encoding(src)
        enc_pe = self.time_enc_pe.pe.unsqueeze(0)
        src = src + enc_pe[:,:src.size()[1],:]
        if self.src_dropout > 0:
            src = self.src_dropout_nn(src)
        

        # encoding path
        if self.n_encoder > 0:
            for mod in self.path_encoder:
                src_rope = None
                if self.use_rope:
                    src_rope = self.rope_pos(traj_coords = src_coords)
                src = mod(src, src_mask, rope_pos = src_rope)
                
        if self.norm_first:
            src = self.final_enc_norm(src)
            
        # encoding images    
        if self.image_encoder is not None:
            image_src = self.image_encoder(image_src)
            image_src = self.img_input_proj(image_src)
            if self.input_encoder == 'shared_gaussian':
                # pos_enc [1,H*W,model_dim]
                pos_enc = self.pos_proj.forward_features().unsqueeze(0)
                prefix = image_src.size(1) - pos_enc.shape[1]
                image_src[:,prefix:,:] = image_src[:,prefix:,:] + pos_enc
            if self.input_encoder == 'shared_gaussian_base':
                pos_enc = self.img_pos_proj.forward_features().unsqueeze(0)
                prefix = image_src.size(1) - pos_enc.shape[1]
                image_src[:,prefix:,:] = image_src[:,prefix:,:] + pos_enc

            # enhancing features
            if self.n_feature_enhancer > 0 and not (self.n_eye_decoder > 0):
                
                img_enh = image_src
                for mod in self.feature_enhancer:
                    src_rope = None
                    image_rope = None
                    if self.use_rope:
                        src_rope, image_rope = self.rope_pos(traj_coords = src_coords, patch_res = self.patch_resolution)
                    src, img_enh = mod(src, img_enh, src1_mask = src_mask, src2_mask = None, src1_rope = src_rope, src2_rope = image_rope)
                if self.norm_first:
                    src = self.final_fenh_norm_src(src)
                    
            if self.mixed_image_features:
                if self.norm_first:
                    image_src = self.final_fsrc_norm_image(image_src)
                    img_enh = self.final_fenh_norm_image(img_enh)
                if self.enh_features_dropout > 0:
                    img_enh = self.enh_features_dropout_nn(img_enh)
                image_src = self.mix_enh_image_features(image_src, img_enh)
            elif self.use_enh_img_features:
                img_enh = self.final_fenh_norm_image(img_enh)
                if self.enh_features_dropout > 0:
                    img_enh = self.enh_features_dropout_nn(img_enh)
                image_src = img_enh
            else:
                image_src = self.final_fenh_norm_image(image_src)
            
            if self.n_adapter > 0:
                for mod in self.adapter:
                    image_src = mod(image_src, None)
                if self.norm_first:
                    image_src = self.adapter_norm(image_src)
            if self.n_eye_decoder > 0:
                for mod in self.eye_decoder:
                    for mod in self.eye_decoder:
                        src = mod(src, image_src, src_mask, None)
                if self.norm_first:
                    src = self.final_fenh_norm_src(src)
                
        self.src = src
        self.image_src = image_src
        self.src_coords = src_coords
    
    def decode_fixation(self, tgt, tgt_mask, src_mask, in_tgt = None, **kwargs):
        if in_tgt is not None:
            tgt = in_tgt
        src = self.src
        if self.use_denoised_coordinates:
            output = self.decode_denoise(**kwargs)
            den_src = output['denoise']
            true_src = kwargs['src'].clone()
            true_src[:,:,:2] = den_src[:,:,:2]
            src = self.input_encoding(true_src)
            
        image_src = self.image_src
        src_coords = self.src_coords
        
        if tgt is not None:
            tgt_coords = tgt[:,:,:2].clone()
        else:
            tgt_coords = None
        start = self.start_token.expand(src.size(0),-1,-1)
        if tgt is not None:
            if self.input_encoder == 'fourier' or self.input_encoder == 'fourier_sum' or self.input_encoder == 'nerf_fourier':
                dec_coords = tgt[:,:,:2]
                dec_dur = tgt[:,:,2]
                dec_coords = self.dec_coords_pe(dec_coords)
                dec_dur = self.dec_time_pe(dec_dur)
                tgt = dec_coords + dec_dur
                # add the time and coords 
            elif self.input_encoder == 'linear' or self.input_encoder == 'image_features_concat':
                # apply the linear projections
                tgt = self.dec_input_proj(tgt)
            elif self.input_encoder == 'fourier_concat':
                # apply the fourier encodings
                tgt = self.dec_inputs_pe(tgt)
            elif self.input_encoder == 'shared_gaussian':
                dec_coords = tgt[:,:,:2]
                dec_dur = tgt[:,:,2]
                dec_coords = self.pos_proj(dec_coords)
                dec_dur = self.dur_proj(dec_dur)
                tgt = dec_coords + dec_dur
            elif self.input_encoder == 'shared_gaussian_base':
                dec_coords = tgt[:,:,:2]
                dec_dur = tgt[:,:,2]
                dec_coords = self.fix_pos_proj(dec_coords)
                dec_dur = self.dur_proj(dec_dur)
                tgt = dec_coords + dec_dur
            else:
                raise ValueError(f"Unsupported input_encoder: {self.input_encoder}")
            # apply the order positional encodings
            if not self.use_kv_cache:
                tgt = torch.cat([start, tgt], dim = 1)
        else:
            tgt = start
        if self.word_dropout_prob > 0:
            tgt = self.word_dropout(tgt)
        dec_pe = self.time_dec_pe.pe.unsqueeze(0)
        if self.use_kv_cache:
            cached_input_count = self.decoder[0].get_cached_input_count()
            tgt = tgt + dec_pe[:,cached_input_count,:].unsqueeze(1)
        else:
            tgt = tgt + dec_pe[:,:tgt.size()[1],:]
        if self.tgt_dropout > 0:
            tgt = self.tgt_dropout_nn(tgt)

        if self.input_encoder == 'image_features_concat' and tgt_coords is not None:
            visual_tokens = image_src[:,1:,:]
            B = tgt_coords.size(0)
            coords = torch.floor(tgt_coords*16).long()
            coords[coords > 15] = 15
            visual_tokens = visual_tokens.view(B, self.patch_resolution[0], self.patch_resolution[1], self.model_dim)
            grid_h = coords[:, :, 0] 
            grid_w = coords[:, :, 1]
            batch_idx = torch.arange(B).unsqueeze(1).to(tgt_coords.device)
            selected = visual_tokens[batch_idx, grid_h, grid_w]
            start_token = tgt[:,0:1,:]
            tgt = torch.cat([start_token, self.mix_image_features(torch.cat([tgt[:,1:,:], selected], dim = -1))], dim = 1)
        # decoding
        output = tgt
        for mod in self.decoder:
            src_rope = tgt_rope = image_rope = None
            if self.use_rope and tgt_coords is not None:
                [src_rope, tgt_rope], image_rope = self.rope_pos(traj_coords = [src_coords, tgt_coords], patch_res = self.patch_resolution)
            output = mod(output, image_src,src , tgt_mask, src_mask, src_rope = tgt_rope, mem1_rope = image_rope, mem2_rope = src_rope)

        if self.norm_first:
            output = self.final_dec_norm(output)
        # output heads
        if self.head_type == 'multi_mlp':
            coord_out = self.coord_head(output)
            dur_out = self.dur_head(output)
            cls_out = self.end_head(output)
            return {'coord': coord_out, 'dur': dur_out, 'cls': cls_out}
        elif self.head_type == 'argmax_regressor':
            reg_out = self.regressor_head(output)
            coord_out = self.argmax_regressor(reg_out, image_src)
            dur_out = self.dur_head(output)
            cls_out = self.end_head(output)
            return {'coord': coord_out, 'dur': dur_out, 'cls': cls_out}
        elif self.head_type == 'mlp' or self.head_type == 'linear':
            reg_out = self.regression_head(output)
            cls_out = self.end_head(output)
            return {'reg': reg_out, 'cls': cls_out}
        elif self.head_type == 'start_head':
            start_out = self.start_head(output[:,0:1,:])
            tail_out = self.coord_head(output[:,1:,:])
            coord_out = torch.cat([start_out, tail_out], dim = 1)
            dur_out = self.dur_head(output)
            cls_out = self.end_head(output)
            return {'start': start_out, 'coord': coord_out, 'dur': dur_out, 'cls': cls_out}
        elif self.head_type == 'heatmap':
            heatmaps = self.trajectory_heatmap_generator(image_src, output)
            dur_out = self.dur_head(output)
            cls_out = self.end_head(output)
            return {'heatmaps': heatmaps, 'dur': dur_out, 'cls': cls_out}
        
    def decode_denoise(self, **kwargs):
        if hasattr(self, 'denoise_head'):
            src = self.src
            output = self.denoise_head(src, **kwargs)
            return {'denoise': output}
        else:
            return {}
        
    def set_scheduled_sampling(self, scheduled_sampling):
        self.scheduled_sampling = scheduled_sampling
        self.scheduled_sampling.set_model(self)
        
    def forward(self, **kwargs):
        # src, tgt shape (B,L,F)
        if self.scheduled_sampling is not None and ('pass_sampler' not in kwargs or kwargs['pass_sampler'] is False):
            return self.scheduled_sampling(**kwargs)
        if 'pass_sampler' not in kwargs or kwargs['pass_sampler'] is False:
            self.encode(**kwargs)
        if self.phase == 'Denoise':
            return self.decode_denoise(**kwargs)
        elif self.phase == 'Fixation':
            return self.decode_fixation(**kwargs)
        elif self.phase == 'Combined':
            denoise_output = self.decode_denoise(**kwargs)  
            fixation_output = self.decode_fixation(**kwargs)
            return {**denoise_output, **fixation_output}
        return self.decode_fixation(**kwargs)
        
    