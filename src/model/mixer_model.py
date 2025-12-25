import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from src.model.blocks import TransformerEncoder, DoubleInputDecoder, MLP, FeatureEnhancer, ArgMaxRegressor, LearnableCoordinateDropout, ResidualRegressor
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
                       image_dim = None,
                       pos_enc_sigma = 1.0,
                       use_rope = False,
                       word_dropout_prob = 0.3,
                       phases = None,
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
        self.use_enh_img_features = use_enh_img_features
        self.word_dropout_prob = word_dropout_prob
        self.phase = None
        self.denoise_modules = []
        self.fixation_modules = []
        # SPECIAL TOKENS
        self.start_token = nn.Parameter(torch.randn(1,1,model_dim,**factory_mode))
        self.fixation_modules.append(self.start_token)
        if word_dropout_prob > 0:
            self.word_dropout = LearnableCoordinateDropout(model_dim=model_dim, dropout_prob=word_dropout_prob, **factory_mode)
            self.fixation_modules.append(self.word_dropout)
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
            self.denoise_modules.append(self.img_input_proj)
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
        
        path_layer = TransformerEncoder(model_dim = model_dim,
                                           total_dim = total_dim,
                                           n_heads = n_heads,
                                           ff_dim = ff_dim,
                                           dropout_p = dropout_p,
                                           activation= activation,
                                           norm_first= norm_first,
                                           **factory_mode)
        self.path_encoder = _get_clones(path_layer,n_encoder) 
        for mod in self.path_encoder:
            self.denoise_modules.append(mod)
        if n_feature_enhancer > 0:
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
        
        # DECODER
        decoder_layer = DoubleInputDecoder(model_dim = model_dim,
                                           total_dim = total_dim,
                                           n_heads = n_heads,
                                           ff_dim = ff_dim,
                                           dropout_p = dropout_p,
                                           activation= activation,
                                           norm_first= norm_first,
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
        
        # HEADS
        if head_type == 'mlp':
            self.regression_head = MLP(model_dim,
                                           mlp_head_hidden_dim,
                                           output_dim,
                                           **factory_mode)
            self.end_head = MLP(model_dim,
                                     mlp_head_hidden_dim,
                                     1,
                                     **factory_mode)
            self.fixation_modules.append(self.regression_head)
            self.fixation_modules.append(self.end_head)
        elif head_type == 'linear':
            self.regression_head = nn.Linear(model_dim, output_dim,**factory_mode)
            self.end_head = nn.Linear(model_dim,1,**factory_mode)
            self.fixation_modules.append(self.regression_head)
            self.fixation_modules.append(self.end_head)
        elif head_type == 'multi_mlp':
            self.coord_head = MLP(model_dim,
                                           mlp_head_hidden_dim,
                                           2,
                                           **factory_mode)
            self.dur_head = MLP(model_dim,
                                           mlp_head_hidden_dim,
                                           1,
                                           **factory_mode)
            
            self.end_head = MLP(model_dim,
                                     mlp_head_hidden_dim,
                                     1,
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
                                     **factory_mode)
            self.fixation_modules.append(self.regressor_head)
            self.fixation_modules.append(self.argmax_regressor)
            self.fixation_modules.append(self.dur_head)
            self.fixation_modules.append(self.end_head)
        else:
            raise ValueError(f"Unsupported head_type: {head_type}")
        
        # DENOISE HEADS
        if phases is not None and 'Denoise' in phases:
            self.denoise_head = ResidualRegressor(model_dim)
            # self.denoise_head =  MLP(model_dim,
            #                     mlp_head_hidden_dim,
            #                     2,
            #                     **factory_mode)
            self.denoise_modules.append(self.denoise_head)

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
        else:
            raise ValueError(f"Unsupported input_encoder: {self.input_encoder}")
        return summ
    
    def set_phase(self, phase):
        self.phase = phase
        if phase == 'Denoise':
            for mod in self.denoise_modules:
                mod.requires_grad_(True)
            for mod in self.fixation_modules:
                mod.requires_grad_(False)
        elif phase == 'Fixation':
            for mod in self.fixation_modules:
                mod.requires_grad_(True)
            for mod in self.denoise_modules:
                mod.requires_grad_(False)
        elif phase == 'Combined':
            for mod in self.fixation_modules:
                mod.requires_grad_(True)
            for mod in self.denoise_modules:
                mod.requires_grad_(True)
    
    def encode(self, src, image_src, src_mask, **kwargs):
        src_coords = src[:,:,:2].clone()
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
            enc_time = self.time_proj(enc_time)
            src = enc_coords + enc_time
        else:
            raise ValueError(f"Unsupported input_encoder: {self.input_encoder}")
        
        enc_pe = self.time_enc_pe.pe.unsqueeze(0)
        src = src + enc_pe[:,:src.size()[1],:]
        
        # encoding path
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
            if self.input_encoder == 'sharded_gaussian':
                # pos_enc [1,H*W,model_dim]
                pos_enc = self.pos_proj.forward_features().unsqueeze(0)
                prefix = image_src.size(1) - pos_enc.shape[0]
                image_src[:,prefix:,:] = image_src[:,prefix:,:] + pos_enc

            # enhancing features
            if self.n_feature_enhancer > 0:
                
                img_enh = image_src
                for mod in self.feature_enhancer:
                    src_rope = None
                    image_rope = None
                    if self.use_rope:
                        src_rope, image_rope = self.rope_pos(traj_coords = src_coords, patch_res = self.patch_resolution)
                    src, img_enh = mod(src, img_enh, src1_mask = src_mask, src2_mask = None, src1_rope = src_rope, src2_rope = image_rope)
                if self.norm_first:
                    src = self.final_fenh_norm_src(src)
            if self.use_enh_img_features:
                image_src = img_enh
            if self.norm_first:
                image_src = self.final_fenh_norm_image(image_src)
                
        self.src = src
        self.image_src = image_src
        self.src_coords = src_coords
    
    def decode_fixation(self, tgt, tgt_mask, src_mask, **kwargs):
        src = self.src
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
            else:
                raise ValueError(f"Unsupported input_encoder: {self.input_encoder}")
            # apply the order positional encodings
            tgt = torch.cat([start, tgt], dim = 1)
        else:
            tgt = start
        if self.word_dropout_prob > 0:
            tgt = self.word_dropout(tgt)
        dec_pe = self.time_dec_pe.pe.unsqueeze(0)
        tgt = tgt + dec_pe[:,:tgt.size()[1],:]
        
        
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
        
    def decode_denoise(self, **kwargs):
        src = self.src
        
        output = self.denoise_head(src, **kwargs)
        return {'denoise': output}
        
        
    def forward(self, **kwargs):
        # src, tgt shape (B,L,F)
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
        
    