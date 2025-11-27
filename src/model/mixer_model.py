import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from src.model.blocks import TransformerEncoder, DoubleInputDecoder, MLP, FeatureEnhancer
from src.model.pos_encoders import PositionalEncoding

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MixerModel(nn.Module):
    def __init__(self, n_encoder,
                       n_decoder,
                       input_dim = 3,
                       output_dim = 3,
                       model_dim = 1024,
                       total_dim = 1024,
                       n_heads = 8,
                       ff_dim = 2048,
                       dropout_p = 0.1,
                       max_pos_enc = 8,
                       max_pos_dec = 4,
                       activation = F.relu,
                       norm_first = False ,
                       head_type = None,
                       mlp_head_hidden_dim = None,
                       device = 'cpu',
                       image_encoder = None,
                       n_feature_enhancer = 1,
                       image_dim = None,
                       dtype = torch.float32):
        super().__init__()
        factory_mode = {'device':device, 'dtype': dtype}
        self.name = 'MixerModel'
        self.model_dim = model_dim
        self.n_heads = n_heads
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
        # special token
        self.start_token = nn.Parameter(torch.randn(1,1,model_dim,**factory_mode))
        # input processing
        self.enc_input_proj = nn.Linear(input_dim, model_dim, **factory_mode)
        self.dec_input_proj = nn.Linear(input_dim, model_dim, **factory_mode)
        if image_encoder is not None:
            if image_dim is None:
                raise ValueError("image_dim must be provided if image_encoder is used")
            if image_dim == model_dim:
                self.img_input_proj = nn.Identity()
            else:
                self.img_input_proj = nn.Linear(image_dim, model_dim, **factory_mode)
        
        self.enc_pe = PositionalEncoding(max_pos_enc, model_dim,**factory_mode)
        self.dec_pe = PositionalEncoding(max_pos_dec, model_dim,**factory_mode)
        # encoding
        path_layer = TransformerEncoder(model_dim = model_dim,
                                           total_dim = total_dim,
                                           n_heads = n_heads,
                                           ff_dim = ff_dim,
                                           dropout_p = dropout_p,
                                           activation= activation,
                                           norm_first= norm_first,
                                           **factory_mode)
        self.path_encoder = _get_clones(path_layer,n_encoder) 
        
        feature_enhancer_layer = FeatureEnhancer(model_dim = model_dim,
                                           total_dim = total_dim,
                                           n_heads = n_heads,
                                           ff_dim = ff_dim,
                                           dropout_p = dropout_p,
                                           activation= activation,
                                           norm_first= norm_first,
                                           **factory_mode)
        self.feature_enhancer = _get_clones(feature_enhancer_layer,n_feature_enhancer)
        
        # decoding
        decoder_layer = DoubleInputDecoder(model_dim = model_dim,
                                           total_dim = total_dim,
                                           n_heads = n_heads,
                                           ff_dim = ff_dim,
                                           dropout_p = dropout_p,
                                           activation= activation,
                                           norm_first= norm_first,
                                           **factory_mode)
        self.decoder = _get_clones(decoder_layer,n_decoder)
        
        if  norm_first:
            self.final_dec_norm = nn.LayerNorm(model_dim, eps = 1e-5, **factory_mode)
            self.final_enc_norm = nn.LayerNorm(model_dim, eps = 1e-5, **factory_mode)
        if head_type == 'mlp':
            self.regression_head = MLP(model_dim,
                                           mlp_head_hidden_dim,
                                           output_dim,
                                           **factory_mode)
            self.end_head = MLP(model_dim,
                                     mlp_head_hidden_dim,
                                     1,
                                     **factory_mode)
        elif head_type == 'linear':
            self.regression_head = nn.Linear(model_dim, output_dim,**factory_mode)
            self.end_head = nn.Linear(model_dim,1,**factory_mode)
        else:
            raise ValueError(f"Unsupported head_type: {head_type}")

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
        return summ

    def forward(self, src, image_src, tgt, src_mask=None, tgt_mask=None, **kwargs):
        # src, tgt shape (B,L,F)
        src = self.enc_input_proj(src)
        tgt = self.dec_input_proj(tgt)
        # sum up the positional encodings (max_pos, model_dim) -> (L, model_dim)
        enc_pe = self.enc_pe.pe.unsqueeze(0)
        dec_pe = self.dec_pe.pe.unsqueeze(0)
        start = self.start_token.expand(tgt.size(0),-1,-1)
        src = src + enc_pe[:,:src.size()[1],:]
        # add the start to tgt
        tgt = torch.cat([start, tgt], dim = 1)
        tgt = tgt + dec_pe[:,:tgt.size()[1],:]
        # encoding path
        for mod in self.path_encoder:
            src = mod(src, src_mask)
        if self.norm_first:
            src = self.final_enc_norm(src)
            
        # encoding images    
        if self.image_encoder is not None:
            image_src = self.image_encoder(image_src)
            image_src = self.img_input_proj(image_src)
            
        # decoding
        output = tgt
        for mod in self.decoder:
            output = mod(output, src, image_src, tgt_mask, src_mask, None)

        if self.norm_first:
            output = self.final_dec_norm(output)
        # output heads
        reg_out = self.regression_head(output)
        cls_out = self.end_head(output)
        return {'reg': reg_out, 'cls': cls_out}
    