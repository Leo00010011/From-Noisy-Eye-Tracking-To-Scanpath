import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from src.model.blocks import TransformerEncoder, TransformerDecoder, MLP
from src.model.pos_encoders import PositionalEncoding, GaussianFourierPosEncoder

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class PathModel(nn.Module):
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
                       input_encoder = 'linear',
                       norm_first = False ,
                       head_type = None,
                       mlp_head_hidden_dim = None,
                       src_dropout = 0,
                       tgt_dropout = 0,
                       device = 'cpu',
                       dtype = torch.float32):
        super().__init__()
        factory_mode = {'device':device, 'dtype': dtype}
        self.name = 'PathModel'
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
        self.input_encoder = input_encoder
        self.src_dropout = src_dropout
        self.tgt_dropout = tgt_dropout
        # special token
        self.start_token = nn.Parameter(torch.randn(1,1,model_dim,**factory_mode))
        if src_dropout > 0:
            self.src_dropout_nn = nn.Dropout(src_dropout)
        if tgt_dropout > 0:
            self.tgt_dropout_nn = nn.Dropout(tgt_dropout)
        # input processing
        if input_encoder == 'linear':
            self.enc_input_proj = nn.Linear(input_dim, model_dim, **factory_mode)
            self.dec_input_proj = nn.Linear(input_dim, model_dim, **factory_mode)
        elif input_encoder == "shared_gaussian":
            self.pos_proj = GaussianFourierPosEncoder(2, 15, model_dim//2, model_dim, 1,input_encoder = input_encoder, patch_size = 16 ,**factory_mode)
            self.time_proj = GaussianFourierPosEncoder(1, 15, model_dim//2, model_dim, 1,input_encoder = input_encoder, patch_size = 16 ,**factory_mode)
            self.dur_proj = GaussianFourierPosEncoder(1, 15, model_dim//2, model_dim, 1,input_encoder = input_encoder, patch_size = 16 ,**factory_mode)
        else:
            raise ValueError(f"Unsupported input_encoder: {input_encoder}")
        self.enc_pe = PositionalEncoding(max_pos_enc, model_dim,**factory_mode)
        self.dec_pe = PositionalEncoding(max_pos_dec, model_dim,**factory_mode)
        # encoding
        encoder_layer = TransformerEncoder(model_dim = model_dim,
                                           total_dim = total_dim,
                                           n_heads = n_heads,
                                           ff_dim = ff_dim,
                                           dropout_p = dropout_p,
                                           activation= activation,
                                           norm_first= norm_first,
                                           **factory_mode)
        self.encoder = _get_clones(encoder_layer,n_encoder)
        # decoding
        decoder_layer = TransformerDecoder(model_dim = model_dim,
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
                                           model_dim//2 if input_encoder == 'shared_gaussian' else mlp_head_hidden_dim,
                                           output_dim,
                                           hidden_dropout_p = 0.1,
                                           output_dropout_p = 0.1,
                                           include_dropout = input_encoder == 'shared_gaussian',
                                           **factory_mode)
            self.end_head = MLP(model_dim,
                                     model_dim//2 if input_encoder == 'shared_gaussian' else mlp_head_hidden_dim,
                                     1,
                                     include_dropout = False,
                                     **factory_mode)
        elif head_type == 'linear':
            self.regression_head = nn.Linear(model_dim, output_dim,**factory_mode)
            self.end_head = nn.Linear(model_dim,1,**factory_mode)
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
        else:
            raise ValueError(f"Unsupported head_type: {head_type}")

    def param_summary(self):
        # When iterating, each element is automatically resolved
        resolved_dims = [dim for dim in self.mlp_head_hidden_dim]
        summ = """PathModel Summary:
        Number of Encoder Layers: {}
        Number of Decoder Layers: {}
        Model Dimension: {}
        Number of Heads: {}
        Feed Forward Dimension: {}
        Dropout Probability: {}
        Head Type: {}
        """.format(self.n_encoder,
                   self.n_decoder,
                   self.model_dim,
                   self.n_heads,
                   self.ff_dim,
                   self.dropout_p,
                   self.head_type)
        if self.head_type == 'mlp':
            summ += f"    MLP Head Hidden Dimension: {resolved_dims}\n"
        elif self.head_type == 'multi_mlp':
            summ += f"    MultiMLP Head Hidden Dimension: {resolved_dims}\n"
        return summ
    def set_phase(self, phase):
        return

    def encode(self, src, src_mask, **kwargs):
         # src, tgt shape (B,L,F)
        if self.input_encoder == 'linear':
            src = self.enc_input_proj(src)
        elif self.input_encoder == 'shared_gaussian':
            enc_coords = src[:,:,:2]
            enc_time = src[:,:,2]
            enc_coords = self.pos_proj(enc_coords)
            enc_time = self.time_proj(enc_time)
            src = enc_coords + enc_time
        else:
            raise ValueError(f"Unsupported input_encoder: {self.input_encoder}")
        # add the start to tgt
        
        if self.src_dropout > 0:
            src = self.src_dropout_nn(src)
        # sum up the positional encodings (max_pos, model_dim) -> (L, model_dim)
        enc_pe = self.enc_pe.pe.unsqueeze(0)
        src = src + enc_pe[:,:src.size()[1],:]
      
        # encoding
        memory = src
        for mod in self.encoder:
            memory = mod(memory, src_mask)
        if self.norm_first:
            memory = self.final_enc_norm(memory)
        self.memory = memory
        self.src_mask = src_mask
        
    def decode_fixation(self, tgt, tgt_mask, src_mask, **kwargs):
        memory = self.memory
        src_mask = self.src_mask
        start = self.start_token.expand(memory.size(0),-1,-1)
        if tgt is not None:
            if self.input_encoder == 'linear':
                tgt = self.dec_input_proj(tgt)
            elif self.input_encoder == 'shared_gaussian':
                dec_coords = tgt[:,:,:2]
                dec_dur = tgt[:,:,2]
                dec_coords = self.pos_proj(dec_coords)
                dec_dur = self.dur_proj(dec_dur)
                tgt = dec_coords + dec_dur
            else:
                raise ValueError(f"Unsupported input_encoder: {self.input_encoder}")
            tgt = torch.cat([start, tgt], dim = 1)
        else:
            tgt = start
        if self.tgt_dropout > 0:
            tgt = self.tgt_dropout_nn(tgt)
        dec_pe = self.dec_pe.pe.unsqueeze(0)
        tgt = tgt + dec_pe[:,:tgt.size()[1],:]
        # decoding
        output = tgt
        for mod in self.decoder:
            output = mod(output, memory, tgt_mask, src_mask)

        if self.norm_first:
            output = self.final_dec_norm(output)
        # output heads
        if self.head_type == 'linear' or self.head_type == 'mlp':
            reg_out = self.regression_head(output)
            cls_out = self.end_head(output)
            return {'reg': reg_out, 'cls': cls_out}
        elif self.head_type == 'multi_mlp':
            coord_out = self.coord_head(output)
            dur_out = self.dur_head(output)
            cls_out = self.end_head(output)
            return {'coord': coord_out, 'dur': dur_out, 'cls': cls_out}

    def forward(self, **kwargs):
        self.encode(**kwargs)
        return self.decode_fixation(**kwargs)
       
    