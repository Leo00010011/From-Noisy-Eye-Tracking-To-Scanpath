import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from src.model.blocks import TransformerEncoder, TransformerDecoder, MLP
from src.model.pos_encoders import PositionalEncoding

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
                       norm_first = False ,
                       head_type = None,
                       mlp_head_hidden_dim = None,
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
        # special token
        self.start_token = nn.Parameter(torch.randn(1,1,model_dim,**factory_mode))
        # input processing
        self.enc_input_proj = nn.Linear(input_dim, model_dim, **factory_mode)
        self.dec_input_proj = nn.Linear(input_dim, model_dim, **factory_mode)
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
        return summ

    def forward(self,src, tgt, src_mask = None, tgt_mask = None, **kwargs):

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
        # encoding
        memory = src
        for mod in self.encoder:
            memory = mod(memory, src_mask)
        if self.norm_first:
            memory = self.final_enc_norm(memory)
        # decoding
        output = tgt
        for mod in self.decoder:
            output = mod(output, memory, tgt_mask, src_mask)

        if self.norm_first:
            output = self.final_dec_norm(output)
        # output heads
        reg_out = self.regression_head(output)
        cls_out = self.end_head(output)
        return {'reg': reg_out, 'cls': cls_out}
    