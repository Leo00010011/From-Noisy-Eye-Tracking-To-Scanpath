import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np

class MultiHeadedAttention(nn.Module):
    def __init__(self,model_dim, total_dim ,n_heads, is_self_attention = False, is_causal = False, device = 'cpu', dtype = torch.float32):
        super().__init__()
        self.model_dim = model_dim
        self.total_dim = total_dim
        self.n_heads = n_heads
        self.is_self_attention = is_self_attention
        self.is_causal = is_causal
        self.head_dim = total_dim//n_heads
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        if is_self_attention:
            self.proj_in = nn.Linear(model_dim,total_dim*3, bias = False, **factory_kwargs)
        else:
            self.proj_q = nn.Linear(model_dim,total_dim, bias = False, **factory_kwargs )
            self.proj_kv = nn.Linear(model_dim, total_dim * 2, bias = False, **factory_kwargs )
        self.proj_out = nn.Linear(total_dim,model_dim, bias = False, **factory_kwargs )
    
    def forward(self,query, key = None):
        # in_proj
        if self.is_self_attention:
            result = self.proj_in(query)
            query, key, value = torch.chunk(result, 3,dim = -1)
        else:
            query = self.proj_q(query)
            result = self.proj_kv(key)
            key, value = torch.chunk(result, 2,dim = -1)
        # reshape
        # (B,L_seq, total_dim) -> (B, L_seq, n_heads, head_dim) -> (B, n_heads, L_seq, head_dim)
        query = query.unflatten(-1,[self.n_heads, self.head_dim]).transpose(1,2).contiguous()
        key = key.unflatten(-1,[self.n_heads, self.head_dim]).transpose(1,2).contiguous()
        value = value.unflatten(-1,[self.n_heads, self.head_dim]).transpose(1,2).contiguous()
        # scaled dot product
        attn_output = F.scaled_dot_product_attention(query, key, value,is_causal= self.is_causal)
        # (B, n_heads, L_seq, head_dims)
        attn_output = attn_output.transpose(1,2).contiguous().flatten(-2)
        # (B, L_seq, total)
        attn_output = self.proj_out(attn_output)
        return attn_output 

class TransformerEncoder(nn.Module):
    def __init__(self,model_dim = 1024, total_dim = 1024, n_heads = 8, ff_dim = 2048,dropout_p = .1,activation = F.relu,eps = 1e-5,norm_first = False, device = 'cpu', dtype = torch.float32):
        super().__init__()
        self.model_dim = model_dim
        self.total_dim = total_dim
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.dropout_p = dropout_p
        self.eps = eps
        self.activation = activation
        self.norm_first = norm_first
        factory_kwargs = {'device': device, 'dtype': dtype}
        # self_attention
        self.self_attn = MultiHeadedAttention(model_dim,
                                             total_dim,
                                             n_heads,
                                             is_self_attention=True,
                                             is_causal= False,
                                             **factory_kwargs)
        self.norm1 = nn.LayerNorm(model_dim,eps = eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout_p)
        # ff
        self.linear1 = nn.Linear(model_dim, ff_dim, **factory_kwargs)
        self.dropout2 = nn.Dropout(dropout_p)
        self.linear2 = nn.Linear(ff_dim, model_dim, **factory_kwargs)
        self.norm2 = nn.LayerNorm(model_dim, eps = eps, **factory_kwargs)
        self.dropout3 = nn.Dropout(dropout_p)
    
    def __self_attention(self, x):
        return self.dropout1(self.self_attn(x))

    def __feed_forward(self, x):
        return self.dropout3(self.linear2(self.dropout2(self.activation(self.linear1(x)))))

    def forward(self, src):
        x = src
        if self.norm_first:
            x = x + self.__self_attention(self.norm1(x))
            x = x + self.__feed_forward(self.norm2(x))
        else:
            x = self.norm1(x + self.__self_attention(x))
            x = self.norm2(x + self.__feed_forward(x))
        return x
    


class TransformerDecoder(nn.Module):
    def __init__(self,model_dim = 1024, total_dim = 1024, n_heads = 8, ff_dim = 2048,dropout_p = .1,activation = F.relu,eps = 1e-5,norm_first = False, device = 'cpu', dtype = torch.float32):
        super().__init__()
        self.model_dim = model_dim
        self.total_dim = total_dim
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.dropout_p = dropout_p
        self.eps = eps
        self.activation = activation
        self.norm_first = norm_first
        factory_kwargs = {'device': device, 'dtype': dtype}
        # sa
        self.self_attn = MultiHeadedAttention(model_dim,
                                             total_dim,
                                             n_heads,
                                             is_self_attention=True,
                                             is_causal= True,
                                             **factory_kwargs)
        self.norm1 = nn.LayerNorm(model_dim,eps = eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout_p)
        # ca
        self.cross_attn = MultiHeadedAttention(model_dim,
                                             total_dim,
                                             n_heads,
                                             is_self_attention=False,
                                             is_causal= False,
                                             **factory_kwargs)
        self.norm2 = nn.LayerNorm(model_dim, eps = eps, **factory_kwargs)
        self.dropout2 = nn.Dropout(dropout_p)
        # ff
        self.linear1 = nn.Linear(model_dim, ff_dim, **factory_kwargs)
        self.dropout3 = nn.Dropout(dropout_p)
        self.linear2 = nn.Linear(ff_dim, model_dim, **factory_kwargs)
        self.norm3 = nn.LayerNorm(model_dim,eps = eps, **factory_kwargs)
        self.dropout4 = nn.Dropout(dropout_p)

    def __self_attention(self, x):
        return self.dropout1(self.self_attn(x))

    def __cross_attention(self, src, mem):
        return self.dropout2(self.cross_attn(src,mem))
    
    def __feed_forward(self, x):
        return self.dropout4(self.linear2(self.dropout3(self.activation(self.linear1(x)))))

    
    def forward(self, src, mem):
        x = src
        if self.norm_first:
            x = x + self.__self_attention(self.norm1(x))
            x = x + self.__cross_attention(self.norm2(x),mem)
            x = x + self.__feed_forward(self.norm3(x))
        else:
            x = self.norm1(x + self.__self_attention(x))
            x = self.norm2(x + self.__cross_attention(x,mem))
            x = self.norm3(x + self.__feed_forward(x))

        return x


class PositionalEncoding:
    def __init__(self, max_pos, model_dim, device = 'cpu', dtype = torch.float32):
        self.max_pos = max_pos
        self.model_dim = model_dim

        pe = np.empty((max_pos, model_dim))
        position = np.arange(max_pos)[:,np.newaxis]
        div = np.exp(np.arange(0,model_dim,2)*-(np.log(10000)/model_dim))
        pe[:,0::2] = np.sin(position*div)
        pe[:,1::2] = np.cos(position*div)
        self.pe = torch.Tensor(pe, device = device)

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
                       device = 'cpu',
                       dtype = torch.float32):
        super().__init__()
        factory_mode = {'device':device, 'dtype': dtype}
        self.model_dim = model_dim
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.dropout_p = dropout_p
        self.n_encoder = n_encoder
        self.n_decoder = n_decoder

        # special token
        self.start_token = nn.Parameter(torch.randn(1,model_dim,**factory_mode))
        # input processing
        self.enc_input_proj = nn.Linear(input_dim, model_dim, **factory_mode)
        self.dec_input_proj = nn.Linear(input_dim, model_dim, **factory_mode)
        self.enc_pe = PositionalEncoding(max_pos_dec, model_dim,**factory_mode)
        self.dec_pe = PositionalEncoding(max_pos_enc, model_dim,**factory_mode)
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
        # output
        self.regression_head = nn.Linear(model_dim, output_dim,**factory_mode)
        self.end_head = nn.Linear(model_dim,1,**factory_mode)

    def forward(self,src, tgt):

        # src, tgt shape (B,L,F)
        src = self.enc_input_proj(src)
        tgt = self.dec_input_proj(tgt)
        # add the start to tgt
        # sum up the positional encodings (max_pos, model_dim) -> (L, model_dim)
        enc_pe = self.enc_pe.pe
        dec_pe = self.dec_pe.pe
        start = self.start_token
        src = torch.nested.nested_tensor([src[i] + enc_pe[:src[i].size()[0],:] 
                                          for i in range(src.size()[0])],layout=torch.jagged)
        tgt = torch.nested.nested_tensor([torch.cat([start ,(tgt[i] + dec_pe[:tgt[i].size()[0],:])], dim = 0) 
                                          for i in range(tgt.size()[0])],layout=torch.jagged)

        memory = src
        for mod in self.encoder:
            memory = mod(memory)
        
        output = tgt
        for mod in self.decoder:
            output = mod(output, memory)

        reg_out = self.regression_head(output)
        cls_out = self.end_head(output)
        return reg_out, cls_out
    