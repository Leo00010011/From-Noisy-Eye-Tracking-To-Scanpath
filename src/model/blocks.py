import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, out_dim, hidden_dropout_p = None, output_dropout_p = None,include_dropout = True, device='cpu', dtype=torch.float32):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        if hidden_dropout_p is None:
            hidden_dropout_p = 0
        if output_dropout_p is None:
            output_dropout_p = 0
        # Convert single int to list for consistency
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim, **factory_kwargs))
            layers.append(nn.ReLU())
            if include_dropout:
                layers.append(nn.Dropout(hidden_dropout_p))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, out_dim, **factory_kwargs))
        if include_dropout:
            layers.append(nn.Dropout(output_dropout_p))
        self.head = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.head(x)
# RoPE-related functions:
def rope_rotate_half(x: Tensor) -> Tensor:
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x3 -x4 -x5  x0  x1  x2]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    # x:   [..., D], eg [x0,     x1,   x2,   x3,   x4,   x5]
    # sin: [..., D], eg [sin0, sin1, sin2, sin0, sin1, sin2]
    # cos: [..., D], eg [cos0, cos1, cos2, cos0, cos1, cos2]
    return (x * cos) + (rope_rotate_half(x) * sin)

def prefix_rope(token, sin, cos):
    N = token.shape[-2]
    prefix = N - sin.shape[-2]
    if prefix > 0:
        token_prefix = token[:, :, :prefix, :]
        token = rope_apply(token[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        token = torch.cat((token_prefix, token), dim=-2)  # [B, head, N, D//head]
    else:
        token = rope_apply(token, sin, cos)
    return token

def apply_rope( q: Tensor, k: Tensor, q_rope: Tensor | Tuple[Tensor, Tensor], k_rope: Tensor | Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
    # All operations will use the dtype of rope, the output is cast back to the dtype of q and k
    q_dtype = q.dtype
    k_dtype = k.dtype
    sin_q, cos_q = q_rope
    sin_k, cos_k = k_rope
    rope_dtype = sin_q.dtype
    q = q.to(dtype=rope_dtype)
    k = k.to(dtype=rope_dtype)
    q = prefix_rope(q, sin_q, cos_q)
    k = prefix_rope(k, sin_k, cos_k)
    q = q.to(dtype=q_dtype)
    k = k.to(dtype=k_dtype)
    return q, k

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
        self.factory_kwargs = factory_kwargs
        
        if is_self_attention:
            self.proj_in = nn.Linear(model_dim,total_dim*3, bias = False, **factory_kwargs)
        else:
            self.proj_q = nn.Linear(model_dim,total_dim, bias = False, **factory_kwargs )
            self.proj_kv = nn.Linear(model_dim, total_dim * 2, bias = False, **factory_kwargs )
        self.proj_out = nn.Linear(total_dim,model_dim, bias = False, **factory_kwargs )
    
    def forward(self,query:torch.Tensor,
                     key :torch.Tensor = None,
                     attn_mask :torch.Tensor= None,
                     q_rope :Tuple[torch.Tensor, torch.Tensor]| None = None,
                     k_rope :Tuple[torch.Tensor, torch.Tensor]| None = None):
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
        query = query.unflatten(-1,[self.n_heads, self.head_dim]).transpose(1,2)
        key = key.unflatten(-1,[self.n_heads, self.head_dim]).transpose(1,2)
        value = value.unflatten(-1,[self.n_heads, self.head_dim]).transpose(1,2)
        if q_rope is not None:
            query, key = apply_rope(query, key, q_rope, k_rope if k_rope is not None else q_rope)
        L_b,_,L_q,_ = query.size()
        L_k = key.size(2)
        # scaled dot product
        if attn_mask is None:
            attn_output = F.scaled_dot_product_attention(query, key, value,is_causal= self.is_causal)
        else:
            # attention mask shape (B, L_seq)
            attn_bias = torch.zeros(size = (L_b, 1, L_q, L_k), device = query.device, dtype = query.dtype)
            attn_bias = attn_bias.masked_fill(attn_mask.unsqueeze(1).unsqueeze(2).logical_not(), float("-inf"))
            if self.is_causal:
                temp_mask = torch.ones((L_q, L_k), dtype = torch.bool, device = query.device).tril()
                attn_bias = attn_bias.masked_fill(temp_mask.logical_not(), float("-inf"))            
            attn_output = F.scaled_dot_product_attention(query, key, value, attn_mask = attn_bias)
        # (B, n_heads, L_seq, head_dims)
        attn_output = attn_output.transpose(1,2).flatten(-2)
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

    def __self_attention(self, x, attn_mask, rope_pos = None):
        return self.dropout1(self.self_attn(x, attn_mask=attn_mask, q_rope=rope_pos))

    def __feed_forward(self, x):
        return self.dropout3(self.linear2(self.dropout2(self.activation(self.linear1(x)))))

    def forward(self, src, src_mask = None, rope_pos = None):
        x = src
        if self.norm_first:
            x = x + self.__self_attention(self.norm1(x), src_mask, rope_pos)
            x = x + self.__feed_forward(self.norm2(x))
        else:
            x = self.norm1(x + self.__self_attention(x, src_mask, rope_pos))
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

    def __self_attention(self, x, attn_mask = None):
        return self.dropout1(self.self_attn(x, attn_mask=attn_mask))

    def __cross_attention(self, src, mem, attn_mask = None):
        return self.dropout2(self.cross_attn(src, mem, attn_mask=attn_mask))

    def __feed_forward(self, x):
        return self.dropout4(self.linear2(self.dropout3(self.activation(self.linear1(x)))))

    
    def forward(self, src, mem, tgt_mask = None, memory_mask = None):
        x = src
        if self.norm_first:
            x = x + self.__self_attention(self.norm1(x), attn_mask=tgt_mask)
            x = x + self.__cross_attention(self.norm2(x), mem, attn_mask=memory_mask)
            x = x + self.__feed_forward(self.norm3(x))
        else:
            x = self.norm1(x + self.__self_attention(x, attn_mask=tgt_mask))
            x = self.norm2(x + self.__cross_attention(x, mem, attn_mask=memory_mask))
            x = self.norm3(x + self.__feed_forward(x))
        
        return x
    

class DoubleInputDecoder(nn.Module):
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
        self.self_attn_norm = nn.LayerNorm(model_dim,eps = eps, **factory_kwargs)
        self.self_attn_dropout = nn.Dropout(dropout_p)
        # ca1
        self.first_cross_attn = MultiHeadedAttention(model_dim,
                                            total_dim,
                                            n_heads,
                                            is_self_attention=False,
                                            is_causal= False,
                                            **factory_kwargs)
        self.first_cross_attn_norm = nn.LayerNorm(model_dim, eps = eps, **factory_kwargs)
        self.first_cross_attn_dropout = nn.Dropout(dropout_p)
        # ca2
        self.second_cross_attn = MultiHeadedAttention(model_dim,
                                            total_dim,
                                            n_heads,
                                            is_self_attention=False,
                                            is_causal= False,
                                            **factory_kwargs)
        self.second_cross_attn_norm = nn.LayerNorm(model_dim, eps = eps, **factory_kwargs)
        self.second_cross_attn_dropout = nn.Dropout(dropout_p)
        # ff
        self.linear_up = nn.Linear(model_dim, ff_dim, **factory_kwargs)
        self.linear_up_dropout = nn.Dropout(dropout_p)
        self.linear_down = nn.Linear(ff_dim, model_dim, **factory_kwargs)
        self.linear_down_dropout = nn.Dropout(dropout_p)
        self.linear_norm = nn.LayerNorm(model_dim,eps = eps, **factory_kwargs)

    def __self_attention(self, x, attn_mask = None, src_rope = None):
        return self.self_attn_dropout(self.self_attn(x, attn_mask=attn_mask, q_rope=src_rope))

    def __cross_attention1(self, src, mem, attn_mask = None,src_rope = None, mem1_rope = None):
        return self.first_cross_attn_dropout(self.first_cross_attn(src, mem, attn_mask=attn_mask, q_rope=src_rope, k_rope=mem1_rope))

    def __cross_attention2(self, src, mem, attn_mask = None,src_rope = None, mem2_rope = None):
        return self.second_cross_attn_dropout(self.second_cross_attn(src, mem, attn_mask=attn_mask, q_rope=src_rope, k_rope=mem2_rope))

    def __feed_forward(self, x):
        return self.linear_down_dropout(self.linear_down(self.linear_up_dropout(self.activation(self.linear_up(x)))))

    
    def forward(self, src,
                      mem1,
                      mem2,
                      tgt_mask = None,
                      mem1_mask = None,
                      mem2_mask = None,
                      src_rope = None,
                      mem1_rope = None,
                      mem2_rope = None):
        x = src
        if self.norm_first:
            x = x + self.__self_attention(self.self_attn_norm(x), attn_mask=tgt_mask, src_rope=src_rope)
            x = x + self.__cross_attention1(self.first_cross_attn_norm(x), mem1, attn_mask=mem1_mask, src_rope=src_rope, mem1_rope=mem1_rope)
            x = x + self.__cross_attention2(self.second_cross_attn_norm(x), mem2, attn_mask=mem2_mask, src_rope=src_rope, mem2_rope=mem2_rope)
            x = x + self.__feed_forward(self.linear_norm(x))
        else:
            x = self.self_attn_norm(x + self.__self_attention(x, attn_mask=tgt_mask, src_rope=src_rope))
            x = self.first_cross_attn_norm(x + self.__cross_attention1(x, mem1, attn_mask=mem1_mask, src_rope=src_rope, mem1_rope=mem1_rope))
            x = self.second_cross_attn_norm(x + self.__cross_attention2(x, mem2, attn_mask=mem2_mask, src_rope=src_rope, mem2_rope=mem2_rope))
            x = self.linear_norm(x + self.__feed_forward(x))
        
        return x
    
class FeatureEnhancer(nn.Module):
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
        self.first_self_attn = MultiHeadedAttention(model_dim,
                                            total_dim,
                                            n_heads,
                                            is_self_attention=True,
                                            is_causal= True,
                                            **factory_kwargs)
        self.first_attn_norm = nn.LayerNorm(model_dim,eps = eps, **factory_kwargs)
        self.first_attn_dropout = nn.Dropout(dropout_p)
        
        self.second_self_attn = MultiHeadedAttention(model_dim,
                                            total_dim,
                                            n_heads,
                                            is_self_attention=True,
                                            is_causal= True,
                                            **factory_kwargs)
        self.second_attn_norm = nn.LayerNorm(model_dim,eps = eps, **factory_kwargs)
        self.second_attn_dropout = nn.Dropout(dropout_p)
        # ca1
        self.f2s_cross_attn = MultiHeadedAttention(model_dim,
                                            total_dim,
                                            n_heads,
                                            is_self_attention=False,
                                            is_causal= False,
                                            **factory_kwargs)
        self.f2s_cross_attn_norm1 = nn.LayerNorm(model_dim, eps = eps, **factory_kwargs)
        self.f2s_cross_attn_norm2 = nn.LayerNorm(model_dim, eps = eps, **factory_kwargs)
        self.f2s_cross_attn_dropout = nn.Dropout(dropout_p)
        # ca2
        self.s2f_cross_attn = MultiHeadedAttention(model_dim,
                                            total_dim,
                                            n_heads,
                                            is_self_attention=False,
                                            is_causal= False,
                                            **factory_kwargs)
        self.s2f_cross_attn_norm1 = nn.LayerNorm(model_dim, eps = eps, **factory_kwargs)
        self.s2f_cross_attn_norm2 = nn.LayerNorm(model_dim, eps = eps, **factory_kwargs)
        self.s2f_cross_attn_dropout = nn.Dropout(dropout_p)
        # first ff
        self.first_ff = MLP(model_dim, ff_dim, hidden_dropout_p = dropout_p, 
                            output_dropout_p = dropout_p, out_dim = model_dim, **factory_kwargs)
        self.first_ff_norm = nn.LayerNorm(model_dim, eps = eps, **factory_kwargs)
        # second ff
        self.second_ff = MLP(model_dim, ff_dim, hidden_dropout_p = dropout_p, 
                             output_dropout_p = dropout_p, out_dim = model_dim, **factory_kwargs)
        self.second_ff_norm = nn.LayerNorm(model_dim, eps = eps, **factory_kwargs)

    def __self_attention(self, x , attn_dropout, self_attn, attn_mask = None, src_rope = None):
        return attn_dropout(self_attn(x, attn_mask=attn_mask, q_rope=src_rope))
    

    def __cross_attention(self, first, second, cross_attn_dropout, cross_attn, attn_mask = None, src_rope = None, mem_rope = None):
        return cross_attn_dropout(cross_attn(first, second, attn_mask=attn_mask, q_rope=src_rope, k_rope=mem_rope))

    def forward(self, src1, src2, src1_mask = None, src2_mask = None, src1_rope = None, src2_rope = None):
        x1 = src1
        x2 = src2
    
        if self.norm_first:
            x1 = self.first_attn_norm(x1)
            x2 = self.second_attn_norm(x2)
            x1 = x1 + self.__self_attention(x1,self.first_attn_dropout, self.first_self_attn, attn_mask=src1_mask, src_rope=src1_rope)
            x2 = x2 + self.__self_attention(x2,self.second_attn_dropout, self.second_self_attn, attn_mask=src2_mask, src_rope=src2_rope)
            
            x1 = self.f2s_cross_attn_norm1(x1)
            x2 = self.f2s_cross_attn_norm2(x2)
            x1 = x1 + self.__cross_attention(x1, x2, self.f2s_cross_attn_dropout, self.f2s_cross_attn, attn_mask=src2_mask, src_rope=src1_rope, mem_rope=src2_rope)
            
            # x1 = self.s2f_cross_attn_norm1(x1)
            # x2 = self.s2f_cross_attn_norm2(x2)
            # x2 = x2 + self.__cross_attention(x2, x1, self.s2f_cross_attn_dropout, self.s2f_cross_attn, attn_mask=src1_mask, src_rope=src2_rope, mem_rope=src1_rope)
            
            x1 = x1 + self.first_ff(self.first_ff_norm(x1))
            x2 = x2 + self.second_ff(self.second_ff_norm(x2))
        else:
            x1 = self.first_attn_norm(x1 + self.__self_attention(x1,self.first_attn_dropout, self.first_self_attn, attn_mask=src1_mask, src_rope=src1_rope))
            x2 = self.first_attn_norm(x2 + self.__self_attention(x2,self.second_attn_dropout, self.second_self_attn, attn_mask=src2_mask, src_rope=src2_rope))
            x1 = self.f2s_cross_attn_norm(x1 + self.__cross_attention(x1, x2, self.f2s_cross_attn_dropout, self.f2s_cross_attn, attn_mask=src2_mask, src_rope=src1_rope, mem_rope=src2_rope))
            x2 = self.s2f_cross_attn_norm(x2 + self.__cross_attention(x2, x1, self.s2f_cross_attn_dropout, self.s2f_cross_attn,attn_mask=src1_mask, src_rope=src2_rope, mem_rope=src1_rope))
            x1 = self.first_ff_norm(x1 + self.first_ff(x1))
            x2 = self.second_ff_norm(x2 + self.second_ff(x2))
        
        return x1, x2

    
