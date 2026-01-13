import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
import math


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
            if include_dropout:
                layers.append(nn.Dropout(hidden_dropout_p))
            layers.append(nn.Linear(prev_dim, hidden_dim, **factory_kwargs))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        if include_dropout:
            layers.append(nn.Dropout(output_dropout_p))
        # Output layer
        layers.append(nn.Linear(prev_dim, out_dim, **factory_kwargs))
        
        self.head = nn.Sequential(*layers)
    
    def forward(self, x, **kwargs):
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
    def __init__(self,model_dim, total_dim ,n_heads, is_self_attention = False, is_causal = False, use_kv_cache = False, device = 'cpu', dtype = torch.float32):
        super().__init__()
        self.model_dim = model_dim
        self.total_dim = total_dim
        self.n_heads = n_heads
        self.is_self_attention = is_self_attention
        self.is_causal = is_causal
        self.head_dim = total_dim//n_heads
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.factory_kwargs = factory_kwargs
        self.scale = 1.0 / (self.head_dim ** 0.5)
        self.use_kv_cache = use_kv_cache
        self.kv_cache = None
        
        if is_self_attention:
            self.proj_in = nn.Linear(model_dim,total_dim*3, bias = False, **factory_kwargs)
        else:
            self.proj_q = nn.Linear(model_dim,total_dim, bias = False, **factory_kwargs )
            self.proj_kv = nn.Linear(model_dim, total_dim * 2, bias = False, **factory_kwargs )
        self.proj_out = nn.Linear(total_dim,model_dim, bias = False, **factory_kwargs )
        
    def clear_kv_cache(self):
        self.kv_cache = None
    
    def disable_kv_cache(self):
        self.kv_cache = None
        self.use_kv_cache = False
    
    def forward(self,query:torch.Tensor,
                     key :torch.Tensor = None,
                     attn_mask :torch.Tensor= None,
                     q_rope :Tuple[torch.Tensor, torch.Tensor]| None = None,
                     k_rope :Tuple[torch.Tensor, torch.Tensor]| None = None):
        # Fix reshape when cross attention
        # FIX start token in autoregressive process
        if self.is_self_attention:
            result = self.proj_in(query)
            query, key, value = torch.chunk(result, 3,dim = -1)
        else:
            query = self.proj_q(query)
            if self.kv_cache is None:
                result = self.proj_kv(key)
                key, value = torch.chunk(result, 2,dim = -1)
                if self.use_kv_cache:
                    self.kv_cache = (key, value)
            else:
                key = self.kv_cache[0]
                value = self.kv_cache[1]
                
        # reshape
        L_b,L_q,_ = query.size()
        L_k = key.size(1)
        # (B,L_seq, total_dim) -> (B, L_seq, n_heads, head_dim) -> (B, n_heads, L_seq, head_dim)
        query = query.view(L_b,L_q,self.n_heads, self.head_dim).transpose(1,2) 
        key = key.view(L_b,L_k,self.n_heads, self.head_dim).transpose(1,2)
        value = value.view(L_b,L_k,self.n_heads, self.head_dim).transpose(1,2)
        if self.is_self_attention:
            if self.kv_cache is not None:
                key = torch.cat([self.kv_cache[0], key], dim=2)
                value = torch.cat([self.kv_cache[1], value], dim=2)
                self.kv_cache = (key, value)  
            elif self.use_kv_cache:
                self.kv_cache = (key, value)
        # rope
        if q_rope is not None and k_rope is not None:
            query, key = apply_rope(query, key, q_rope, k_rope if k_rope is not None else q_rope)
        # scaled dot product
        if attn_mask is None:
            causal = self.is_causal and not self.use_kv_cache
            attn_output = F.scaled_dot_product_attention(query, key, value,is_causal= causal, scale=self.scale)
        else:
            # attention mask shape (B, L_seq)
            attn_bias = torch.zeros(size = (L_b, 1, L_q, L_k), device = query.device, dtype = query.dtype)
            attn_bias = attn_bias.masked_fill(attn_mask.unsqueeze(1).unsqueeze(2).logical_not(), float("-inf"))
            if self.is_causal:
                temp_mask = torch.ones((L_q, L_k), dtype = torch.bool, device = query.device).tril()
                attn_bias = attn_bias.masked_fill(temp_mask.logical_not(), float("-inf"))            
            attn_output = F.scaled_dot_product_attention(query, key, value, attn_mask = attn_bias, scale=self.scale)
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
    def __init__(self,model_dim = 1024, total_dim = 1024, n_heads = 8, ff_dim = 2048,dropout_p = .1,activation = F.relu,eps = 1e-5,norm_first = False, use_kv_cache = False, device = 'cpu', dtype = torch.float32):
        super().__init__()
        self.model_dim = model_dim
        self.total_dim = total_dim
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.dropout_p = dropout_p
        self.eps = eps
        self.activation = activation
        self.norm_first = norm_first
        self.use_kv_cache = use_kv_cache
        factory_kwargs = {'device': device, 'dtype': dtype}
        # sa
        self.self_attn = MultiHeadedAttention(model_dim,
                                            total_dim,
                                            n_heads,
                                            is_self_attention=True,
                                            is_causal= True,
                                            use_kv_cache=use_kv_cache,
                                            **factory_kwargs)
        self.self_attn_norm = nn.LayerNorm(model_dim,eps = eps, **factory_kwargs)
        self.self_attn_dropout = nn.Dropout(dropout_p)
        # ca1
        self.first_cross_attn = MultiHeadedAttention(model_dim,
                                            total_dim,
                                            n_heads,
                                            is_self_attention=False,
                                            is_causal= False,
                                            use_kv_cache=use_kv_cache,
                                            **factory_kwargs)
        self.first_cross_attn_norm = nn.LayerNorm(model_dim, eps = eps, **factory_kwargs)
        self.first_cross_attn_dropout = nn.Dropout(dropout_p)
        # ca2
        self.second_cross_attn = MultiHeadedAttention(model_dim,
                                            total_dim,
                                            n_heads,
                                            is_self_attention=False,
                                            is_causal= False,
                                            use_kv_cache=use_kv_cache,
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

    def get_cached_input_count(self):
        if self.use_kv_cache:
            if self.self_attn.kv_cache is not None:
                return self.self_attn.kv_cache[0].shape[2]
            else:
                return 0
        else:
            return 0

    def clear_kv_cache(self):
        self.self_attn.clear_kv_cache()
        self.first_cross_attn.clear_kv_cache()
        self.second_cross_attn.clear_kv_cache()
    
    def disable_kv_cache(self):
        self.use_kv_cache = False
        self.self_attn.disable_kv_cache()
        self.first_cross_attn.disable_kv_cache()
        self.second_cross_attn.disable_kv_cache()
    
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
                                            is_causal= False,
                                            **factory_kwargs)
        self.first_attn_norm = nn.LayerNorm(model_dim,eps = eps, **factory_kwargs)
        self.first_attn_dropout = nn.Dropout(dropout_p)
        self.second_self_attn = MultiHeadedAttention(model_dim,
                                            total_dim,
                                            n_heads,
                                            is_self_attention=True,
                                            is_causal= False,
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
            
            x1 = self.s2f_cross_attn_norm1(x1)
            x2 = self.s2f_cross_attn_norm2(x2)
            x2 = x2 + self.__cross_attention(x2, x1, self.s2f_cross_attn_dropout, self.s2f_cross_attn, attn_mask=src1_mask, src_rope=src2_rope, mem_rope=src1_rope)
            
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

    
class ArgMaxRegressor(nn.Module):
    def __init__(self, H,W, device = 'cpu', dtype = torch.float32):
        super().__init__()
        self.H = H
        self.W = W
        H_coords = (torch.arange(H) + 0.5)/H 
        W_coords = (torch.arange(W) + 0.5)/W 
        
        self.coord_mesh = torch.stack(torch.meshgrid(H_coords, W_coords), dim=-1)
        # create coordinate mesh
        # (H,W,2) -> (1,H,W,2) -> (1,H,W,1,2) -> (1,H*W,1,2)
        self.coord_mesh = self.coord_mesh.unsqueeze(0).unsqueeze(-2).flatten(start_dim=1, end_dim=2)
        self.coord_mesh = self.coord_mesh.to(device)
        self.coord_mesh = self.coord_mesh.to(dtype)
    
    
    def forward(self, x, visual_tokens):
        # x: (B, L, F)
        # visual tokens (B, 1 + H*W,F) 
        # removing extra visual tokens
        N = self.coord_mesh.shape[1]
        prefix = visual_tokens.shape[1] - N
        visual_tokens = visual_tokens[:, prefix:, :]
        # visual_tokens: (B, H*W, F)
        # Compute similarity between each query (x) and spatial location (visual_tokens)
        similarity = torch.matmul(visual_tokens, x.transpose(-2, -1))  # (B, H*W, L)
        # Apply softmax over the spatial dimension so weights sum to one over image positions for each token
        attn_weights = torch.softmax(similarity, dim=1)  # (B, H*W, L)
        # Compute weighted sum: (B, H*W, L, 2)
        weighted = attn_weights.unsqueeze(-1) * self.coord_mesh  # (B, H*W, L, 2)
        # Sum over the spatial dimension (H*W) to get predicted 2D coords per output token
        soft_argmax = weighted.sum(dim=1)               # (B, L, 2)
        return soft_argmax
    

class LearnableCoordinateDropout(nn.Module):
    def __init__(self, model_dim, dropout_prob=0.2, device='cpu', dtype=torch.float32):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.model_dim = model_dim
        self.mask_token = nn.Parameter(torch.randn(1, 1, model_dim, device=device, dtype=dtype))

    def forward(self, x):
        """
        x shape: (B, L, model_dim)
        Returns: (B, L, model_dim) with some steps replaced by self.mask_token
        """
        if not self.training or self.dropout_prob <= 0:
            return x

        B, L, C = x.shape
        
        drop_mask = torch.rand(B, L, 1, device=x.device) < self.dropout_prob
        drop_mask[:, 0, :] = False
        x_dropped = torch.where(drop_mask, self.mask_token.expand(B, L, -1), x)

        return x_dropped


class ResidualRegressor(nn.Module):
    def __init__(self, model_dim, hidden_dropout_p = 0, output_dropout_p = 0, device='cpu', dtype=torch.float32):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.regressor = MLP(model_dim, [model_dim, model_dim//2], 2, hidden_dropout_p = hidden_dropout_p, output_dropout_p = output_dropout_p, include_dropout = False, **factory_kwargs)
        
    def forward(self, src_tokens, src, **kwargs):
        return self.regressor(src_tokens) + src[:,:,:2]
    
class GatedFusion(nn.Module):
    def __init__(self, input_dim, dropout_p=0.0, device='cpu', dtype=torch.float32):
        """
        Gated Fusion Network.
        
        Args:
            input_dim (int): The dimensionality of the input feature vectors.
                             Both inputs must have this dimension.
            dropout_p (float): Dropout probability.
        """
        super(GatedFusion, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.fc_gate = nn.Linear(input_dim * 2, input_dim, **factory_kwargs)
        
        self.dropout = nn.Dropout(dropout_p)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        """
        Forward pass for Gated Fusion.

        Args:
            x (torch.Tensor): Input tensor 1 of shape (batch_size, input_dim) 
                              or (batch_size, seq_len, input_dim).
            y (torch.Tensor): Input tensor 2 of shape (batch_size, input_dim)
                              or (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: Fused output of same shape as inputs.
        """
        combined = torch.cat((x, y), dim=-1)
        gate = self.sigmoid(self.fc_gate(combined))
        output = gate * x + (1 - gate) * y
        output = self.dropout(output)
        
        return output

class TrajectoryHeatmapGenerator(nn.Module):
    def __init__(self, embed_dim, feature_height, feature_width, device='cpu', dtype=torch.float32):
        """
        Args:
            embed_dim (int): The feature dimension (F).
            feature_height (int): The height of the feature map (H) coming from ViT.
            feature_width (int): The width of the feature map (W) coming from ViT.
        """
        super().__init__()
        self.H = feature_height
        self.W = feature_width
        self.embed_dim = embed_dim
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.up_conv = nn.ConvTranspose2d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=4,
            stride=2,
            padding=1,
            **factory_kwargs
        )
        

    def forward(self, image_features, trajectory_tokens):
        """
        Args:
            image_features: Tensor of shape (B, 1 + H*W, F)
            trajectory_tokens: Tensor of shape (B, L, F)

        Returns:
            heatmaps: Tensor of shape (B, L, 2*H * 2*W)
        """
        B, N,F = image_features.shape
        _,L,_ = trajectory_tokens.shape
        # Shape: (B, H*W, F)
        spatial_tokens = image_features[:, 1:, :] 
        # Shape: (B, F, H, W)
        spatial_map = spatial_tokens.transpose(1, 2).reshape(B, F, self.H, self.W)
        # Shape: (B, F, 2H, 2W)
        upsampled_map = self.up_conv(spatial_map)
        # Shape: (B, F, 4*H*W)
        flat_upsampled_map = upsampled_map.view(B, F, -1)
        # Shape: (B, L, 4*H*W)
        heatmaps = torch.bmm(trajectory_tokens, flat_upsampled_map)
        heatmaps = heatmaps.view(B, L, 2 * self.H, 2 * self.W)
        return heatmaps   


class DeformableAttention(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_points=4,
        dropout=0.1,
        geometric_sigma = 0,
        device = 'cpu',
        dtype = torch.float32
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.device = device
        self.dtype = dtype
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads
        self.geometric_sigma = geometric_sigma

        # 1. Sampling Offsets: 
        # Output dim: num_heads * num_points * 2 (x, y)
        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_points * 2, **factory_kwargs)
        
        # 2. Attention Weights:
        # Output dim: num_heads * num_points
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_points, **factory_kwargs)
        
        # 3. Projections
        self.value_proj = nn.Linear(embed_dim, embed_dim, **factory_kwargs)
        self.output_proj = nn.Linear(embed_dim, embed_dim, **factory_kwargs)
        
        self.dropout = nn.Dropout(dropout)
        
        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize offsets to 0
        nn.init.constant_(self.sampling_offsets.weight.data, 0.)
        
        # Grid Init: Initialize bias to the "Star Pattern"
        thetas = torch.arange(self.num_heads, dtype=self.dtype, device=self.device) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1) # (Heads, 2)
        
        # (Heads, Points, 2)
        grid_init = grid_init.unsqueeze(1).repeat(1, self.num_points, 1) 
        
        # Scale the points (Point 0 -> 1x, Point 1 -> 2x, etc.)
        for i in range(self.num_points):
            grid_init[:, i, :] *= i + 1
            
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
            
        nn.init.constant_(self.attention_weights.weight.data, 0.)
        nn.init.constant_(self.attention_weights.bias.data, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.xavier_uniform_(self.output_proj.weight.data)

    def forward(self, query, reference_points, value, spatial_shape):
        """
        Args:
            query: (Batch, Num_Queries, Embed_Dim)
            reference_points: (Batch, Num_Queries, 2) - Normalized [0, 1]
            value: (Batch, H*W, Embed_Dim) - The image feature map
            spatial_shape: (H, W) tuple - Shape of the feature map
            
        Returns:
            output: (Batch, Num_Queries, Embed_Dim)
        """
        bs, num_queries, _ = query.shape
        H, W = spatial_shape
        # 1. Project Input Features
        # (Batch, H*W, Head, Head_Dim) -> (Batch, Head, Head_Dim, H, W)
        value = self.value_proj(value)
        value = value.view(bs, H * W, self.num_heads, self.head_dim)
        value = value.permute(0, 2, 3, 1).view(bs, self.num_heads, self.head_dim, H, W)
        
        # 2. Generate Offsets and Attention Weights
        # Offsets: (Batch, Num_Queries, Heads, Points, 2)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_queries, self.num_heads, self.num_points, 2
        )
        
        # Weights: (Batch, Num_Queries, Heads, Points)
        attention_weights = self.attention_weights(query).view(
            bs, num_queries, self.num_heads, self.num_points
        )
        attention_weights = F.softmax(attention_weights, -1)

        attention_weights = self.dropout(attention_weights)
        # 3. Compute Sampling Locations
        # sampling_offsets are unconstrained, so we normalize them by H and W
        # (Batch, Num_Queries, Heads, Points, 2)
        if self.training and self.geometric_sigma > 0:
            # Create noise: (Batch, Num_Queries, Heads, Points, 2)
            # Use a small sigma, e.g., 0.1 or 0.05
            noise = torch.randn_like(sampling_offsets) * self.geometric_sigma
            sampling_offsets = sampling_offsets + noise
            
        offset_normalizer = torch.tensor([W, H], device=query.device, dtype=query.dtype)
        
        # Ref Points: (Batch, Num_Queries, 1, 1, 2) -> Broadcasat to Heads & Points
        sampling_locations = reference_points[:, :, None, None, :] \
                             + sampling_offsets / offset_normalizer[None, None, None, None, :]
        
        # 4. Grid Sample
        # We need to reshape inputs to use grid_sample efficiently
        # New Batch Size = Batch * Heads
        
        # Reshape Value: (Batch*Heads, Head_Dim, H, W)
        # log view parameters
        value_flat = value.reshape(bs * self.num_heads, self.head_dim, H, W)
        
        # Reshape Grid: (Batch*Heads, Num_Queries, Points, 2)
        # Also convert [0, 1] -> [-1, 1] for grid_sample
        sampling_grid = 2 * sampling_locations - 1
        sampling_grid = sampling_grid.permute(0, 2, 1, 3, 4).flatten(0, 1)

        # Sample! 
        # Output: (Batch*Heads, Head_Dim, Num_Queries, Points)
        sampled_values = F.grid_sample(
            value_flat, 
            sampling_grid, 
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=False
        )
        
        # 5. Apply Attention Weights
        # Reshape weights: (Batch*Heads, 1, Num_Queries, Points)
        attention_weights = attention_weights.permute(0, 2, 1, 3).flatten(0, 1).unsqueeze(1)
        
        # Weighted Sum over Points
        # (Batch*Heads, Head_Dim, Num_Queries)
        output = (sampled_values * attention_weights).sum(-1)
        
        # 6. Final Projection
        # (Batch, Num_Queries, Embed_Dim)
        output = output.view(bs, self.num_heads * self.head_dim, num_queries).transpose(1, 2)
        
        return self.output_proj(output)
    
    
class DeformableDecoder(nn.Module):
    def __init__(self, model_dim = 1024,
                      total_dim = 1024,
                      n_heads = 8,
                      ff_dim = 2048,
                      dropout_p = 0,
                      activation = F.relu,
                      eps = 1e-5,
                      norm_first = False,
                      num_points = 4,
                      spatial_shape = (16, 16),
                      geometric_sigma = 0,
                      device = 'cpu',
                      dtype = torch.float32):
        super().__init__()
        self.spatial_shape = spatial_shape
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
                                             is_causal= False,
                                             **factory_kwargs)
        self.norm1 = nn.LayerNorm(model_dim,eps = eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout_p)
        # ca
        self.cross_attn = DeformableAttention(embed_dim=model_dim,
                                            num_heads=n_heads,
                                            num_points=num_points,
                                            geometric_sigma=geometric_sigma,
                                            dropout= dropout_p,
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

    def __cross_attention(self, src, mem, reference_points = None):
        return self.dropout2(self.cross_attn(query=src, reference_points=reference_points, value=mem, spatial_shape=self.spatial_shape))

    def __feed_forward(self, x):
        return self.dropout4(self.linear2(self.dropout3(self.activation(self.linear1(x)))))

    
    def forward(self, src, mem, tgt_mask = None, reference_points = None):
        x = src
        if self.norm_first:
            x = x + self.__self_attention(self.norm1(x), attn_mask=tgt_mask)
            x = x + self.__cross_attention(self.norm2(x), mem[:,1:,:], reference_points=reference_points)
            x = x + self.__feed_forward(self.norm3(x))
        else:
            x = self.norm1(x + self.__self_attention(x, attn_mask=tgt_mask))
            x = self.norm2(x + self.__cross_attention(x, mem[:,1:,:], reference_points=reference_points))
            x = self.norm3(x + self.__feed_forward(x))
        
        return x
        

    
class DeformableDoubleInputDecoder(nn.Module):
    def __init__(self, model_dim = 1024,
                      total_dim = 1024,
                      n_heads = 8,
                      ff_dim = 2048,
                      dropout_p = 0,
                      activation = F.relu,
                      eps = 1e-5,
                      norm_first = False,
                      use_kv_cache = False,
                      spatial_shape = (16, 16),
                      num_points = 4,
                      device = 'cpu',
                      dtype = torch.float32):
        super().__init__()
        self.model_dim = model_dim
        self.total_dim = total_dim
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.dropout_p = dropout_p
        self.eps = eps
        self.activation = activation
        self.norm_first = norm_first
        self.use_kv_cache = use_kv_cache
        self.num_points = num_points
        self.spatial_shape = spatial_shape
        factory_kwargs = {'device': device, 'dtype': dtype}
        # sa
        self.self_attn = MultiHeadedAttention(model_dim,
                                            total_dim,
                                            n_heads,
                                            is_self_attention=True,
                                            is_causal= True,
                                            use_kv_cache=use_kv_cache,
                                            **factory_kwargs)
        self.self_attn_norm = nn.LayerNorm(model_dim,eps = eps, **factory_kwargs)
        self.self_attn_dropout = nn.Dropout(dropout_p)
        # ca1
        
        self.first_cross_attn = DeformableAttention(embed_dim=model_dim,
                                                    num_heads=n_heads,
                                                    num_points=num_points,
                                                    dropout= dropout_p,
                                            **factory_kwargs)
        self.first_cross_attn_norm = nn.LayerNorm(model_dim, eps = eps, **factory_kwargs)
        self.first_cross_attn_dropout = nn.Dropout(dropout_p)
        # ca2
        self.second_cross_attn = MultiHeadedAttention(model_dim,
                                            total_dim,
                                            n_heads,
                                            is_self_attention=False,
                                            is_causal= False,
                                            use_kv_cache=use_kv_cache,
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

    def __cross_attention2(self, src, mem, reference_points = None):
        return self.dropout2(self.cross_attn(query=src, reference_points=reference_points, value=mem, spatial_shape=self.spatial_shape))

    def __feed_forward(self, x):
        return self.linear_down_dropout(self.linear_down(self.linear_up_dropout(self.activation(self.linear_up(x)))))

    def get_cached_input_count(self):
        if self.use_kv_cache:
            if self.self_attn.kv_cache is not None:
                return self.self_attn.kv_cache[0].shape[2]
            else:
                return 0
        else:
            return 0

    def clear_kv_cache(self):
        self.self_attn.clear_kv_cache()
        self.first_cross_attn.clear_kv_cache()
        self.second_cross_attn.clear_kv_cache()
    
    def disable_kv_cache(self):
        self.use_kv_cache = False
        self.self_attn.disable_kv_cache()
        self.first_cross_attn.disable_kv_cache()
        self.second_cross_attn.disable_kv_cache()
    
    def forward(self, src,
                      mem1,
                      mem2,
                      tgt_mask = None,
                      mem1_mask = None,
                      mem2_mask = None,
                      reference_points = None):
        x = src
        if self.norm_first:
            x = x + self.__self_attention(self.self_attn_norm(x), attn_mask=tgt_mask, src_rope= None)
            x = x + self.__cross_attention1(self.first_cross_attn_norm(x), mem1, attn_mask=mem1_mask, src_rope=None, mem1_rope=None)
            x = x + self.__cross_attention2(self.norm2(x), mem2[:,1:,:], reference_points=reference_points)
            x = x + self.__feed_forward(self.linear_norm(x))
        else:
            x = self.self_attn_norm(x + self.__self_attention(x, attn_mask=tgt_mask, src_rope= None))
            x = self.first_cross_attn_norm(x + self.__cross_attention1(x, mem1, attn_mask=mem1_mask, src_rope= None, mem1_rope=None))
            x = self.second_cross_attn_norm(x + self.__cross_attention2(x, mem2, attn_mask=mem2_mask, src_rope= None, mem2_rope= None))
            x = self.linear_norm(x + self.__feed_forward(x))
        
        return x