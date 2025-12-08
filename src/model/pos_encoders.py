import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

class PositionalEncoding:
    def __init__(self, max_pos, model_dim, device = 'cpu', dtype = torch.float32):
        self.max_pos = max_pos
        self.model_dim = model_dim

        pe = np.empty((max_pos, model_dim),dtype=np.float32)
        position = np.arange(max_pos)[:,np.newaxis]
        div = np.exp(np.arange(0,model_dim,2)*-(np.log(10000)/model_dim))
        pe[:,0::2] = np.sin(position*div)
        pe[:,1::2] = np.cos(position*div)
        self.pe = torch.from_numpy(pe).to(device = device)
class GaussianFourierPosEncoder(nn.Module):
    def __init__(self, input_dim, mapping_size, hidden_dim, output_dim, sigma=1.0, device='cpu', dtype=torch.float32):
        """
        Args:
            mapping_size: Number of random Fourier features (output will be input_dim * mapping_size * 2).
                          Replaces 'num_freq_bands'.
            sigma: Bandwidth parameter. 
                   LOW sigma (e.g., 1.0) = smooth functions (prevents overfitting).
        """
        super().__init__()
        self.input_dim = input_dim
        
        # 1. Create the random matrix B *once*
        # We sample from a Normal distribution N(0, sigma^2)
        # Size: [input_dim, mapping_size]
        if input_dim == 1:
            B = torch.randn((mapping_size), device=device, dtype=dtype) * sigma
        else:
            B = torch.randn((input_dim, mapping_size), device=device, dtype=dtype) * sigma
        
        # 2. Register it as a buffer. 
        # It is NOT a parameter (no gradients), but it IS part of state_dict.
        self.register_buffer('B', B)
        
        self.embed_dim = input_dim * mapping_size * 2 # sin + cos
        
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.mlp.to(device=device, dtype=dtype)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Input_Dim)
        # 1. Project input: (2*pi*x) @ B
        # (B, L, input_dim) @ (input_dim, mapping_size) -> (B, L, mapping_size)
        x = x.unsqueeze(-1)
        # The coordinates are normalized to the range [-1, 1] therefore we multiply by pi instead of 2*pi
        if self.input_dim == 1:
            projected = (torch.pi * x) * self.B
        else:
            projected = (torch.pi * x) * self.B
            projected = projected.flatten(start_dim=-2)
        
        # 2. Fourier features: [sin, cos]
        # -> (B, L, mapping_size * 2)
        x_proj = torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)
        
        return self.mlp(x_proj)