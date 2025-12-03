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
class FourierPosEncoder(nn.Module):
    def __init__(self, input_dim, num_freq_bands, hidden_dim, output_dim, device = 'cpu', dtype = torch.float32):
        """
        Args:
            input_dim: Dimension of input coordinates (e.g., 3 for x,y,z).
            num_freq_bands: Number of frequency bands (L in NeRF papers).
                            Output of encoding will be input_dim * num_freq_bands * 2.
            hidden_dim: Hidden dimension of the MLP.
            output_dim: Output dimension of the MLP.
            include_pi: Whether to multiply input by pi (Standard NeRF = True).
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_freq_bands = num_freq_bands
        factory_mode = {'device':device, 'dtype': dtype}
        # 1. Use register_buffer so this moves to GPU automatically with the model
        #    and is saved in the state_dict.
        freqs = 2.0 ** torch.arange(0, num_freq_bands, **factory_mode)
        self.register_buffer('freqs', freqs)
        
        # Calculate the size of the embedding before the MLP
        # 2 (sin/cos) * num_freq_bands * input_dim
        self.embed_dim = input_dim * num_freq_bands * 2
        
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # (B,L,F) or (B,L)
        if self.input_dim == 1:
            x = x.unsqueeze(-1)
        x = x * torch.pi
        # (B,L,F) -> (B,L,F,freqs)
        x_expanded = x.unsqueeze(-1) * self.freqs 
        # (B,L,F,freqs) -> (B,L,F,freqs*2)
        x_cat = torch.cat([torch.sin(x_expanded) ,torch.cos(x_expanded)], dim=-1)
        # (B,L,F,freqs*2) -> (B,L,F*freqs*2)
        x_flat = x_cat.flatten(start_dim=-2)
        
        return self.mlp(x_flat)