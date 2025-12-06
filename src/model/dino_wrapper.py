import torch
import torch.nn as nn

import os

class DinoV3Wrapper(nn.Module):
    def __init__(self, repo_path, model_name, freeze=True, device='cpu', weights=None, **kwargs):
        super().__init__()
        self.repo_path = repo_path
        self.model_name = model_name
        self.freeze = freeze
        self.device = device

        # Check if repo_path exists. If not, use github repo.
        if os.path.exists(repo_path):
            kwargs = {'source': 'local'}
            repo = repo_path
        else:
            kwargs = {'source': 'github'}
            repo = 'facebookresearch/dinov3'  # Default public repo for DINOv3

        if weights:
            kwargs['weights'] = weights

        self.model = torch.hub.load(repo, model_name, **kwargs)
        self.model.to(device)
        self.embed_dim = self.model.embed_dim
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
                
    def forward(self, x):
        
        with torch.set_grad_enabled(not self.freeze):
            # We assume x is already properly normalized/transformed by the pipeline
            out = self.model.get_intermediate_layers(x, n=1, reshape=True, norm=True, return_class_token=True)
            feats, cls_token = out[0]
            cls_token = cls_token.unsqueeze(1)
            feats = feats.flatten(2,3).permute(0,2,1) # (B, H*W, F)
            return torch.cat([feats, cls_token], dim=1)