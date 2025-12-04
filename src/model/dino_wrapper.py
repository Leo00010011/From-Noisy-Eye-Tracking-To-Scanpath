import torch
import torch.nn as nn

import os

class DinoV3Wrapper(nn.Module):
    def __init__(self, repo_path, model_name, freeze=True, regularization=True, weights=None, device='cpu'):
        super().__init__()
        self.repo_path = repo_path
        self.model_name = model_name
        self.freeze = freeze
        self.regularization = regularization
        self.device = device
        # Check if repo_path exists. If not, use github repo.
        if os.path.exists(repo_path):
            kwargs = {'source': 'local'}
            repo = repo_path
        else:
            kwargs = {'source': 'github'}
            # Note: Ensure 'dinov3' exists, otherwise use 'dinov2'
            repo = 'facebookresearch/dinov3' 

        if weights:
            kwargs['weights'] = weights

        # Load model
        self.model = torch.hub.load(repo, model_name, **kwargs)
        self.model = self.model.to(device)
        self.embed_dim = self.model.embed_dim

        # Initial Freeze Setup
        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # We set the mode initially, but the train() override below is what ensures it sticks
        self._set_model_mode(train_mode=True) 

    def _set_model_mode(self, train_mode):
        """Helper to enforce specific DINO mode based on config"""
        if train_mode and self.regularization:
            print("Enabling image encoder regularization")
            self.model.train()
        else:
            print("Disabling image encoder regularization")
            self.model.eval()

    def train(self, mode=True):
        """
        Override train to ensure DINO stays in the correct state
        even when the wrapper is switched to train mode.
        """
        super().train(mode)
        self._set_model_mode(mode)
                    
    @torch.compiler.disable
    def forward(self, x):
        with torch.no_grad():
            out = self.model.get_intermediate_layers(
                x, n=1, reshape=True, norm=True, return_class_token=True
            )
        feats, cls_token = out[0]
        feats = feats.flatten(2, 3).transpose(1, 2)
        cls_token = cls_token.unsqueeze(1)
        
        return torch.cat([feats, cls_token], dim=1)
