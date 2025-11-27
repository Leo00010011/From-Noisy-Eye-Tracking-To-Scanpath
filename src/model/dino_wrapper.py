import torch
import torch.nn as nn

class DinoV3Wrapper(nn.Module):
    def __init__(self, repo_path, model_name, freeze=True, device='cpu', weights=None):
        super().__init__()
        self.repo_path = repo_path
        self.model_name = model_name
        self.freeze = freeze
        self.device = device
        
        # Load model from local hub
        # The script used: torch.hub.load(DINO_REPO_PATH, 'dinov3_vits16', source='local', weights=name)
        kwargs = {'source': 'local'}
        if weights:
            kwargs['weights'] = weights
            
        self.model = torch.hub.load(repo_path, model_name, **kwargs)
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

