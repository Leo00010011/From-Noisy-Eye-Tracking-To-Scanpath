import torch
import torch.nn.functional as F
import torch.nn as nn

def create_weights(fixation_len, attn_mask, device):
    weights = torch.ones(attn_mask.size(), dtype = torch.float32, device = device)
    div = 1/fixation_len
    div = torch.repeat_interleave(div, repeats=fixation_len, dim=0)
    weights[:,:-1][attn_mask[:,1:]] = div
    return weights.unsqueeze(-1)

def create_cls_targets(cls_out, fixation_len, device):
    batch_idx = torch.arange(cls_out.size()[0])
    cls_targets = torch.zeros(cls_out.size(), dtype = torch.float32, device = device)
    cls_targets[batch_idx,fixation_len] = 1.0    
    return cls_targets

class EntireRegLossFunction(torch.nn.Module):
    def __init__(self, cls_weight = 0.5, 
                 cls_func = torch.nn.functional.binary_cross_entropy_with_logits, 
                 reg_func = torch.nn.functional.mse_loss):
        super().__init__()
        self.cls_weight = cls_weight
        self.cls_func = cls_func
        self.reg_func = reg_func
    
    def set_denoise_weight(self, denoise_weight: float):
        return
    
    def summary(self):
        print(f"EntireRegLossFunction: cls_weight={self.cls_weight}, cls_func={self.cls_func.__name__}, reg_func={self.reg_func.__name__}")

    def forward(self, input, output):
        reg_out = output['reg']
        cls_out = output['cls']
        y = input['tgt']
        attn_mask = input['tgt_mask']
        fixation_len = input['fixation_len']
        device = reg_out.device
        # case where all the fixations have the same size 
        if attn_mask is None:
            print("No attention mask provided")
            attn_mask = torch.ones(cls_out.size()[:-1], dtype = torch.bool, device = device)
        # >>>>>> Classification loss
        weights = create_weights(fixation_len, attn_mask, device)
        cls_targets = create_cls_targets(cls_out, fixation_len, device)
        cls_loss = self.cls_func(cls_out[attn_mask], cls_targets[attn_mask], weight=weights[attn_mask])
        
        # >>>>>> Regression loss
        attn_mask = attn_mask.unsqueeze(-1).expand(-1,-1,3)
        attn_mask = attn_mask[:,1:,:]
        reg_loss = self.reg_func(reg_out[:,:-1,:][attn_mask], y[attn_mask])
        info = {
            'cls_loss': float(cls_loss.item()),
            'reg_loss': float(reg_loss.item()),
        }
        loss = self.cls_weight * cls_loss + (1 - self.cls_weight) * reg_loss
        return loss, info
    
class SeparatedRegLossFunction(torch.nn.Module):
    def __init__(self, cls_weight = 0.5, dur_weight = 0.5,
                 cls_func = torch.nn.functional.binary_cross_entropy_with_logits, 
                 coord_func = torch.nn.functional.mse_loss,
                 dur_func = torch.nn.functional.mse_loss):
        super().__init__()
        self.cls_weight = cls_weight
        self.dur_weight = dur_weight
        self.cls_func = cls_func
        self.coord_func = coord_func
        self.dur_func = dur_func
        self.weights = None
        
    def set_denoise_weight(self, denoise_weight: float):
        return
    
    def set_weights(self, weights: torch.Tensor):
        self.weights = weights
    
    def summary(self):
        print(f"SeparatedRegLossFunction: cls_weight={self.cls_weight}, dur_weight={self.dur_weight}, cls_func={self.cls_func.__name__}, coord_func={self.coord_func.__name__}, dur_func={self.dur_func.__name__}")

    def forward(self, input, output):
        coord_out = output['coord']
        dur_out = output['dur']
        cls_out = output['cls']
        y = input['tgt']
        attn_mask = input['tgt_mask']
        fixation_len = input['fixation_len']
        device = coord_out.device
        # case where all the fixations have the same size 
        if attn_mask is None:
            print("No attention mask provided")
            attn_mask = torch.ones(cls_out.size()[:-1], dtype = torch.bool, device = device)
        # >>>>>> Classification loss
        weights = create_weights(fixation_len, attn_mask, device)
        cls_targets = create_cls_targets(cls_out, fixation_len, device)
        cls_loss = self.cls_func(cls_out[attn_mask], cls_targets[attn_mask], weight=weights[attn_mask])
        
        # >>>>>> Regression loss
        attn_mask = attn_mask.unsqueeze(-1)
        attn_mask = attn_mask[:,1:,:]
        dur_loss = self.dur_func(dur_out[:,:-1,:][attn_mask], y[:,:,2:][attn_mask])
        attn_mask = attn_mask.expand(-1,-1,2)
        coord_weights = None
        if self.weights is not None:
            coord_weights = self.weights[:attn_mask.size(1)]
            coord_weights = coord_weights.unsqueeze(-1).unsqueeze(0).expand(attn_mask.size())
            coord_weights = coord_weights[attn_mask]
        coord_loss = self.coord_func(coord_out[:,:-1,:][attn_mask], y[:,:,:2][attn_mask], weight=coord_weights)
        info = {
            'cls_loss': float(cls_loss.item()),
            'coord_loss': float(coord_loss.item()),
            'dur_loss': float(dur_loss.item()),
        }
        loss = cls_loss + self.cls_weight * (((1-self.dur_weight) * coord_loss + self.dur_weight * dur_loss))
        return loss, info

class DenoiseRegLoss(torch.nn.Module):
    def __init__(self, denoise_loss):
        super().__init__()
        self.denoise_loss = denoise_loss
    
    def summary(self):
        loss_name = self.denoise_loss.__name__ if hasattr(self.denoise_loss, '__name__') else type(self.denoise_loss).__name__
        print(f"DenoiseRegLoss: denoise_loss={loss_name}")

    def forward(self, input, output):
        loss = self.denoise_loss(output['denoise'], input['clean_x'][:,:,:2])
        info = {'denoise_loss': float(loss.item())}
        return loss, info
    
class CombinedLossFunction(torch.nn.Module):
    def __init__(self, denoise_loss, fixation_loss, denoise_weight = 0):
        super().__init__()
        self.denoise_loss = denoise_loss
        self.fixation_loss = fixation_loss
        self.denoise_weight = denoise_weight
        
    def set_denoise_weight(self, denoise_weight: float):
        self.denoise_weight = denoise_weight
    
    def summary(self):
        print(f"CombinedLossFunction: denoise_weight={self.denoise_weight}")
        print("  Denoise Loss:")
        if hasattr(self.denoise_loss, 'summary'):
            self.denoise_loss.summary()
        else:
            print(f"    {self.denoise_loss}")
        print("  Fixation Loss:")
        if hasattr(self.fixation_loss, 'summary'):
            self.fixation_loss.summary()
        else:
            print(f"    {self.fixation_loss}")

    def forward(self, input, output):
        denoise_loss = 0
        fixation_loss = 0
        info = {}
        if self.denoise_weight > 0:
            denoise_loss, denoise_info = self.denoise_loss(input, output)
            info['denoise_loss'] = float(denoise_loss.item())
            info.update(denoise_info)
        
        if (1 - self.denoise_weight) > 0:
            fixation_loss, fixation_info = self.fixation_loss(input, output)
            info['fixation_loss'] = float(fixation_loss.item())
            info.update(fixation_info)
        
        loss = self.denoise_weight * denoise_loss + (1 - self.denoise_weight) * fixation_loss
        # Build info dict only with relevant losses and subdicts
        return loss, info
    
class SpatialFocalLossWithLogits(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (float): Weighting factor for the rare class (the target pixel).
            gamma (float): Focusing parameter. Higher values reduce loss for easy examples.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, L, Num_Pixels) - Raw output from the model.
            targets: (B, L) - Flattened indices of the ground truth locations.
                              Values should be in range [0, Num_Pixels - 1].
        """
        # 1. Compute Log Softmax for numerical stability
        # log_pt shape: (B, L, Num_Pixels)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 2. Gather the log_prob of the true target class
        # We look up the value at the target index for each step in the sequence
        # targets.unsqueeze(-1) shape: (B, L, 1)
        target_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1))
        
        # Remove the extra dim -> (B, L)
        target_log_probs = target_log_probs.squeeze(-1)
        
        # 3. Compute Pt (Probability of true class)
        pt = torch.exp(target_log_probs)
        
        # 4. Focal Loss Formula
        # Loss = -alpha * (1 - pt)^gamma * log(pt)
        focal_term = (1 - pt) ** self.gamma
        loss = -self.alpha * focal_term * target_log_probs

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
    
