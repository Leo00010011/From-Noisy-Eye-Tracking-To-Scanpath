import torch
import torch.nn.functional as F
import torch.nn as nn
import math

epsilon = 1e-7

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


class EndBinaryCrossEntropy(torch.nn.Module):
    def __init__(self, cls_func = torch.nn.functional.binary_cross_entropy_with_logits):
        super().__init__()
        self.cls_func = cls_func
        self.__name__ = 'end_bce_with_logits'
    
    def forward(self, input, output):
        coord_out = output['coord']
        cls_out = output['cls']
        attn_mask = input['tgt_mask']
        fixation_len = input['fixation_len']
        device = coord_out.device
        weights = create_weights(fixation_len, attn_mask, device)
        cls_targets = create_cls_targets(cls_out, fixation_len, device)
        return self.cls_func(cls_out[attn_mask], cls_targets[attn_mask], weight=weights[attn_mask])

class EndSoftMax(torch.nn.Module):
    def __init__(self, cls_func = None):
        super().__init__()
        if cls_func is None:
            cls_func = nn.CrossEntropyLoss()
        self.cls_func = cls_func
        self.__name__ = 'end_softmax'
    
    def forward(self, input, output):
        cls_out = output['cls']
        attn_mask = input['tgt_mask']
        fixation_len = input['fixation_len']
        logits = cls_out.squeeze(-1) 
        logits = logits.masked_fill(~attn_mask, -1e9)
        return self.cls_func(logits, fixation_len)

def MLPLogNormalDistribution(log_normal_mu, log_normal_sigma2, gt):
    # sigma2 is a raw MLP output — clamp to strictly positive before sqrt/division
    sigma2 = F.softplus(log_normal_sigma2) + epsilon
    gt_safe = gt.clamp(min=epsilon)
    logpdf = (- torch.log(gt_safe)
              - 0.5 * torch.log(2 * math.pi * sigma2)
              - (torch.log(gt_safe) - log_normal_mu) ** 2 / (2 * sigma2))
    return -logpdf

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
        # Whether dur_func expects (mu, sigma2, gt, mask) instead of (pred, gt)
        self._dur_is_lognormal = (dur_func is MLPLogNormalDistribution)

    def set_denoise_weight(self, denoise_weight: float):
        return

    def set_weights(self, weights: torch.Tensor):
        self.weights = weights

    def summary(self):
        dur_name = self.dur_func.__name__ if hasattr(self.dur_func, '__name__') else type(self.dur_func).__name__
        coord_name = self.coord_func.__name__ if hasattr(self.coord_func, '__name__') else type(self.coord_func).__name__
        print(f"SeparatedRegLossFunction: cls_weight={self.cls_weight}, dur_weight={self.dur_weight}, "
              f"cls_func={self.cls_func.__name__}, coord_func={coord_name}, dur_func={dur_name}")

    def _dur_loss(self, dur_out, y_dur, seq_mask):
        """
        dur_out : (B, L, 1) for pointwise losses, (B, L, 2) for distribution losses
        y_dur   : (B, L, 1)  — ground-truth duration, already shifted/trimmed
        seq_mask: (B, L)     — bool, True where valid
        """
        if self._dur_is_lognormal:
            # dur_out[:,  :, 0] = mu, dur_out[:, :, 1] = sigma^2
            mu     = dur_out[:, :, 0:1][seq_mask.unsqueeze(-1)].view(-1)
            sigma2 = dur_out[:, :, 1:2][seq_mask.unsqueeze(-1)].view(-1)
            gt     = y_dur[seq_mask.unsqueeze(-1)].view(-1)
            # MLPLogNormalDistribution returns a per-sample loss vector; take the mean
            return MLPLogNormalDistribution(mu, sigma2, gt).mean()
        else:
            flat_mask = seq_mask.unsqueeze(-1).expand_as(dur_out[:, :, :1])
            return self.dur_func(dur_out[:, :, :1][flat_mask], y_dur[flat_mask])

    def forward(self, input, output):
        coord_out = output['coord']   # (B, L, 2)
        dur_out   = output['dur']     # (B, L, 1) or (B, L, 2)
        cls_out   = output['cls']
        y         = input['tgt']      # (B, L, 3)  — [x, y, dur]
        attn_mask = input['tgt_mask'] # (B, L)
        fixation_len = input['fixation_len']
        device = coord_out.device

        if not hasattr(self, '_debug_printed'):
            self._debug_printed = True
            print("[DEBUG] tgt nan:", y.isnan().any().item(), "| tgt inf:", y.isinf().any().item())
            print("[DEBUG] coord_out nan:", coord_out.isnan().any().item(), "| coord_out inf:", coord_out.isinf().any().item())
            print("[DEBUG] dur_out nan:", dur_out.isnan().any().item(), "| dur_out inf:", dur_out.isinf().any().item())
            print("[DEBUG] cls_out nan:", cls_out.isnan().any().item(), "| cls_out inf:", cls_out.isinf().any().item())
            print("[DEBUG] tgt min/max:", y.min().item(), y.max().item())
            print("[DEBUG] dur_out min/max:", dur_out.min().item(), dur_out.max().item())

        if attn_mask is None:
            print("No attention mask provided")
            attn_mask = torch.ones(cls_out.size()[:-1], dtype=torch.bool, device=device)
            input['tgt_mask'] = attn_mask

        # >>>>>> Classification loss
        cls_loss = self.cls_func(input, output)

        # Shift: predictions at t predict targets at t+1, so drop the last pred and first target.
        # seq_mask: (B, L-1), valid positions after the shift
        seq_mask  = attn_mask[:, 1:]          # (B, L-1)
        coord_pred = coord_out[:, :-1, :]     # (B, L-1, 2)
        dur_pred   = dur_out[:, :-1, :]       # (B, L-1, 1 or 2)
        y_coord    = y[:, :, :2]              # (B, L-1, 2)
        y_dur      = y[:, :, 2:]              # (B, L-1, 1)

        # >>>>>> Duration loss
        dur_loss = self._dur_loss(dur_pred, y_dur, seq_mask)

        # >>>>>> Coordinate loss
        coord_mask = seq_mask.unsqueeze(-1).expand_as(coord_pred)   # (B, L-1, 2)
        coord_weights = None
        if self.weights is not None:
            w = self.weights[:seq_mask.size(1)]
            coord_weights = w.unsqueeze(-1).unsqueeze(0).expand_as(coord_pred)
            coord_weights = coord_weights[coord_mask]
        coord_loss = self.coord_func(coord_pred[coord_mask], y_coord[coord_mask], weight=coord_weights)

        info = {
            'cls_loss':   float(cls_loss.item()),
            'coord_loss': float(coord_loss.item()),
            'dur_loss':   float(dur_loss.item()),
        }
        reg_loss = (1 - self.dur_weight) * coord_loss + self.dur_weight * dur_loss
        loss = self.cls_weight * cls_loss + (1 - self.cls_weight) * reg_loss
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
    
class PenaltyReducedFocalLoss(nn.Module):
    def __init__(self, alpha=2.0,
                 beta=4.0,
                 cls_func = torch.nn.functional.binary_cross_entropy_with_logits,
                 cls_weight = 0.5,
                 dur_func = torch.nn.functional.l1_loss,
                 dur_weight = 0.5):
        """
        Implementation of Eq (4) from the image.
        
        Args:
            alpha (float): Power factor for the prediction error (focal term).
            beta (float): Power factor for the distance from ground truth center.
                          Higher beta = less penalty for near-misses.
            reduction (str): 'mean' (divides by HW as in paper) or 'sum'.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.cls_func = cls_func
        self.cls_weight = cls_weight
        self.dur_func = dur_func
        self.dur_weight = dur_weight

    def set_denoise_weight(self, denoise_weight: float):
        return
    
    def summary(self):
        print(f"PenaltyReducedFocalLoss: alpha={self.alpha}, beta={self.beta}, cls_weight={self.cls_weight}, dur_weight={self.dur_weight}, cls_func={self.cls_func.__name__}, dur_func={self.dur_func.__name__}")
    
    def forward(self, input, output):
        """
        Args:
            logits: (B, C, H, W) or (B, L, H*W) - Raw model outputs (before Sigmoid).
            targets: (B, C, H, W) or (B, L, H*W) - Ground truth Gaussian heatmap.
                     IMPORTANT: Peaks must be exactly 1.0. Values in [0, 1].
        
        Returns:
            loss: Scalar loss
        """
        logits = output['heatmaps']
        targets = input['heatmaps']
        dur_out = output['dur']
        cls_out = output['cls']
        y = input['tgt']
        attn_mask = input['tgt_mask']
        fixation_len = input['fixation_len']
        device = logits.device
        
      
        
        if attn_mask is None:
            print("No attention mask provided")
            attn_mask = torch.ones(cls_out.size()[:-1], dtype = torch.bool, device = device)
        # >>>>>> Classification loss
        weights = create_weights(fixation_len, attn_mask, device)
        cls_targets = create_cls_targets(cls_out, fixation_len, device)
        cls_loss = self.cls_func(cls_out[attn_mask], cls_targets[attn_mask], weight=weights[attn_mask])
        
        attn_mask = attn_mask.unsqueeze(-1)
        attn_mask = attn_mask[:,1:,:]
        dur_loss = self.dur_func(dur_out[:,:-1,:][attn_mask], y[:,:,2:][attn_mask])
        
        attn_mask = attn_mask.squeeze(-1)
        logits = logits[:,:-1,:][attn_mask]
        targets = targets[attn_mask]
        preds = torch.sigmoid(logits)
        preds = torch.clamp(preds, min=1e-4, max=1 - 1e-4)

        epsilon = 2e-1
        pos_inds = (targets >= (1.0 - epsilon)).float()
        neg_inds = 1 - pos_inds
        pos_loss = torch.log(preds) * torch.pow(1 - preds, self.alpha) * pos_inds

        neg_loss = torch.log(1 - preds) * torch.pow(preds, self.alpha) * \
                   torch.pow(1 - targets, self.beta) * neg_inds

        num_pixels = preds.numel() # H * W * B ...
        
        coord_loss = 0
        coord_loss = coord_loss - (pos_loss + neg_loss).sum()
        coord_loss = coord_loss / num_pixels
        
        reg_loss = self.dur_weight * dur_loss + (1 - self.dur_weight) * coord_loss
        total_loss = self.cls_weight * cls_loss + (1 - self.cls_weight) * reg_loss
        info = {
            'cls_loss': float(cls_loss.item()),
            'coord_loss': float(coord_loss.item()),
            'dur_loss': float(dur_loss.item()),
            'total_loss': float(total_loss.item()),
        }
        return total_loss, info
        
