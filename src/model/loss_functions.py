import torch

def create_weights(fixation_len, attn_mask, device):
    weights = torch.ones(attn_mask.size(), dtype = torch.float32, device = device)
    div = 1/fixation_len
    div = torch.repeat_interleave(div, repeats=fixation_len, dim=0)
    weights[:,:-1][attn_mask[:,1:]] = div
    return weights

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
    
    def forward(self, input, output):
        reg_out = output['reg']
        cls_out = output['cls']
        y = input['tgt']
        attn_mask = input['tgt_mask']
        fixation_len = input['fixation_len']
        device = reg_out.device
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
        loss = cls_loss + self.cls_weight * reg_loss
        return loss, info