import torch


def create_cls_targets(cls_out, fixation_len):
    batch_idx = torch.arange(cls_out.size()[0])
    cls_targets = torch.zeros(cls_out.size(), dtype = torch.float32)
    cls_targets[batch_idx,fixation_len] = 1.0
    return cls_targets

def accuracy(cls_out, attn_mask, cls_targets):
    cls_preds = torch.sigmoid(cls_out) >= 0.5
    attn_mask = attn_mask.unsqueeze(-1)
    correct = (cls_preds == cls_targets) & attn_mask
    accuracy = correct.sum().item() / attn_mask.sum().item()
    return accuracy

def precision(cls_out, attn_mask, cls_targets, cls = 1):
    cls_preds = torch.sigmoid(cls_out) >= 0.5
    attn_mask = attn_mask.unsqueeze(-1)
    true_positives = ((cls_preds == cls) & (cls_targets == cls) & attn_mask).sum().item()
    predicted_positives = ((cls_preds == cls) & attn_mask).sum().item()
    precision = true_positives / predicted_positives if predicted_positives > 0 else 0.0
    return precision

def recall(cls_out, attn_mask, cls_targets, cls = 1):
    cls_preds = torch.sigmoid(cls_out) >= 0.5
    attn_mask = attn_mask.unsqueeze(-1)
    true_positives = ((cls_preds == cls) & (cls_targets == cls) & attn_mask).sum().item()
    actual_positives = ((cls_targets == cls) & attn_mask).sum().item()
    recall = true_positives / actual_positives if actual_positives > 0 else 0.0
    return recall


def compute_loss(reg_out,cls_out, y, attn_mask, fixation_len):
    criterion_reg = torch.nn.MSELoss()
    criterion_cls = torch.nn.BCEWithLogitsLoss()
    # the end token should not have a regression
    attn_mask_reg = attn_mask.clone()
    batch_idx = torch.arange(cls_out.size()[0])
    attn_mask_reg[batch_idx, fixation_len] = False

    # >>>>>> Classification loss
    # balance the classification loss
    weights = torch.ones(cls_out.size(), dtype = torch.float32)
    div = 1/fixation_len
    div = torch.repeat_interleave(div, repeats=fixation_len, dim=0).unsqueeze(-1)
    weights[attn_mask_reg] = div
    # the end token must be 1, because of the start token the number of fixations points to the end
    cls_targets = torch.zeros(cls_out.size(), dtype = torch.float32)
    cls_targets[batch_idx,fixation_len] = 1.0    
    cls_loss = criterion_cls(cls_out[attn_mask], cls_targets[attn_mask])
    
    # >>>>>> Regression loss
    # reshape the reg_mask
    attn_mask_reg = attn_mask_reg.unsqueeze(-1).expand(-1,-1,3)
    # reshape the attn_mask and remove the start token
    attn_mask = attn_mask.unsqueeze(-1).expand(-1,-1,3)
    attn_mask = attn_mask[:,1:,:]
    reg_loss = criterion_reg(reg_out[attn_mask_reg], y[attn_mask])
     # Example target: 1 if point exists, else 0
    return cls_loss, reg_loss