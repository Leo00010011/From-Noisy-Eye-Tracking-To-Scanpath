import torch
import matplotlib.pyplot as plt


def create_cls_targets(cls_out, fixation_len):
    batch_idx = torch.arange(cls_out.size()[0])
    cls_targets = torch.zeros(cls_out.size(), 
                              dtype = torch.float32,
                              device = cls_out.device)
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

def eval_reg(reg, y, y_mask):
    y_mask = y_mask.unsqueeze(-1)[:,1:,:]
    count = y_mask.sum().item()
    diff = (reg[:,:-1,:] - y)* y_mask
    diff_xy = diff[:,:,:2]
    reg_error = torch.sqrt(torch.sum(diff_xy**2, dim=-1))
    dur_error = torch.abs(diff[:,:,2])
    reg_error = reg_error.sum().item() / count
    dur_error = dur_error.sum().item() / count
    return reg_error, dur_error

