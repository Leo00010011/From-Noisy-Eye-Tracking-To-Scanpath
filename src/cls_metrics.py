import torch
import json
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


def compute_loss(reg_out,cls_out, y, attn_mask, fixation_len):
    criterion_reg = torch.nn.MSELoss()
    criterion_cls = torch.nn.BCEWithLogitsLoss()
    # the end token should not have a regression
    attn_mask_reg = attn_mask.clone()
    batch_idx = torch.arange(cls_out.size()[0])
    attn_mask_reg[batch_idx, fixation_len] = False

    # >>>>>> Classification loss
    # balance the classification loss
    weights = torch.ones(cls_out.size(), dtype = torch.float32, device = reg_out.device)
    div = 1/fixation_len
    div = torch.repeat_interleave(div, repeats=fixation_len, dim=0).unsqueeze(-1)
    weights[attn_mask_reg] = div
    # the end token must be 1, because of the start token the number of fixations points to the end
    cls_targets = torch.zeros(cls_out.size(), dtype = torch.float32, device = reg_out.device)
    cls_targets[batch_idx,fixation_len] = 1.0    
    cls_loss = criterion_cls(cls_out[attn_mask], cls_targets[attn_mask])
    
    # >>>>>> Regression loss
    # reshape the reg_mask
    attn_mask_reg = attn_mask_reg.unsqueeze(-1).expand(-1,-1,3)
    # reshape the attn_mask and remove the start token
    attn_mask = attn_mask.unsqueeze(-1).expand(-1,-1,3)
    attn_mask = attn_mask[:,1:,:]
    reg_loss = criterion_reg(reg_out[attn_mask_reg], y[attn_mask])
    return cls_loss, reg_loss

def plt_training_metrics(path):
    metrics = None
    with open(path, 'r') as f:
        metrics = json.load(f)
    fig, axis = plt.subplots(1,3,figsize=(20,5))
    for k in metrics.keys():
        if k != 'epoch':
            if k == 'regression loss':
                axis[1].plot(metrics[k], label= "reg_loss_train")
            elif k == 'regression_loss':
                axis[1].plot(metrics['epoch'], metrics[k], label="reg_loss_val")
            elif k == 'classification loss':
                axis[2].plot(metrics[k], label="cls_loss_train")
            elif k == 'classification_loss':
                axis[2].plot(metrics['epoch'],metrics[k], label="cls_loss_val")
            else:
                axis[0].plot(metrics['epoch'], metrics[k], label=k)
    fig.tight_layout()

    axis[0].legend()
    axis[1].legend()
    axis[2].legend()
    plt.show()

def validate(model, val_dataloader, epoch, device, metrics, log = True):
    model.eval()
    with torch.no_grad():
        acc_acum = 0
        pre_pos_acum = 0
        rec_pos_acum = 0
        pre_neg_acum = 0
        rec_neg_acum = 0
        cnt = 0
        for batch in val_dataloader:
            x,x_mask,y, y_mask, fixation_len = batch
            x = x.to(device=device)
            y = y.to(device=device)
            if x_mask is not None:
                x_mask = x_mask.to(device = device)
            if y_mask is not None:
                y_mask = y_mask.to(device = device)
            fixation_len = fixation_len.to(device = device)

            reg_out, cls_out = model(x,y, src_mask = x_mask, tgt_mask = y_mask)
            reg_loss, cls_loss = compute_loss(reg_out,cls_out, y, y_mask, fixation_len)
            reg_loss_acum += reg_loss.item()
            cls_loss_acum += cls_loss.item()
            cls_targets = create_cls_targets(cls_out, fixation_len)
            acc_acum += accuracy(cls_out, y_mask, cls_targets)
            pre_pos_acum += precision(cls_out, y_mask, cls_targets)
            rec_pos_acum += recall(cls_out, y_mask, cls_targets)
            pre_neg_acum += precision(cls_out, y_mask, cls_targets, cls = 0)
            rec_neg_acum += recall(cls_out, y_mask, cls_targets, cls = 0)
            cnt += 1
        metrics['epoch'].append(epoch + 1)
        metrics['reg_loss_val'].append(reg_loss_acum / cnt)
        metrics['cls_loss_val'].append(cls_loss_acum / cnt)
        metrics['accuracy'].append(acc_acum / cnt)
        metrics['precision_pos'].append(pre_pos_acum / cnt)
        metrics['recall_pos'].append(rec_pos_acum / cnt)
        metrics['precision_neg'].append(pre_neg_acum / cnt)
        metrics['recall_neg'].append(rec_neg_acum / cnt)
            
        if log:
            print('>>>>>>> Validation results:')
            print('epoch: ',metrics['epoch'][-1])
            print('reg_loss_val: ',metrics['reg_loss_val'][-1])
            print('cls_loss_val: ',metrics['cls_loss_val'][-1])            
            print('accuracy: ',metrics['accuracy'][-1])
            print('precision_pos: ',metrics['precision_pos'][-1])
            print('recall_pos: ',metrics['recall_pos'][-1])
            print('precision_neg: ',metrics['precision_neg'][-1])
            print('recall_neg: ',metrics['recall_neg'][-1])
            print('<<<<<<<<<<<<<<<<<<')
        
    model.train()