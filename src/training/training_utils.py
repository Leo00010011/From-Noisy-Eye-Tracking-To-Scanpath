import torch
import numpy as np
import json
from src.eval.eval_metrics import create_cls_targets, accuracy, precision, recall, eval_reg

class MetricsStorage:
    def __init__(self, filepath: str = None, decisive_metric: str = 'reg_loss_val'):
        self.metrics = {
            'epoch': [],
            'reg_loss_train': [],
            'reg_error_val': [],
            'duration_error_val': [],
            'reg_loss_val': [],
            'cls_loss_train': [],
            'cls_loss_val': [],
            'accuracy': [],
            'precision_pos': [],
            'recall_pos': [],
            'precision_neg': [],
            'recall_neg': []
        }
        self.total_reg_loss = 0
        self.total_cls_loss = 0
        self.num_batches = 0
        self.filepath = filepath
        self.decisive_metric = decisive_metric
        self.best_metric_value = np.inf
    
    def init_epoch(self):
        self.total_reg_loss = 0
        self.total_cls_loss = 0
        self.num_batches = 0

    def update_batch_loss(self, reg_loss, cls_loss):
        self.total_reg_loss += reg_loss.item()
        self.total_cls_loss += cls_loss.item()
        self.num_batches += 1
    
    def finalize_epoch(self):
        avg_reg_loss = self.total_reg_loss / self.num_batches
        avg_cls_loss = self.total_cls_loss / self.num_batches
        self.metrics['reg_loss_train'].append(avg_reg_loss)
        self.metrics['cls_loss_train'].append(avg_cls_loss)
        return avg_reg_loss, avg_cls_loss
    
    def update_best(self):
        if self.metrics[self.decisive_metric][-1] < self.best_metric_value:
            self.best_metric_value = self.metrics[self.decisive_metric][-1]
            return True
        return False

    def save_metrics(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.metrics, f)


def validate(model, val_dataloader, epoch, device, metrics, log = True):
    model.eval()
    with torch.no_grad():
        acc_acum = 0
        pre_pos_acum = 0
        rec_pos_acum = 0
        pre_neg_acum = 0
        rec_neg_acum = 0
        reg_loss_acum = 0
        cls_loss_acum = 0
        reg_error_acum = 0
        duration_error_acum = 0
        cnt = 0
        for batch in val_dataloader:
            input = move_data_to_device(batch, device)

            output = model(**input)
            cls_loss, reg_loss = compute_loss(input, output)
            reg_loss_acum += reg_loss.item()
            cls_loss_acum += cls_loss.item()
            cls_out = output['cls']
            reg_out = output['reg']
            y = input['tgt']
            y_mask = input['tgt_mask']
            fixation_len = input['fixation_len']
            cls_targets = create_cls_targets(cls_out, fixation_len)
            acc_acum += accuracy(cls_out, y_mask, cls_targets)
            pre_pos_acum += precision(cls_out, y_mask, cls_targets)
            rec_pos_acum += recall(cls_out, y_mask, cls_targets)
            pre_neg_acum += precision(cls_out, y_mask, cls_targets, cls = 0)
            rec_neg_acum += recall(cls_out, y_mask, cls_targets, cls = 0)
            reg_error, duration_error = eval_reg(reg_out, y, y_mask)
            reg_error_acum += reg_error
            duration_error_acum += duration_error
            cnt += 1
        metrics['epoch'].append(epoch + 1)
        metrics['reg_error_val'].append(reg_error_acum / cnt)
        metrics['duration_error_val'].append(duration_error_acum / cnt)
        metrics['reg_loss_val'].append(reg_loss_acum / cnt)
        metrics['cls_loss_val'].append(cls_loss_acum / cnt)
        metrics['accuracy'].append(acc_acum / cnt)
        metrics['precision_pos'].append(pre_pos_acum / cnt)
        metrics['recall_pos'].append(rec_pos_acum / cnt)
        metrics['precision_neg'].append(pre_neg_acum / cnt)
        metrics['recall_neg'].append(rec_neg_acum / cnt)
            
        if log:
            print(f'>>>>>>> Validation results at epoch {metrics["epoch"][-1]}:')
            print('reg_loss_val: ',metrics['reg_loss_val'][-1])
            print('cls_loss_val: ',metrics['cls_loss_val'][-1])  
            print('reg_error_val: ',metrics['reg_error_val'][-1])
            print('duration_error_val: ',metrics['duration_error_val'][-1])
            print('accuracy: ',metrics['accuracy'][-1])
            print('precision_pos: ',metrics['precision_pos'][-1])
            print('recall_pos: ',metrics['recall_pos'][-1])
            print('precision_neg: ',metrics['precision_neg'][-1])
            print('recall_neg: ',metrics['recall_neg'][-1])
            print('<<<<<<<<<<<<<<<<<<')
    model.train()


def compute_loss(input, output):
    # TODO Warning on natural exponential in MSE
    reg_out = output['reg']
    cls_out = output['cls']
    y = input['tgt']
    attn_mask = input['tgt_mask']
    fixation_len = input['fixation_len']
    
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
    # reg_loss = criterion_reg(reg_out[attn_mask_reg], y[attn_mask])
    reg_loss = torch.nn.functional.mse_loss(reg_out[attn_mask_reg], y[attn_mask])
    return cls_loss, reg_loss

def move_data_to_device(batch, device):
    device_batch = {}
    for key, item in batch.items():
        if item is None:
            device_batch[key] = None
        else:
            device_batch[key] = item.to(device=device)
    return device_batch

