import torch
import numpy as np
import json
from src.eval.eval_metrics import create_cls_targets, accuracy, precision, recall, eval_reg, eval_denoise
from src.eval.eval_utils import invert_transforms

class MetricsStorage:
    def __init__(self, filepath: str = None, decisive_metric: str = 'reg_loss_val'):
        self.metrics = {
            'epoch': [],
            'reg_error_val': [],
            'duration_error_val': [],
            'outliers_count_val': [],
            'accuracy': [],
            'precision_pos': [],
            'recall_pos': [],
            'precision_neg': [],
            'recall_neg': [],
            'denoise_error_val': []
        }
        self.loss_info = {}
        self.num_batches = 0
        self.filepath = filepath
        self.decisive_metric = decisive_metric
        self.best_metric_value = np.inf
    
    def init_epoch(self):
        self.loss_info = {}
        self.num_batches = 0
        
    def compute_normalized_regression_metrics(self,input, output, dataloadeer ):
        input, output = invert_transforms(input, output, dataloadeer)
        results_dict = {}
        if 'reg' in output:
            reg_out = output['reg']
            y, y_mask = input['tgt'], input['tgt_mask']
            coord_error, dur_error = eval_reg(reg_out, y, y_mask)
            results_dict['coord_error'] = coord_error
            results_dict['dur_error'] = dur_error
        # FIX ME: Sub pixel average denoise error???
        if 'denoise' in output:
            denoise_out = output['denoise']
            clean_x = input['clean_x']
            coord_error = eval_denoise(denoise_out, clean_x)
            results_dict['denoise_error'] = coord_error
        
        for key, value in results_dict.items():
            if key not in self.loss_info:
                self.loss_info[key] = value
            else:
                self.loss_info[key] += value

    def update_batch_loss(self, info: dict):
        for key, value in info.items():
            if key not in self.loss_info:
                self.loss_info[key] = value
            else:
                self.loss_info[key] += value
        self.num_batches += 1
    
    def finalize_epoch(self):
        agg_loss_info = {}
        for key, value in self.loss_info.items():
            key_str = f'{key}_train'
            if key_str not in self.metrics:
                self.metrics[key_str] = []
            avg_loss = value / self.num_batches
            self.metrics[key_str].append(avg_loss)
            agg_loss_info[key] = avg_loss
        return agg_loss_info
    
    def update_best(self):
        if self.metrics[self.decisive_metric][-1] < self.best_metric_value:
            self.best_metric_value = self.metrics[self.decisive_metric][-1]
            return True
        return False

    def save_metrics(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.metrics, f)


def validate(model, loss_fn, val_dataloader, epoch, device, metrics, log = True):
    model.eval()
    with torch.no_grad():
        denoise_coord_error_acum = 0
        acc_acum = 0
        pre_pos_acum = 0
        rec_pos_acum = 0
        pre_neg_acum = 0
        rec_neg_acum = 0
        outliers_count_acum = 0
        coord_error_acum = 0
        duration_error_acum = 0
        loss_info = {}
        cnt = 0
        for batch in val_dataloader:
            input = move_data_to_device(batch, device)

            output = model(**input)
            _ , info = loss_fn(input, output)
            for key, value in info.items():
                if key not in loss_info:
                    loss_info[key] = value
                else:
                    loss_info[key] += value
            input, output = invert_transforms(input, output, val_dataloader, remove_outliers = True)
            if 'reg' in output:
                reg_out = output['reg']
                cls_out = output['cls']
                y = input['tgt']
                y_mask = input['tgt_mask']
                fixation_len = input['fixation_len']
                
                reg_error, duration_error = eval_reg(reg_out, y, y_mask)
                coord_error_acum += reg_error
                duration_error_acum += duration_error
                outliers_count_acum += output['outliers_count']
                cls_targets = create_cls_targets(cls_out, fixation_len)
                acc_acum += accuracy(cls_out, y_mask, cls_targets)
                pre_pos_acum += precision(cls_out, y_mask, cls_targets)
                rec_pos_acum += recall(cls_out, y_mask, cls_targets)
                pre_neg_acum += precision(cls_out, y_mask, cls_targets, cls = 0)
                rec_neg_acum += recall(cls_out, y_mask, cls_targets, cls = 0)
            if 'denoise' in output:
                denoise_out = output['denoise']
                clean_x = input['clean_x']
                coord_error = eval_denoise(denoise_out, clean_x)
                denoise_coord_error_acum += coord_error
            cnt += 1
        for key, value in loss_info.items():
            key_str = f'{key}_val'
            if key_str not in metrics:
                metrics[key_str] = []
            metrics[key_str].append(value / cnt)
        if coord_error_acum > 0:
            metrics['accuracy'].append(acc_acum / cnt)
            metrics['epoch'].append(epoch + 1)
            metrics['reg_error_val'].append(coord_error_acum / cnt)
            metrics['duration_error_val'].append(duration_error_acum / cnt)
            metrics['outliers_count_val'].append(outliers_count_acum)
            metrics['precision_pos'].append(pre_pos_acum / cnt)
            metrics['recall_pos'].append(rec_pos_acum / cnt)
            metrics['precision_neg'].append(pre_neg_acum / cnt)
            metrics['recall_neg'].append(rec_neg_acum / cnt)
        if denoise_coord_error_acum > 0:
            metrics['denoise_error_val'].append(denoise_coord_error_acum / cnt)
        if log:
            print(f'>>>>>>> Validation results at epoch {metrics["epoch"][-1]}:')
            for key, value in info.items():
                print(f'{key}_val: {value}')
            if 'reg_error_val' in metrics:
                print('coordinate_error_val: ',metrics['reg_error_val'][-1])
                print('duration_error_val: ',metrics['duration_error_val'][-1])
                print('accuracy: ',metrics['accuracy'][-1])
                print('precision_pos: ',metrics['precision_pos'][-1])
                print('recall_pos: ',metrics['recall_pos'][-1])
                print('precision_neg: ',metrics['precision_neg'][-1])
                print('recall_neg: ',metrics['recall_neg'][-1])
            if 'denoise_error_val' in metrics:
                print('denoise_error_val: ',metrics['denoise_error_val'][-1])
            
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
    if attn_mask is None:
        attn_mask = torch.ones(cls_out.size()[:-1], dtype = torch.bool, device = reg_out.device)
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

