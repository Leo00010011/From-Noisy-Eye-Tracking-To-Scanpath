import torch
from torch.optim.lr_scheduler import _LRScheduler
import math
import numpy as np
import json
from src.eval.eval_metrics import create_cls_targets, accuracy, precision, recall, eval_reg, eval_denoise
from src.eval.eval_utils import invert_transforms, concat_reg


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
        metrics['epoch'].append(epoch + 1)
        if coord_error_acum > 0:
            metrics['accuracy'].append(acc_acum / cnt)
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
            if coord_error_acum > 0:
                print('coordinate_error_val: ',metrics['reg_error_val'][-1])
                print('duration_error_val: ',metrics['duration_error_val'][-1])
                print('accuracy: ',metrics['accuracy'][-1])
                print('precision_pos: ',metrics['precision_pos'][-1])
                print('recall_pos: ',metrics['recall_pos'][-1])
                print('precision_neg: ',metrics['precision_neg'][-1])
                print('recall_neg: ',metrics['recall_neg'][-1])
            if denoise_coord_error_acum > 0:
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

class WarmupStableDecayScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, stable_steps, decay_steps, min_lr=1e-6, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.min_lr = min_lr
        self.total_steps = warmup_steps + stable_steps + decay_steps
        super(WarmupStableDecayScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        
        # 1. Warmup Phase: Linear increase from 0 to base_lr
        if step < self.warmup_steps:
            factor = float(step) / float(max(1, self.warmup_steps))
            return [base_lr * factor for base_lr in self.base_lrs]
        
        # 2. Stable Phase: Constant base_lr
        # This is where you should ramp up your Scheduled Sampling
        elif step < self.warmup_steps + self.stable_steps:
            return [base_lr for base_lr in self.base_lrs]
        
        # 3. Decay Phase: Cosine decay to min_lr
        else:
            # How far are we into the decay phase?
            decay_step = step - (self.warmup_steps + self.stable_steps)
            decay_step = min(decay_step, self.decay_steps) # Cap it
            
            cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_step / self.decay_steps))
            
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_decay
                for base_lr in self.base_lrs
            ]


def inverted_sigmoid(x,k = 10):
    x_tensor = torch.as_tensor(x, dtype=torch.float32)
    return 1 - k / (k + torch.exp(x_tensor / k))
class ScheduledSampling:
    def __init__(self, active_epochs,
                warmup_epochs,
                device,
                min_prob = 0,
                dtype = torch.float32,
                steps_per_epoch = 128,
                use_kv_cache = False,
                n_updates = -1):
        self.device = device
        self.active_epochs = active_epochs
        self.use_model_prob = 0.0
        self.current_batch = 0
        self.warmup_epochs = warmup_epochs
        self.steps_per_epoch = steps_per_epoch
        self.model = None
        self.use_kv_cache = use_kv_cache
        self.n_updates = n_updates
        self.min_prob = min_prob
        
        self.total_warmup_steps = warmup_epochs*steps_per_epoch
        self.total_active_steps = active_epochs*steps_per_epoch
        
    def set_model(self, model):
        self.model = model

    def step(self):
        if self.current_batch < self.warmup_epochs*self.steps_per_epoch:
            self.current_batch += 1
        elif self.current_batch < (self.warmup_epochs + self.active_epochs)*self.steps_per_epoch:
            self.current_batch += 1
            active_steps = self.current_batch - self.total_warmup_steps
            eval_epoch = (active_steps) / self.steps_per_epoch
            if self.n_updates > 0:
                step_size = self.total_active_steps / self.n_updates
                eval_epoch = ((active_steps // step_size)*step_size)/self.steps_per_epoch
            # normalize to 0-50
            eval_epoch = eval_epoch / self.total_active_steps*50
            prob = inverted_sigmoid(eval_epoch, 10)*(1 - self.min_prob) + self.min_prob
            self.use_model_prob = min(prob,.8)
        else:
            self.use_model_prob = .8

    def get_current_ratio(self):
        return self.use_model_prob

    def get_latest_output(self, output):
        latest_output = {}
        for key, value in output.items():
            if key == 'denoise':
                latest_output[key] = value
            else:
                latest_output[key] = value[:, -1:, :]
        return latest_output
    
    def get_final_output(self, output):
        final_output = {}
        for key in output[0].keys():
            if key == 'denoise':
                final_output[key] = output[-1][key]
            else:
                value = [output[i][key] for i in range(len(output))]
                final_output[key] = torch.concat(value, dim=1)
        return final_output
    
    def __call__(self, **input):
        use_model_prob = self.use_model_prob
        output = None
        input['pass_sampler'] = True
        tgt_mask = input['tgt_mask']
        seq_len = tgt_mask.size(1)
        tgt = input['tgt']
        ori_tgt = tgt
        if 'in_tgt' in input:
            ori_tgt = input['in_tgt']
            input['in_tgt'] = None
        input['tgt'] = None
        input['tgt_mask'] = None
        self.model.encode(**input)
        t = 0
        final_output = []
        has_to_eval = True
        while t < seq_len or has_to_eval:
            has_to_eval = False
            output = self.model(**input) 
            final_output.append(self.get_latest_output(output))
            if t == seq_len - 1:
                break
            reg = concat_reg(output)
            current_step_pred = reg[:, -1:, :] 
            if (torch.rand(1).item() < use_model_prob) or not self.model.training:
                next_token = current_step_pred.detach()
            else:
                next_token = ori_tgt[:, t, :].unsqueeze(1)
                
            if input['tgt'] is None or self.use_kv_cache:
                input['tgt'] = next_token
            else:
                input['tgt'] = torch.concat([input['tgt'], next_token], dim=1)
            t += 1     
        if self.use_kv_cache:
            self.model.clear_kv_cache()
        input['tgt_mask'] = tgt_mask
        input['tgt'] = tgt
        if 'in_tgt' in input:
            input['in_tgt'] = ori_tgt
        output = self.get_final_output(final_output)
        return output

def get_cosine_schedule_alphas_bar(num_steps, s=0.008):
    steps = np.linspace(0, num_steps, num_steps)
    
    # The cosine function ensures a smooth decay to zero
    f_t = np.cos(((steps / num_steps) + s) / (1 + s) * (math.pi / 2)) ** 2
    alphas_cumprod = f_t / f_t[0]
    
    # Ensure it doesn't hit absolute zero too fast for stability
    return np.clip(alphas_cumprod, a_min=0.0001, a_max=None)

class DenoiseDropoutScheduler:
    def __init__(self,base_prob, model, active_epochs, warmup_epochs, steps_per_epoch = 128):
        self.base_prob = base_prob
        self.active_epochs = active_epochs
        self.warmup_epochs = warmup_epochs
        self.steps_per_epoch = steps_per_epoch
        self.probs = get_cosine_schedule_alphas_bar(active_epochs*steps_per_epoch, s = 0.002)
        self.probs = np.sqrt((self.probs))*base_prob
        self.current_step = 1
        self.model = model
        
        
    def step(self):
        # active = (self.warmup_epochs + self.active_epochs)*self.steps_per_epoch
        self.current_step += 1
        
    def get_prob(self):
        if self.current_step < self.warmup_epochs*self.steps_per_epoch:
            return self.base_prob
        elif self.current_step < (self.warmup_epochs + self.active_epochs)*self.steps_per_epoch:
            prev = self.warmup_epochs*self.steps_per_epoch
            return self.probs[self.current_step - prev]
        else:
            return 0
            
    def __call__(self):
        self.model.src_word_dropout.dropout_prob = self.get_prob()
    
    def __repr__(self):
        return f"DenoiseDropoutScheduler(active_epochs={self.active_epochs}, warmup_epochs={self.warmup_epochs}, steps_per_epoch={self.steps_per_epoch})"