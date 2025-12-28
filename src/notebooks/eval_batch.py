# %% [markdown]
# # Imports

# %%
import os
import torch
from tqdm import tqdm
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import sys
import gc

if not os.path.exists('data'):
    new_directory_path = "..\\..\\"
    os.chdir(new_directory_path)
print("Current working directory:", os.getcwd())
sys.path.append(os.getcwd())

from src.eval.eval_metrics import precision,recall,create_cls_targets, accuracy, eval_reg
from src.eval.eval_utils import plt_training_metrics, gather_best_metrics, invert_transforms, eval_autoregressive
from src.eval.vis_scanpath import draw_scanpath_mpl
from src.model.model_io import load_models_with_data
from src.training.training_utils import move_data_to_device, compute_loss


# %% [markdown]
# ## Methods

# %%
def get_coords(idx, x,y, fixation_len, reg_out):
    x_coords = x[idx,:,:2].cpu().numpy().T
    y_coords = y[idx,:fixation_len[idx],:2].cpu().numpy().T
    reg_coords = reg_out[idx,:fixation_len[idx],:2].cpu().numpy().T
    x_coords = x_coords[:,::-1]
    y_coords = y_coords[:,::-1]
    reg_coords = reg_coords[:,::-1]
    return x_coords, y_coords, reg_coords

def batch_to_list(x, fixation_len = None):
    # convert from padded batch [B,T,F] to list of numpy arrays with shape [F,N]
    x_list = []
    for i in range(x.size(0)):
        l = 0
        if fixation_len is None:
            l = x.size(1)
        else:
            l = fixation_len[i]
        x_i = x[i,:l, :].cpu().numpy().T
        x_list.append(x_i)
    return x_list

def plot_classification_scores(cls_out,fixation_len, title="Classification Scores"):
    acum = np.zeros(7)
    count = 0
    skipped_count = 0
    for i in range(cls_out.shape[0]):
        cls_out_sample = cls_out[i]
        cls_out_sample = torch.sigmoid(cls_out_sample).cpu().numpy().T
        if fixation_len[i] < 3 or fixation_len[i] > cls_out_sample.shape[1]-4:
            skipped_count += 1
            continue
        acum += cls_out_sample[0,fixation_len[i]-3: fixation_len[i]+4 ]
        count +=1
    avg = acum / count
    image_data = avg.reshape(7, 1).T
    plt.figure(figsize=(7,1)) # Set the figure size to be tall and thin
    plt.imshow(image_data, cmap='gray', interpolation='nearest', aspect='auto')
    text_color = 'green' 
    for i in range(image_data.shape[0]):
        for j in range(image_data.shape[1]):
            score = image_data[i, j]
            
            plt.text(j, i, f'{score:.2f}', 
                     ha="center", va="center", color=text_color, fontsize=10)
    print(f'Skipped samples: {skipped_count} out of {cls_out.shape[0]}')
    plt.title(title)
    plt.xticks(range(7), [f'Token {i-3}' if i != 3 else 'END' for i in range(7)]) # Optional: keep y-axis labels for reference
    plt.yticks([])
    plt.show()
    
def slim_input_output(input, output):
    slim_input = {
        'src': input['src'],
        'tgt': input['tgt'],
        'tgt_mask': input['tgt_mask'],
        'fixation_len': input['fixation_len'],
    }
    slim_output = {
        'reg': output['reg'],
        'cls': output['cls'],
    }
    return slim_input, slim_output

# %% [markdown]
# # Eval

# %% [markdown]
# ## Review Metrics

# %%
# names = ['path model',
#         'W/o Image -> Eye attn',
#         'Mixer model with enh img features',
#         'W Image-> Eye attn']
# 
# 
# 
# ckpt_path = [os.path.join('outputs','2025-12-10','15-57-12'),
#              os.path.join('outputs','2025-12-18','12-35-29'),
#              os.path.join('outputs','2025-12-18','16-20-34'),
#              os.path.join('outputs','2025-12-23','15-57-34')]


# %%
names = ['path model',
        'src dropout path model',
        'multi head mixer model',
        'drop multi head',
        'enh image mixer model',
        'linear enh mixer model']



ckpt_path = [os.path.join('outputs','2025-12-10','15-57-12'),
             os.path.join('outputs','2025-12-27','19-45-53'),
             os.path.join('outputs','2025-12-27','23-44-04'),
             os.path.join('outputs','2025-12-28','11-30-12'),
             os.path.join('outputs','2025-12-28','12-12-12'),
             os.path.join('outputs','2025-12-28','12-49-51')]


# %% [markdown]
# ## Checkout Output

# %% [markdown]
# ### Eval Just One Batch

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
inputs_outputs = []

models_and_data = load_models_with_data(ckpt_path)
print(f'Model {names[0]}')
for i, ((model, _, _, test_dataloader), ckpt_path, name) in enumerate(zip(models_and_data, ckpt_path, names)):    
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'image_encoder'):
            print('rope reg: ', model.image_encoder.model.rope_embed.training)
        else:
            print('No rope reg')
        current_model = {
            'checkpoint_path': ckpt_path,
            'model_name': name,
            'inputs': [],
            'outputs': [],
        }
        acc_acum = 0
        cls_loss_acum = 0
        reg_loss_acum = 0
        pre_pos_acum = 0
        rec_pos_acum = 0
        pre_neg_acum = 0
        rec_neg_acum = 0
        outliers_count_acum = 0
        coord_error_acum = 0
        duration_error_acum = 0
        count = 0
        
        for batch in tqdm(test_dataloader):
            input = move_data_to_device(batch, device)
            output = eval_autoregressive(model, input, only_last = True)
            input, output = slim_input_output(input, output)
            input, output = invert_transforms(input, output, test_dataloader, remove_outliers = True)
            current_model['inputs'].append(input)
            current_model['outputs'].append(output)
            reg_out, cls_out = output['reg'], output['cls']
            y, y_mask, fixation_len = input['tgt'], input['tgt_mask'], input['fixation_len']
            cls_loss, reg_loss = compute_loss(input, output)
            outliers_count_acum += output['outliers_count']
            cls_loss_acum += cls_loss.item()
            reg_loss_acum += reg_loss.item()
            coord_error, dur_error = eval_reg(reg_out, y, y_mask)
            coord_error_acum += coord_error
            duration_error_acum += dur_error
            cls_targets = create_cls_targets(cls_out, fixation_len)
            acc_acum += accuracy(cls_out, y_mask, cls_targets)
            pre_pos_acum += precision(cls_out, y_mask, cls_targets)
            rec_pos_acum += recall(cls_out, y_mask, cls_targets)
            pre_neg_acum += precision(cls_out, y_mask, cls_targets, cls = 0)
            rec_neg_acum += recall(cls_out, y_mask, cls_targets, cls = 0)
            count += 1
        inputs_outputs.append(current_model)
        # print(f'Cls Loss: {cls_loss_acum/count:.4f}, Reg Loss: {reg_loss_acum/count:.4f}')
        print(f'Outliers count: {outliers_count_acum}')
        print('accuracy: ',acc_acum/count)
        print('precision_pos: ',pre_pos_acum/count)
        print('recall_pos: ',rec_pos_acum/count)
        print('precision_neg: ',pre_neg_acum/count)
        print('recall_neg: ',rec_neg_acum/count)
        print(f'Regression error (pixels): {coord_error_acum/count:.4f}, Duration error ({duration_error_acum/count:.4f})')
        print('--------------------------------')
    if i < len(names) - 1:
        print(f'Model {names[i + 1]}')
    del model
    torch.cuda.empty_cache()
    gc.collect()
torch.save(inputs_outputs, f'inputs_outputs.pth')