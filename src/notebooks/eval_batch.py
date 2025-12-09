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
from src.eval.eval_utils import plt_training_metrics, gather_best_metrics, invert_transforms
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
    




# %% [markdown]
# # Eval

# %% [markdown]
# ## Review Metrics

# %%
# ckpt_path = [os.path.join('outputs','2025-11-19','18-48-14'),
#              os.path.join('outputs','2025-11-27','17-35-19'),
#              os.path.join('outputs','2025-11-28','12-28-42'),
#              os.path.join('outputs','2025-12-03','17-10-55'),]
names = ['best path model ',
'best regression ',
'best but classification ',
'other best duration but classification ',
'best duration good recall ']


ckpt_path = [os.path.join('outputs','2025-11-19','18-48-14'),
os.path.join('outputs','2025-12-06','19-38-17'),
os.path.join('outputs','2025-12-03','12-58-11'),
os.path.join('outputs','2025-11-27','17-30-18'),
os.path.join('outputs','2025-12-09','11-56-23')]

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
    for batch in tqdm(test_dataloader):
        input = move_data_to_device(batch, device)
        with torch.no_grad():
            output = model(**input)
            input, output = invert_transforms(input, output, test_dataloader)
            inputs_outputs.append((ckpt_path,name, input, output))
            cls_loss, reg_loss = compute_loss(input, output)
            print(f'Cls Loss: {cls_loss:.4f}, Reg Loss: {reg_loss:.4f}')
            reg_out, cls_out = output['reg'], output['cls']
            y, y_mask, fixation_len = input['tgt'], input['tgt_mask'], input['fixation_len']
            cls_targets = create_cls_targets(cls_out, fixation_len)
            print('accuracy: ',accuracy(cls_out, y_mask, cls_targets))
            print('precision_pos: ',precision(cls_out, y_mask, cls_targets))
            print('recall_pos: ',recall(cls_out, y_mask, cls_targets))
            print('precision_neg: ',precision(cls_out, y_mask, cls_targets, cls = 0))
            print('recall_neg: ',recall(cls_out, y_mask, cls_targets, cls = 0))
            reg_error, dur_error = eval_reg(reg_out, y, y_mask)
            print(f'Regression error (pixels): {reg_error:.4f}, Duration error ({dur_error:.4f})')
        break
    print('--------------------------------')
    print(f'Model {name}')
    del model
    torch.cuda.empty_cache()
    gc.collect()
torch.save(inputs_outputs, f'inputs_outputs.pth')