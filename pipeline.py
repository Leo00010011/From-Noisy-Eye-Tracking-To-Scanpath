

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.datasets import FreeViewInMemory, seq2seq_jagged_collate_fn, seq2seq_padded_collate_fn
from src.model import PathModel
import numpy as np


def compute_loss(reg_out,cls_out, y, padding_mask, fixation_len, cls_weigth = .5):
    criterion_reg = torch.nn.MSELoss()
    criterion_cls = torch.nn.BCEWithLogitsLoss()
    y_mask = torch.logical_not(padding_mask)
    cls_targets = torch.zeros(cls_out.size(), dtype = torch.float32)
    batch_idx = torch.arange(cls_targets.size()[0])
    cls_targets[batch_idx,fixation_len] = True
    cls_loss = criterion_cls(cls_out[y_mask], cls_targets[y_mask])
    for i in range(cls_targets.size()[0]):
        print(cls_targets[i, y_mask[i]].long()[-1])




datasetv2 = FreeViewInMemory(sample_size= 13,log = True, start_index=2)

model = PathModel(input_dim = 3,
                  output_dim = 3,
                  n_encoder = 2,
                  n_decoder = 2,
                  model_dim = 256,
                  total_dim = 256,
                  n_heads = 4,
                  ff_dim = 512,
                  max_pos_enc=15,
                  max_pos_dec=26)



dataloader = DataLoader(datasetv2, batch_size=128, shuffle=True, num_workers=0, collate_fn= seq2seq_padded_collate_fn)
for batch in tqdm(dataloader):
    x,x_mask,y, y_mask, fixation_len = batch
    break
    

reg_out,cls_out = model(x,y, x_mask, y_mask)

reg_out.shape

print(reg_out.shape)
print(cls_out.shape)



# compute_loss(reg_out,cls_out, y, y_mask, fixation_len)

# fixation_len


