import matplotlib.pyplot as plt
from src.data.datasets import FreeViewImgDataset, FreeViewInMemory, CoupledDataloader
from src.data.parsers import CocoFreeView
from torch.utils.data import Subset, DataLoader
from src.data.datasets import seq2seq_padded_collate_fn
from time import time

PathDataset = FreeViewInMemory(sample_size= 13,
                                     log = False, 
                                     start_index=2)
data = CocoFreeView()
data.filter_by_idx(PathDataset.data_store['filtered_idx'])
dataset = FreeViewImgDataset(data)



def main():
    dataloader = CoupledDataloader(PathDataset, dataset, batch_size=128, shuffle=True, num_workers=4)

    count = 100
    s = -1
    for i, (img_batch, et_data_batch) in enumerate(dataloader):
        if s != -1:
            print(f"Time to load and pad batch: {time()-s:.2f} seconds")
            print(f'>>> Batch {i} loaded')
        padded_inputs, input_mask, padded_targets, target_mask, fixation_len  = et_data_batch
        count -= 1
        s = time()
        if count == 0:
            break

if __name__ == "__main__":
    main()