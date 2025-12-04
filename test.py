import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from src.data.datasets import FreeViewImgDataset, FreeViewInMemory, CoupledDataloader
from src.data.parsers import CocoFreeView
from torch.utils.data import Subset, DataLoader
from src.data.datasets import seq2seq_padded_collate_fn
from src.model.model_io import load_models_with_data
from time import time


def test_image_eye_tracking_pairing():
    PathDataset = FreeViewInMemory(sample_size= -1,
                                        log = False, 
                                        start_index=2)
    data = CocoFreeView()
    data.filter_by_idx(PathDataset.data_store['filtered_idx'])
    dataset = FreeViewImgDataset(data)
    del data
    other_coco = CocoFreeView()
    # create filtered index
    mask = np.ones(len(other_coco), dtype = bool)
    mask[PathDataset.data_store['filtered_idx']] = False
    idx = np.arange(len(other_coco))[mask]
    for new_idx, true_idx in tqdm(enumerate(idx)):
        gt_path = other_coco.get_img_path(true_idx)
        my_path = dataset.img_path[new_idx] 
        if gt_path != my_path:
            print('mismatch at index ', new_idx)
            print('gt: ', gt_path)
            print('my: ', my_path)
            return
        x,y,t = other_coco.get_scanpath(true_idx)
        true_scanpath = np.vstack((x,y,t))
    
        _, new_scanpath  = PathDataset[new_idx]
        if not np.array_equal(true_scanpath, new_scanpath):
            print('scanpath mismatch at index ', new_idx)
            return
    print('All indices match!')



def main():
    # test_image_eye_tracking_pairing()
    ckpt_path = ['outputs\\2025-11-19\\18-48-14',
             'outputs\\2025-11-27\\17-35-19',
             'outputs\\2025-11-28\\12-28-42',
             'outputs\\2025-12-03\\17-10-55']

    
    models_and_data = load_models_with_data(ckpt_path)
if __name__ == "__main__":
    main()