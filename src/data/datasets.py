from src.data.parsers import CocoFreeView
from src.preprocess.noise import add_random_center_correlated_radial_noise, discretization_noise
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import torch
import h5py
import os
import math
import numpy as np
from PIL import Image
from tqdm import tqdm

PAD_TOKEN_ID = 0.5

def extract_random_period(start_index, period_duration, noisy_samples, fixations, fixation_mask, sampling_rate, downsample_period, random_offset = True):
    size = math.ceil(period_duration/downsample_period)
    if random_offset:
        down_offset = np.random.randint(start_index, noisy_samples.shape[1] - size + 1, 1, dtype = int)[0]
    else:
        down_offset = start_index
    # get the values in the original sampling rate
    ori_period = 1000/sampling_rate
    conversion_factor = downsample_period/ori_period
    ori_idx = math.floor(down_offset*conversion_factor)
    ori_size = math.ceil((size - 1)*conversion_factor)
    last_idx = ori_idx + ori_size
    # TEST
    # get the fisrt fixation and if it is not completely included get the next one
    if fixation_mask[ori_idx] > 0:
        if (ori_idx - 1) >= 0 and fixation_mask[ori_idx - 1] == fixation_mask[ori_idx]:
            start_fixation = fixation_mask[ori_idx] + 1
        else:
            start_fixation = fixation_mask[ori_idx]
    else:
        # if the first value is a saccade look for the first fixation
        current_idx = ori_idx + 1
        while current_idx < (ori_idx + ori_size) and fixation_mask[current_idx] == 0:
            current_idx += 1
        if current_idx == (ori_idx + ori_size):
            # if there is not a fixation return an empty array
            return noisy_samples[:, down_offset:down_offset + size],np.array([]), -1,-1
        else:
            start_fixation = fixation_mask[current_idx]
    # search the last fixation
    if fixation_mask[last_idx] > 0:
        if (last_idx + 1) < fixation_mask.shape[0] and fixation_mask[last_idx + 1] == fixation_mask[last_idx]:
            end_fixation = fixation_mask[last_idx] - 1
        else:
            end_fixation = fixation_mask[last_idx]
    else:
        current_idx = last_idx - 1
        while current_idx > ori_idx and fixation_mask[current_idx] == 0:
            current_idx -= 1
        end_fixation = fixation_mask[current_idx]
    # the mask are saved shifted in order to assign 0 to the saccade samples
    start_fixation -= 1
    end_fixation -= 1
    x = noisy_samples[:, down_offset:down_offset + size]
    y = fixations[:, start_fixation: end_fixation + 1]
    return x, y, start_fixation, end_fixation

    

class FreeViewBatch(Dataset):
    '''
    The noisy and downsampled simulated eye-tracking and the section of the scanpath that fits entirely in that part
    '''
    def __init__(self,
                 data_path = None,
                 sample_duration=-1,
                 sampling_rate=60,
                 downsample_int=200,
                 batch_size=128,
                 min_scanpath_duration = 3000,
                 max_fixation_duration = 1200,
                 log = False,
                 debug = False):
        super().__init__()
        if data_path is None:
            data_path = os.path.join('data', 'Coco FreeView')
        self.sampling_rate = sampling_rate
        self.sample_duration = sample_duration # 90% larger than 20 at downsample 200
        self.downsample = downsample_int
        self.min_scanpath_duration = min_scanpath_duration
        self.max_fixation_duration = max_fixation_duration
        self.log = log
        self.batch_size = batch_size
        self.debug = debug
        self.ori_path = os.path.join(data_path, 'dataset.hdf5')
        self.ori_data = None        
        if not os.path.exists(self.ori_path):
            print('Execute preprocess')
        self.shuffled_path = self.ori_path.replace('.hdf5', '_shuffled.hdf5')
        self.shuffled_data = None
        self.json_dataset = None
        


    def __len__(self):
        with h5py.File(self.ori_path,'r') as ori_data:
            return math.ceil(ori_data['down_gaze'].shape[0]/self.batch_size)
    
    def sample_count(self):
        with h5py.File(self.ori_path,'r') as ori_data:
            return ori_data['down_gaze'].shape[0]


    def get_single_item(self, index):
        # reading is inefficient because is reading from memory one by one
        if self.ori_data is None:
            self.ori_data = h5py.File(self.ori_path,'r')
        x = self.ori_data['down_gaze'][index].reshape((3,-1))
        y = self.ori_data['fixations'][index].reshape((3,-1))
        if self.sample_duration != -1:
            fixation_mask = self.ori_data['fixation_mask'][index]
            x, y, start_fixation, end_fixation = extract_random_period(self.sample_duration,
                                                x,
                                                y,
                                                fixation_mask,
                                                self.sampling_rate,
                                                self.downsample)
            # if start_fixation != -1:
            #     gaze = self.ori_data['gaze'][index].reshape((3,-1))
            #     test_segment_is_inside(x,start_fixation, end_fixation,gaze, fixation_mask)

        x, _ = add_random_center_correlated_radial_noise(x, [320//2, 512//2], 1/16,
                                                                  radial_corr=.2,
                                                                  radial_avg_norm=4.13,
                                                                  radial_std=3.5,
                                                                  center_noise_std=100,
                                                                  center_corr=.3,
                                                                  center_delta_norm=300,
                                                                  center_delta_r=.3)
        return x, y
    
    def __getitem__(self, index):
        # reading is 3x faster with batch size 128 
        # but can´t use the workers of the torch.dataloader (epoch in 4.9)
        if self.shuffled_data is None:
            self.shuffled_data = h5py.File(self.shuffled_path,'r')
        batch_size = self.batch_size
        down_gaze = self.shuffled_data['down_gaze'][index*batch_size:(index + 1)*batch_size]
        fixations = self.shuffled_data['fixations'][index*batch_size:(index + 1)*batch_size]
        vals = None
        if self.sample_duration != -1:
            fixation_mask = self.shuffled_data['fixation_mask'][index*batch_size:(index + 1)*batch_size]
            # gaze = self.data['gaze'][index]
            # vals = (down_gaze,fixations,fixation_mask, gaze)
            vals = (down_gaze,fixations,fixation_mask)
        else:
            vals = (down_gaze,fixations)
        x_batch = []
        y_batch = []
        for value in zip(*vals):
            x = value[0].reshape((3,-1))        
            y = value[1].reshape((3,-1))
            if self.sample_duration != -1:
                fixation_mask = value[2]
                x, y, start_fixation, end_fixation = extract_random_period(self.sample_duration,
                                                    x,
                                                    y,
                                                    fixation_mask)
                # if start_fixation != -1:
                #     gaze = value[3].reshape((3,-1))
                #     test_segment_is_inside(x,start_fixation, end_fixation,gaze, fixation_mask)

            x_batch.append(x)
            y_batch.append(y)
        x_batch, _ = add_random_center_correlated_radial_noise(x_batch, [320//2, 512//2], 1/16,
                                                                radial_corr=.2,
                                                                radial_avg_norm=4.13,
                                                                radial_std=3.5,
                                                                center_noise_std=100,
                                                                center_corr=.3,
                                                                center_delta_norm=300,
                                                                center_delta_r=.3)
        # self.close_and_remove_data()
        self.shuffled_data.close()
        self.shuffled_data = None
        
        return x_batch, y_batch
    
    def shuffle_dataset(self):
        
        with h5py.File(self.ori_path,'r') as ori_data:
            dataset_names = ['down_gaze', 'fixations', 'fixation_mask', 'gaze']
            if self.log:
                print('reading original data')
            original_data = {name: ori_data[name][:] for name in dataset_names}
        idx = np.arange(original_data['down_gaze'].shape[0])
        np.random.shuffle(idx)
        for name in dataset_names:
            original_data[name] = original_data[name][idx]

        with h5py.File(self.shuffled_path, 'w') as f_out:
            for name, data in original_data.items():
                f_out.create_dataset(
                    name,
                    data=data,
                )
        if self.log:
            print('shuffled data saved')

    def close_and_remove_data(self):
        if self.ori_data is not None:
            # if self.log:
            #     print('closing original data file')
            self.ori_data.close()
            self.ori_data = None
        if self.shuffled_data is not None:
            # if self.log:
            #     print('closing shuffled data file')
            self.shuffled_data.close()
            self.shuffled_data = None




class FreeViewInMemory(Dataset):
    def __init__(self,
                 data_path = None,
                 transforms = [],
                 log = False):
        self.data_path = data_path
        self.log = log
        self.transforms = transforms
        self.data_store = {}
        
        if data_path is None:
            data_path = os.path.join('data','Coco FreeView')
        self.data_path = data_path
        file_path = os.path.join(data_path, 'dataset.hdf5')

        with h5py.File(file_path, 'r') as f:
            for key in f.keys():
                self.data_store[key] = f[key][:] # [:] reads all data
        if self.log:
            print('Data loaded in memory')
            if not transforms:
                print('No transforms provided')
            else:
                print('Transforms:')
                for transform in transforms:
                    print(transform)
        self.length = self.data_store['down_gaze'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """
        Fetches a single sample from RAM, applies sampling and noise.
        """
        # Get the pre-loaded data for this index
        x = self.data_store['down_gaze'][index].reshape((3, -1)).copy()
        y = self.data_store['fixations'][index].reshape((3, -1)).copy()
        fixation_mask = self.data_store['fixation_mask'][index]
        input = {'x': x, 'y': y, 'fixation_mask': fixation_mask}
        for transform in self.transforms:
            input = transform(input)
        if 'clean_x' in input:
            return input['x'], input['y'], input['clean_x']
        return input['x'], input['y']
    

def build_attn_mask(input_lengths, allow_start = False, return_none = True):
    max_length = int(max(input_lengths))
    batch_size = len(input_lengths)
    if all([length == max_length for length in input_lengths]) and return_none:
        return None
    if allow_start:
        max_length += 1
        input_lengths = input_lengths + 1
    attn_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    for i, length in enumerate(input_lengths):
        attn_mask[i, :length] = True
    return attn_mask

def seq2seq_padded_collate_fn(batch):
    fixation_len = torch.asarray([item[1].shape[1] for item in batch], dtype=int)
    input_mask = build_attn_mask([item[0].shape[1] for item in batch])
    target_mask = build_attn_mask(fixation_len, allow_start = True, return_none = False)

    input_sequences = [torch.from_numpy(item[0].T).float() for item in batch]
    target_sequences = [torch.from_numpy(item[1].T).float() for item in batch]
    output = {}
    if len(batch[0]) == 3:
        clean_x_sequences = [torch.from_numpy(item[2].T).float() for item in batch]
        padded_clean_x = torch.nn.utils.rnn.pad_sequence(
            clean_x_sequences, 
            batch_first=True, 
            padding_value=PAD_TOKEN_ID
        )
        output['clean_x'] = padded_clean_x
    

    
    padded_inputs = torch.nn.utils.rnn.pad_sequence(
        input_sequences, 
        batch_first=True, 
        padding_value=PAD_TOKEN_ID
    )
    
    padded_targets = torch.nn.utils.rnn.pad_sequence(
        target_sequences, 
        batch_first=True, 
        padding_value=PAD_TOKEN_ID
    )
    output['src'] = padded_inputs
    output['src_mask'] = input_mask
    output['tgt'] = padded_targets
    output['tgt_mask'] = target_mask
    output['fixation_len'] = fixation_len
    return output


def seq2seq_jagged_collate_fn(batch):
    input_sequences = [torch.from_numpy(item[0].T).float() for item in batch]
    target_sequences = [torch.from_numpy(item[1].T).float() for item in batch]
    
    inputs = torch.nested.nested_tensor(input_sequences,layout=torch.jagged)
    targets = torch.nested.nested_tensor(target_sequences,layout=torch.jagged)
    
    return inputs, targets

def search(mask, fixation, side = 'right'):
    start = 0
    stop = mask.shape[0]
    step = 1
    if side == 'left':
        start = mask.shape[0] - 1 
        stop = -1
        step = -1

    for i in range(start,stop, step):
        if mask[i] == fixation:
            return i
    return -1

def test_segment_is_inside(index, x, si,ei,gaze, fixation_mask):
    sidx = search(fixation_mask, si + 1, side = 'right')
    eidx = search(fixation_mask, ei + 1, side = 'left')
    if sidx == -1:
        print(f'{index}❌ Start Fixation not found: si:{si + 1} \n {fixation_mask}')
        return
    if eidx == -1:
        print(f'{index}❌ End Fixation not found: si:{ei + 1} \n {fixation_mask}')
        return
    if x[2,0] <= gaze[2,sidx] and (x[2,-1] + 200) >= gaze[2,eidx]:
        print(f'{index}✅Pass: DS [{x[2,0]},{x[2,-1]}] Ori [{gaze[2,sidx]},{gaze[2,eidx]}]')
        # return 
    else:
        print(f'{index}❌Outside: DS [{x[2,0]},{x[2,-1]}] Ori [{gaze[2,sidx]},{gaze[2,eidx]}]')


def location_test(index, si, ei, gaze, fixation_mask, fixations):
    sidx = search(fixation_mask, si + 1, side = 'right')
    eidx = search(fixation_mask, ei + 1, side = 'left')
    if sidx == -1:
        print(f'{index}❌ Start Fixation not found: si:{si + 1} \n {fixation_mask}')
        return
    if eidx == -1:
        print(f'{index}❌ End Fixation not found: si:{ei + 1} \n {fixation_mask}')
        return
    max_dist = 0
    for i in range(sidx, eidx + 1):
        if fixation_mask[i] == 0:
            continue
        f_index = fixation_mask[i] - 1
        fx = fixations[0,f_index]
        fy = fixations[1,f_index]
        gx = gaze[0,i]
        gy = gaze[1,i]
        dist = math.sqrt((fx - gx)**2 + (fy - gy)**2)
        max_dist = max(max_dist, dist)
        if dist > 100:
            print(f'{index}❌ Fixation point too far from gaze point at index {i}: Fixation({fx},{fy}) Gaze({gx},{gy}) Dist:{dist}')
            return
    # print(f'✅ All fixation points are within acceptable distance from gaze points between indices {sidx} and {eidx}.')
    print(f'✅ Max distance between fixation and gaze points: {max_dist}')

class FreeViewImgDataset(Dataset):
    def __init__(self, data:CocoFreeView, transform = None):
        self.img_path = [data.get_img_path(idx) for idx in range(len(data))]
        self.transform = transform
    
    def __getitem__(self, idx):       
        path = self.img_path[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, idx
    
    def __len__(self):
        return len(self.img_path)


class DeduplicatedMemoryDataset(Dataset):
    def __init__(self, data:CocoFreeView, resize_size=256, transform=None):
        """
        Args:
            data: Source data object with .get_img_path(i)
            resize_size: Target size for caching
        """
        self.data = data
        self.resize_size = resize_size
        self.data_path = data.data_path
        self.ingest_transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((resize_size, resize_size), antialias=True),
            v2.ToDtype(torch.uint8, scale=False)
        ])

        self.runtime_transform = transform
        self.all_image_path = os.path.join(self.data_path, f'all_images_{resize_size}.pth')
        unique_paths, indices = self.build_index()
        self.unique_paths = unique_paths
        self.indices = indices
        if os.path.exists(self.all_image_path):
            print('Image Bank found starting load')
            self.image_bank = torch.load(self.all_image_path)
        else:
            print('Image Bank not found starting build')
            image_bank = self.build_image_bank()
            torch.save(image_bank, self.all_image_path)
            self.image_bank = image_bank

    def build_index(self):
        print("Scanning dataset for duplicate images...")
        data = self.data
        total_len = len(data)
        path_to_id = {}      # Maps file_path -> unique_id
        unique_paths = []    # List of unique paths to load later
        indices = []  
        # We iterate once to build the map
        for i in range(total_len):
            path = data.get_img_path(i)
            
            if path not in path_to_id:
                # Found a new unique image
                unique_id = len(unique_paths)
                path_to_id[path] = unique_id
                unique_paths.append(path)
            
            # Record which unique image this sample points to
            indices.append(path_to_id[path])

        # Store indices as a Tensor (int64) for shared memory efficiency
        self.indices = torch.tensor(indices, dtype=torch.long)
        
        num_unique = len(unique_paths)
        print(f"Found {total_len} samples, but only {num_unique} unique images.")
        return unique_paths, indices

    def build_image_bank(self):
        resize_size = self.resize_size
        unique_paths = self.unique_paths
        num_unique = len(unique_paths)
        
        # --- 3. ALLOCATE & LOAD UNIQUE IMAGES ---
        print(f"Allocating RAM for {num_unique} unique images...")
        image_bank = torch.empty(
            (num_unique, 3, resize_size, resize_size), 
            dtype=torch.uint8
        )

        print("Hydrating unique image bank...")
        for i, path in tqdm(enumerate(unique_paths), total=num_unique):
            img = Image.open(path).convert("RGB")
            image_bank[i] = self.ingest_transform(img)
        return image_bank

    def __getitem__(self, idx):
        unique_idx = self.indices[idx]
        img = self.image_bank[unique_idx]
        img = self.runtime_transform(img)
        return img, idx

    def __len__(self):
        return len(self.indices)


class CoupledDataloader:
    """
    A data loader that allows to get eye-tracking data coupled with images. Useful to be able to use workers despite that freeviewinmemory is not thread-safe.
    """
    def __init__(self,
                 path_dataset: FreeViewInMemory,
                 dataset: FreeViewImgDataset,
                 batch_size: int,
                 shuffle: bool,
                 num_workers: int,
                 prefetch_factor: int = 2,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 drop_last_batch: bool = True):
        self.path_dataset = path_dataset
        self.dataset = dataset
        self.dataloader = DataLoader(dataset,
                                     batch_size=batch_size,
                                     shuffle=shuffle,
                                     num_workers=num_workers,
                                     persistent_workers=persistent_workers,
                                     prefetch_factor=prefetch_factor,
                                     pin_memory=pin_memory,
                                     drop_last=drop_last_batch)

    def __iter__(self):
        for img_batch, idx_batch in self.dataloader:
            et_data_batch = [self.path_dataset[i] for i in idx_batch]
            et_data_batch = seq2seq_padded_collate_fn(et_data_batch)
            et_data_batch['image_src'] = img_batch
            yield et_data_batch

    def __len__(self):
        return len(self.dataloader)
