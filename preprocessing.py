    
import h5py
from src.preprocess.simulation import gen_gaze, downsample
from src.data.parsers import CocoFreeView
from tqdm import tqdm


def preprocess(sampling_rate, downsample_int,min_scanpath_duration,sample_size,min_fixation_duration, max_fixation_duration, log):
    # TODO Optimization, do not remove noisy scanpath cut wrong parts
    data = CocoFreeView()
    gen_data = []
    original_data_count = len(data)
    filtered_idx = []
    for index in tqdm(range(original_data_count), desc="Processing"):
        gaze, fixations, fixation_mask = gen_gaze(data,
                                                    index, sampling_rate,
                                                    get_scanpath=True,
                                                    get_fixation_mask=True)
        down_gaze = downsample(gaze, down_time_step=downsample_int)
        if (gaze[2,-1] < max(min_scanpath_duration, (sample_size - 1)*downsample_int) or
            fixations[2].max() > max_fixation_duration or
            fixations[2].min() < min_fixation_duration) :
            filtered_idx.append(index)
            continue
        gen_data.append({'down_gaze': down_gaze,
                            'fixations': fixations,
                            'fixation_mask': fixation_mask,
                            'gaze': gaze})
    if log:
        removed = original_data_count - len(gen_data)
        print(f'Removed: {removed} - {(removed/original_data_count)*100}% ')
    # save
    return gen_data, filtered_idx

def save_gen_data(save_path, gen_data, filtered_idx, log):
    with h5py.File(save_path, 'w') as f:
        # Create datasets with shape (43000,) and the vlen dtype
        item = gen_data[0]
        k_dset = dict()
        for k in item.keys():
            k_dset[k] = f.create_dataset(k, len(gen_data), dtype= h5py.special_dtype(vlen= item[k].dtype))

        # Loop and store each item
        for i, item in enumerate(gen_data):
            for k in item.keys():
                if item[k].ndim > 1:
                    k_dset[k][i] = item[k].flatten()
                else:
                    k_dset[k][i] = item[k]
        # Save filtered indices
        f.create_dataset('filtered_idx', data=filtered_idx)
    
    if log:
        print('generated items saved')


if __name__ == "__main__":
    sampling_rate = 60
    downsample_int = 200
    min_scanpath_duration = 3000
    sample_size = 8
    min_fixation_duration = 30
    max_fixation_duration = 1200
    save_path = 'data/Coco FreeView/dataset.hdf5'
    log = True
    gen_data, filtered_idx = preprocess(sampling_rate, downsample_int, min_scanpath_duration, sample_size,min_fixation_duration ,max_fixation_duration , log)

    save_gen_data(save_path, gen_data,filtered_idx, log=log)