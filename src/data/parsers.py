import os
import json
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils import is_number

###
# NOTE ABOUT THE ANGLE TO PIXEL CONVERSION
# ----------------------------------
# In the Github Repo (https://github.com/cvlab-stonybrook/Scanpath_Prediction) 
# they said that the COCO search dataset was made
# in an 1680x1050 display, but they downscaled to 512x320.
# Looking at the publication of HAT which is a second one that
# was made for the same people of COCO. They said :
# "We compute Y by smoothing the ground-truth fixation map with a Gaussian kernel
# with the kernel size being one degree of visual angle."
# In the code (https://github.com/cvlab-stonybrook/HAT/tree/main?tab=readme-ov-file)
# I found that they used the scipy.ndimage.filters.gaussian_filter 
# with sigma equal to 16. This is made over a fixation map that was downscaled as they 
# said previously. Can I now compute the visual angle #
# ------------------------------------
class CocoFreeView:
    def __init__(self, data_path = None):
        #TODO Add Coco Search to the Dataset
        if data_path is None:
            data_path = os.path.join('data', 'Coco FreeView')
        json_path = os.path.join(data_path, 'COCOFreeView_fixations_trainval.json')
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        ta_folder = Path(os.path.join(data_path, 'COCOSearch18-images-TA'))
        tp_folder = Path(os.path.join(data_path, 'COCOSearch18-images-TP'))

        stimulus_paths = list(ta_folder.rglob('*.jpg')) + list(tp_folder.rglob('*.jpg'))

        ntop = {str(path).split(os.path.sep)[-1]: str(path) for path in stimulus_paths }
        for scan_path in data:
            path = ntop[scan_path['name']]
            scan_path['img_path'] = path
            scan_path['class'] = path.split(os.path.sep)[-2]
        self.df = pd.DataFrame.from_dict(data)
        row = self.df.iloc[0]
        
        img = cv2.imread(row['img_path'])
        self.ori_res = img.shape[:2]
        self.dest_res = (320,512)
        # conversion factor from pixel to angles
        self.ptoa = 1/16
    
    def __len__(self):
        return len(self.df)

    def summary(self):
        df = self.df
        print('all stimuli ', len(df['name'].unique()))
        for split in  ['train', 'valid']:
            split_df = df.loc[df['split'] == split, ['name','subject']]
            print(split.upper())
            print('scanpaths: ', len(split_df))
            print('stimuli: ', split_df['name'].nunique())
            count = split_df.groupby('subject')['name'].nunique()
            print('min stimuli per subject: ', count.min())
            print('max stimuli per subject: ', count.max())
            count = split_df.groupby('name')['subject'].nunique()
            print('min subject per stimuli: ', count.min())
            print('max subject per stimuli: ', count.max())    
            
    def get_scanpath(self, idx, downscale = True):
        row = self.df.iloc[idx]
        fx,fy = 1,1
        if downscale:
            fx = self.dest_res[1] / self.ori_res[1]
            fy = self.dest_res[0] / self.ori_res[0]

        if is_number(idx):
            x = np.array(row['X'])*fx
            y = np.array(row['Y'])*fy
            t = np.array(row['T'])
        else:
            x = row['X'].apply(lambda arr: np.array(arr) * fx)
            y = row['Y'].apply(lambda arr: np.array(arr) * fy)
            t = row['T'].apply(np.array)
        return x,y,t   
    
    def get_img_path(self, idx):
        row = self.df.iloc[idx]
        return row['img_path']

    def get_img(self, idx, downscale = True):
        row = self.df.iloc[idx]
        img = cv2.imread(row['img_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if downscale:
            img = cv2.resize(img, dsize = (self.dest_res[1], self.dest_res[0]))
        return img
    
    def filter_by_idx(self, filtered_idx = []):
        df = self.df
        if len(filtered_idx) > 0:
            df = df[~df.index.isin(filtered_idx)]
            df = df.reset_index(drop=True)
        self.df = df
        
    def get_all_subjects(self):
        return self.df['subject'].unique()

    def get_all_stimuli(self):
        return self.df['name'].unique()
    
    def get_disjoint_splits(self, train_subjects, val_subjects, test_subjects,
                            train_stimuli, val_stimuli, test_stimuli):
        df = self.df
        train_df = df[df['subject'].isin(train_subjects) & df['name'].isin(train_stimuli)]
        val_df = df[df['subject'].isin(val_subjects) | df['name'].isin(val_stimuli)]
        test_df = df[(df['subject'].isin(test_subjects) | df['name'].isin(test_stimuli)) & ~df.index.isin(val_df.index)]
        return train_df.index, val_df.index, test_df.index








######################################################
# TODO Check if we can add video information to improve stats
class OurDataset:
    def __init__(self, data_path = None, img_path = None):
        if data_path is None:
            data_path = os.path.join('data', 'Ours', 'result_total_data_prosegur.csv')
        if img_path is None:
            img_path = os.path.join('data', 'Ours', 'prosegur_img.jpg')
        data = pd.read_csv(data_path)
        data = data[data['Module'].str.contains('BI')]
        data_small = data[['User','TimeBlock','XEyeTracking','YEyeTracking']]

        user_list = [group[["XEyeTracking", "YEyeTracking", "TimeBlock"]].to_numpy().T 
                     for _, group in data_small.groupby("User")]
        
        img = cv2.imread(img_path)
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x_size = img.shape[1]/100
        y_size = img.shape[0]/100
        self.user_arrays = []
        for gaze in user_list:
            gaze[0,:] = (gaze[0,:] + 0.5)*x_size
            gaze[1,:] = (gaze[1,:] + 0.5)*y_size
            self.user_arrays.append(gaze)
        


    def summary(self):
        print('sample_num: ',len(self.user_arrays))
        print('subject_num:', len(self.user_arrays))
        print('samples_per_subject:', 1)

    def __len__(self):
        return len(self.user_arrays)

    def get_eye_track(self,idx):
        return [self.user_arrays[i] for i in idx]