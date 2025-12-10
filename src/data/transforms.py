import torch
from src.data.datasets import extract_random_period, PAD_TOKEN_ID
from src.preprocess.noise import add_random_center_correlated_radial_noise, discretization_noise
import numpy as np

class ExtractRandomPeriod:
    def __init__(self, start_index, period_duration, sampling_rate, downsample_period, random_offset):
        self.start_index = start_index
        self.period_duration = period_duration
        self.sampling_rate = sampling_rate
        self.downsample_period = downsample_period
        self.random_offset = random_offset
        self.modify_y = False

    def __call__(self,input):
        x, y, _, _ = extract_random_period(
            self.start_index, 
            self.period_duration, 
            input['x'], 
            input['y'], 
            input['fixation_mask'], 
            self.sampling_rate, 
            self.downsample_period,
            self.random_offset
        )
        input['x'] = x
        input['y'] = y
        return input
        
    def __repr__(self):
        return f'ExtractRandomPeriod'

    def __str__(self):
        return f'''+ ExtractRandomPeriod
        start_index={self.start_index}, 
        period_duration={self.period_duration}, 
        sampling_rate={self.sampling_rate}, 
        downsample_period={self.downsample_period}'''
        
class Normalize:
    def __init__(self,key, mode, max_value):
        self.key = key
        self.max_value = max_value
        self.mode = mode
        self.modify_y = key == 'y'

    def __call__(self,input):
        # shape (F,L)
        x = input[self.key]
        if isinstance(x, torch.Tensor):
            if isinstance(self.max_value, torch.Tensor):
                self.max_value = self.max_value.to(x.device)
            else:
                self.max_value = torch.tensor(self.max_value).to(x.device)
        else:
            if isinstance(self.max_value, torch.Tensor):
                self.max_value = self.max_value.cpu().numpy()
        if self.mode == 'coords':
            if isinstance(self.max_value, torch.Tensor):
                x[:2] = x[:2] / self.max_value.unsqueeze(-1)
            else:
                x[:2] = x[:2] / self.max_value[..., np.newaxis]
            
        elif self.mode == 'time':
            x[2] = x[2] / self.max_value
        input[self.key] = x
        return input
    
    def inverse(self, y, tgt_mask):
        # shape (B,L,F)
        if isinstance(y, torch.Tensor):
            if isinstance(self.max_value, torch.Tensor):
                self.max_value = self.max_value.to(y.device)
            else:
                self.max_value = torch.tensor(self.max_value).to(y.device)
        if self.mode == 'coords':
            y[:,:,:2] = y[:,:,:2] * self.max_value
            y.masked_fill(~tgt_mask, PAD_TOKEN_ID)
        elif self.mode == 'time':
            y[:,:,2] = y[:,:,2] * self.max_value
            y.masked_fill(~tgt_mask, PAD_TOKEN_ID)
        return y

    def __repr__(self):
        return f'Normalize'
    
    def __str__(self):
        return f'''+ Normalize
        key={self.key}, 
        max_value={self.max_value}'''
        
class LogNormalizeDuration:
    def __init__(self, mean, std, scale, use_tan):
        self.mean = mean
        self.std = std
        self.scale = scale
        self.use_tan = use_tan
        self.modify_y = True
        
        
    def __call__(self,input):
        # shape (F,L)
        d = input['y'][2]
        # atan normalization
        if self.use_tan:
            d = (np.log1p(d) - self.mean) / self.std
            d = (1 / np.pi) * np.arctan(d) + 0.5
        else:
            d = (np.log1p(d) - self.mean) / self.std
            d = d * self.scale
        input['y'][2] = d
        return input
    
    def inverse(self, y, tgt_mask):
        # shape (B,L,F)
        d = y[:,:,2]
        if self.use_tan == True:
            d = torch.tan(torch.pi*(d - 0.5))
            d = torch.exp((d*self.std) + self.mean) - 1
        else:
            d = torch.exp((d/self.scale*self.std) + self.mean) - 1
        y[:,:,2] = d
        y.masked_fill(~tgt_mask, PAD_TOKEN_ID)
        return y
    
    def __repr__(self):
        return f'LogNormalizeDuration'
    
    def __str__(self):
        return f'''+ LogNormalizeDuration
        mean={self.mean}, 
        std={self.std}, 
        scale={self.scale}'''
        
class StandarizeTime:
    def __init__(self) -> None:
        self.modify_y = False
        
    def __call__(self,input):
        x = input['x']
        x[2] = x[2] - x[2,0]
        input['x'] = x
        return input

    def __repr__(self):
        return f'StandarizeTime'
    
    def __str__(self):
        return f'''+ StandarizeTime'''

# >>>>  NOISE
    
    
class AddRandomCenterCorrelatedRadialNoise:
    def __init__(self, initial_center, ptoa,
                 radial_corr, 
                 radial_avg_norm,
                 radial_std ,
                 center_noise_std, 
                 center_corr,
                 center_delta_norm,
                 center_delta_r ):
        self.initial_center = initial_center
        self.ptoa = ptoa
        self.radial_corr = radial_corr
        self.radial_avg_norm = radial_avg_norm
        self.radial_std = radial_std
        self.center_noise_std = center_noise_std
        self.center_corr = center_corr
        self.center_delta_norm = center_delta_norm
        self.center_delta_r = center_delta_r
        self.modify_y = False

    def __call__(self, input):

        x, _ = add_random_center_correlated_radial_noise(
            input['x'], 
            self.initial_center, 
            self.ptoa, 
            self.radial_corr,
            self.radial_avg_norm, 
            self.radial_std, 
            self.center_noise_std, 
            self.center_corr, 
            self.center_delta_norm, 
            self.center_delta_r
        )
        input['x'] = x
        return input
    
    def __repr__(self):
        return f'AddRandomCenterCorrelatedRadialNoise'
    def __str__(self):
        return f'''+ AddRandomCenterCorrelatedRadialNoise
        initial_center={self.initial_center}, 
        ptoa={self.ptoa}, 
        radial_corr={self.radial_corr}, 
        radial_avg_norm={self.radial_avg_norm}, 
        radial_std={self.radial_std}, 
        center_noise_std={self.center_noise_std}, 
        center_corr={self.center_corr}, 
        center_delta_norm={self.center_delta_norm}, 
        center_delta_r={self.center_delta_r}'''

class DiscretizationNoise:
    def __init__(self, image_shape):
        self.image_shape = image_shape
        self.modify_y = False

    def __call__(self, input):
        x = discretization_noise(self.image_shape, input['x'])
        input['x'] = x
        return input
    
    def __repr__(self):
        return f'DiscretizationNoise'
    def __str__(self):
        return f'''+ DiscretizationNoise
        image_shape={self.image_shape}'''
        
    
