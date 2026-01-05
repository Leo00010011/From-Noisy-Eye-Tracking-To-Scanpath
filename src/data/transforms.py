import torch
import joblib
from src.data.datasets import extract_random_period, PAD_TOKEN_ID
from src.preprocess.noise import add_random_center_correlated_radial_noise, discretization_noise
import numpy as np

class ExtractRandomPeriod:
    def __init__(self, start_index, period_duration, sampling_rate, downsample_period, random_offset, key = 'x'):
        self.start_index = start_index
        self.period_duration = period_duration
        self.sampling_rate = sampling_rate
        self.downsample_period = downsample_period
        self.random_offset = random_offset
        self.modify_y = key == 'y'
        self.key = key

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
    
    def inverse(self, y, tgt_mask, key):
        if key != self.key:
            return y
        # shape (B,L,F)
        if isinstance(y, torch.Tensor):
            if isinstance(self.max_value, torch.Tensor):
                self.max_value = self.max_value.to(y.device)
            else:
                self.max_value = torch.tensor(self.max_value).to(y.device)
        if self.mode == 'coords':
            y[:,:,:2] = y[:,:,:2] * self.max_value
        elif self.mode == 'time':
            y[:,:,2] = y[:,:,2] * self.max_value
        if tgt_mask is not None:
            y.masked_fill(~tgt_mask, PAD_TOKEN_ID)
        return y

    def __repr__(self):
        return f'Normalize'
    
    def __str__(self):
        return f'''+ Normalize
        key={self.key}, 
        max_value={self.max_value}'''
        
class LogNormalizeDuration:
    def __init__(self, mean, std, scale, use_tan, key = 'y'):
        self.mean = mean
        self.std = std
        self.scale = scale
        self.key = key
        self.use_tan = use_tan
        self.modify_y = key == 'y'
        
        
    def __call__(self,input):
        # shape (F,L)
        d = input[self.key][2]
        # atan normalization
        if self.use_tan:
            d = (np.log1p(d) - self.mean) / self.std
            d = (1 / np.pi) * np.arctan(d) + 0.5
        else:
            d = (np.log1p(d) - self.mean) / self.std
            d = d * self.scale
        input[self.key][2] = d
        return input
    
    def inverse(self, y, tgt_mask, key):
        # shape (B,L,F)
        if key != self.key:
            return y
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
        
class QuantileNormalizeDuration:
    def __init__(self, key = 'y', pkl_path = 'quantile_transformer.pkl'):
        self.key = key
        self.modify_y = key == 'y'
        self.quantile_transformer = joblib.load(pkl_path)
        
    def __call__(self,input):
        # shape (F,L)
        x = input[self.key][2]
        x = x.reshape(-1, 1)
        x = self.quantile_transformer.transform(x)
        x = x.reshape(-1)
        input[self.key][2] = x
        return input
    
    def inverse(self, y, tgt_mask, key):
        if key != self.key:
            return y
            
        # y is in range (0, 1)
        d_pred = y[:, :, 2]
        d_norm_clamped = torch.clamp(d_pred, 0.0, 1.0)
        # Reshape for sklearn: (Batch * Len, 1)
        b, l = d_pred.shape
        d_flat = d_norm_clamped.detach().cpu().numpy().reshape(-1, 1)
        
        # Inverse Transform: Maps (0,1) -> Raw Durations
        # No exp() needed, it recovers the raw scale directly!
        d_raw = self.quantile_transformer.inverse_transform(d_flat)
        
        # Reshape back and fill
        d_raw = torch.from_numpy(d_raw).view(b, l).to(d_pred.device)
        
        y[:, :, 2] = d_raw
        y.masked_fill(~tgt_mask, PAD_TOKEN_ID)
        return y
    
    def __repr__(self):
        return f'QuantileNormalizeDuration'
    def __str__(self):
        return f'''+ QuantileNormalizeDuration'''
    
        
class StandarizeTime:
    def __init__(self, key = 'x') -> None:
        self.key = key
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
class SaveCleanX:
    def __init__(self, key = 'clean_x'):
        self.key = key
        self.modify_y = False
    
    def __call__(self, input):
        input['clean_x'] = input['x'].copy()
        return input
    
    def __repr__(self):
        return f'SaveCleanX'
    def __str__(self):
        return f'''+ SaveCleanX'''
    
class AddRandomCenterCorrelatedRadialNoise:
    def __init__(self, initial_center, ptoa,
                 radial_corr, 
                 radial_avg_norm,
                 radial_std ,
                 center_noise_std, 
                 center_corr,
                 center_delta_norm,
                 center_delta_r,
                 return_center_path=False,
                 key = 'x'):
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
        self.return_center_path = return_center_path
        self.key = key

    def __call__(self, input):
        x, center_path = add_random_center_correlated_radial_noise(
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
        
        if self.return_center_path:
            input['center_path'] = center_path
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
    def __init__(self, image_shape, key = 'x'):
        self.key = key
        self.image_shape = image_shape
        self.modify_y = False

    def __call__(self, input):
        x = discretization_noise(self.image_shape, input[self.key])
        input[self.key] = x
        return input
    
    def __repr__(self):
        return f'DiscretizationNoise'
    def __str__(self):
        return f'''+ DiscretizationNoise
        image_shape={self.image_shape}'''
        


class AddGaussianNoiseToFixations:
    def __init__(self, std):
        self.key = 'in_tgt'
        self.std = std
        
    def __call__(self,input):
        # shape (F,L)
        if self.std == 0:
            return input
        y_clone = input['y'].copy()
        noise = np.random.normal(0, self.std, (2, y_clone.shape[1]))
        y_clone[:2] += noise
        input['in_tgt'] = y_clone
        return input
    
    def inverse(self, y, tgt_mask, key):
        return y
    



class AddHeatmaps:
    def __init__(self, sigma=5.0, image_size=(32, 32), device = 'cpu', dtype = torch.float32):
        self.sigma = sigma
        self.key = 'y'
        self.modify_y = True
        self.image_size = image_size
        self.device = device
        self.dtype = dtype
        factory_kwargs = {'device': device, 'dtype': dtype}
        height, width = image_size
        self.pos_x = torch.arange(width, **factory_kwargs)
        self.pos_y = torch.arange(height, **factory_kwargs)
        xx, yy = torch.meshgrid(self.pos_y, self.pos_x, indexing='ij')
        self.xx = self.xx.unsqueeze(0).to(device, dtype)
        self.yy = self.yy.unsqueeze(0).to(device, dtype)
        
    def generate_gaussian_heatmaps(self, coords):
        num_points = coords.shape[0]
        target_x = coords[:, 0].view(num_points, 1, 1)*self.image_size[0]
        target_y = coords[:, 1].view(num_points, 1, 1)*self.image_size[1]
        dist_sq = (self.xx - target_x)**2 + (self.yy - target_y)**2
        heatmaps = torch.exp(-dist_sq / (2 * self.sigma**2))
        return heatmaps
    
    def get_coords_from_heatmaps(self, heatmaps):
        B, N, H, W = heatmaps.shape
        probs = heatmaps.view(B, N, -1)
        probs = probs / (probs.sum(dim=2, keepdim=True) + 1e-6) # +epsilon for stability
        probs = probs.view(B, N, H, W)
        expected_x = torch.sum(probs * self.pos_x.view(1,1, 1, W), dim=(2, 3))/W
        expected_y = torch.sum(probs * self.pos_y.view(1,1, H, 1), dim=(2, 3))/H
        return torch.stack([expected_x, expected_y], dim=-1)
        
    def __call__(self, input):
        y = input['y']
        heatmaps = self.generate_gaussian_heatmaps(y)
        input['heatmaps'] = heatmaps
        return input
    
    def inverse(self, y, tgt_mask, key):
        # shape (B,L,H,W) -> (N,H,W)
        heatmaps = y['heatmaps']
        B,L,_,_ = heatmaps.shape
        heatmaps = heatmaps[tgt_mask]
        coords = get_coords_from_heatmaps(heatmaps)
        # new output
        y_new = torch.fill((B,L,3), PAD_TOKEN_ID)
        y_new[tgt_mask,:2] = coords
        y_new[tgt_mask,2] = y['dur']
        y['reg'] = y_new
        return y
    
    def __repr__(self):
        return f'AddHeatmaps'
    def __str__(self):
        return f'''+ AddHeatmaps
        sigma={self.sigma}'''