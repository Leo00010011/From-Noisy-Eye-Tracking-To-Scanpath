import json
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plt_training_metrics(path_list, start_epoch=0):
    metric_list = []
    for idx, path in enumerate(path_list):
        with open(path, 'r') as f:
            metric_list.append(json.load(f))
    
    start_idx = []
    for metrics in metric_list:
        for j, e in enumerate(metrics['epoch']):
            if e >= start_epoch:
                start_idx.append(j)
                break
    
    fig, axis = plt.subplots(2,2,figsize=(20,10))
    for idx, (start_idx, metrics) in enumerate(zip(start_idx,metric_list)):
        for k in metrics.keys():
            if k != 'epoch':
                if k == 'reg_loss_train' or k == 'regression loss':
                    axis[0,0].plot(list(range(len(metrics[k])))[start_epoch:],metrics[k][start_epoch:], label= f"{idx}reg_loss_train")
                elif k == 'reg_loss_val' or k == 'regression_loss':
                    axis[0,0].plot(metrics['epoch'][start_idx:], metrics[k][start_idx:], label=f"{idx}reg_loss_val")
                elif k == 'cls_loss_train' or k == 'classification loss':
                    axis[0,1].plot(list(range(len(metrics[k])))[start_epoch:],metrics[k][start_epoch:], label=f"{idx}cls_loss_train")
                elif k == 'cls_loss_val' or k == 'classification_loss':
                    axis[0,1].plot(metrics['epoch'][start_idx:],metrics[k][start_idx:], label=f"{idx}cls_loss_val")
                elif k == 'reg_error_val' or k == 'duration_error_val':
                    axis[1,0].plot(metrics['epoch'][start_idx:],metrics[k][start_idx:], label=f"{idx}{k}")
                else:
                    axis[1,1].plot(metrics['epoch'][start_idx:], metrics[k][start_idx:], label=f"{idx}{k}")
    # axis[0,0].hlines([8000], xmin=0, xmax=metrics['epoch'][-1], colors='red', linestyles='dashed')
    axis[0,0].legend()
    axis[0,1].legend()
    axis[1,0].legend()
    axis[1,1].legend()
    fig.tight_layout()
    
    
def gather_best_metrics(path_list):
    metric_list = []
    for path in path_list:
        with open(path, 'r') as f:
            metric_list.append(json.load(f))
    
    best_metrics_list = []
    for metrics in metric_list:
        best_metrics = {}
        # obtain the epoch where the best regression loss in the validation set is achieved
        if 'reg_loss_val' in metrics.keys():
            best_reg_loss_val = min(metrics['reg_loss_val'])
            best_reg_loss_val_epoch = metrics['reg_loss_val'].index(best_reg_loss_val)
            
        else:
            best_reg_loss_val = min(metrics['regression_loss'])
            best_reg_loss_val_epoch = metrics['regression_loss'].index(best_reg_loss_val)
        for k in metrics.keys():
            if k == 'reg_loss_train' or k == 'regression loss':
                best_metrics['reg_loss_train'] = metrics[k][metrics['epoch'][best_reg_loss_val_epoch] - 1]
            elif k == 'reg_loss_val' or k == 'regression_loss':
                best_metrics['reg_loss_val'] = metrics[k][best_reg_loss_val_epoch]
            elif k == 'cls_loss_train' or k == 'classification loss':
                best_metrics['cls_loss_train'] = metrics[k][metrics['epoch'][best_reg_loss_val_epoch] - 1]
            elif k == 'cls_loss_val' or k == 'classification_loss':
                best_metrics['cls_loss_val'] = metrics[k][best_reg_loss_val_epoch]
            elif k == 'reg_error_val' or k == 'duration_error_val':
                best_metrics[k] = min(metrics[k])
            else:
                best_metrics[k] = max(metrics[k])
        best_metrics_list.append(best_metrics)
    return pd.DataFrame(best_metrics_list)


def amplitude_array(gaze, ptoa=None):
    if ptoa is None:
        ptoa = 1/16
    coords = gaze[:2, :]
    return np.linalg.norm(np.diff(coords, axis=1), axis=0)*ptoa


def plot_amplitude_dist(gaze_list, label_list, ptoa_list, bin_count=30, bin_min=0, bin_max=20):
    bins = np.linspace(bin_min, bin_max, bin_count)
    for gaze, label, ptoa in zip(gaze_list, label_list, ptoa_list):
        amp = np.concat([amplitude_array(gaze_sample, ptoa)
                        for gaze_sample in gaze])
        counts, bin_edges = np.histogram(amp, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        norm_counts = counts/counts.sum()
        plt.plot(bin_centers, norm_counts, label=label)
    plt.vlines([2], 0, 0.1, colors='red', label='scanpath_mode')
    plt.legend()
    plt.title('Amplitude Distribution in Degrees')

def invert_transforms(inputs, outputs, dataloader, remove_outliers = False):
    if 'reg' in outputs:
        pred_reg = outputs['reg']
    else:
        pred_reg = torch.concat([outputs['coord'], outputs['dur']], dim=2)
    gt_reg = inputs['tgt']
    if hasattr(dataloader, 'path_dataset'):
        transforms = dataloader.path_dataset.transforms
    else:
        transforms = dataloader.dataset.dataset.transforms
    # reverse the transforms
    for transform in reversed(transforms):
        if transform.modify_y:
            pred_reg = transform.inverse(pred_reg, inputs['tgt_mask'].unsqueeze(-1))
            gt_reg = transform.inverse(gt_reg, inputs['tgt_mask'][:,1:].unsqueeze(-1))
    if remove_outliers:
        y_mask = inputs['tgt_mask'][:,1:].unsqueeze(-1)
        outliers = (pred_reg[:,:-1,2:] > 1200) & y_mask
        pred_reg[:,:-1,2:][outliers] = 1200
        outputs['outliers_count'] = outliers.sum().item()
    outputs['reg'] = pred_reg
    inputs['tgt'] = gt_reg
    return inputs, outputs

def compute_angles(gaze):
    coords = gaze[:2, :]
    gaze_vec = np.diff(coords, axis=1)
    return np.arctan2(gaze_vec[1], gaze_vec[0]) * 180 / np.pi


def plot_angle_distribution(gaze_list, label_list, bin_counts=30):

    for gaze, label in zip(gaze_list, label_list):
        ang = np.concatenate([compute_angles(gaze_sample)
                             for gaze_sample in gaze])
        counts, edges = np.histogram(ang, bins=bin_counts, range=(-180, 180))
        norm_counts = counts / counts.sum()
        bin_centers = (edges[:-1] + edges[1:]) / 2
        plt.plot(bin_centers, norm_counts, label=label)

    plt.xticks(
        [-180, -90, 0, 90, 180],
        ["Left", "Down", "Right", "Up", "Left"]
    )

    plt.xlabel("Orientation")
    plt.ylabel("Normalized frequency")
    plt.title("Gaze orientation histogram")
    plt.legend()


def consecutive_angles(gaze):
    coords = gaze[:2, :].T
    vec = np.diff(coords, axis=0)
    v1 = vec[:-1]
    v2 = vec[1:]
    dot_product = np.sum(v1*v2, axis=1)
    cross_product = v1[:, 0]*v2[:, 1] * v2[:, 0]*v1[:, 1]
    angles = np.arctan2(dot_product, cross_product)
    angles = angles/np.pi*180
    return angles


def plot_consecutive_angles(gaze_list, label_list, bin_counts=30):

    for gaze, label in zip(gaze_list, label_list):
        ang = np.concat([consecutive_angles(gaze_sample)
                        for gaze_sample in gaze])
        counts, edges = np.histogram(ang, bins=bin_counts, range=(-180, 180))
        norm_counts = counts / counts.sum()
        bin_centers = (edges[:-1] + edges[1:]) / 2
        plt.plot(bin_centers, norm_counts, label=label)

    plt.xticks([-180, -90, 0, 90, 180])

    plt.xlabel("Orientation")
    plt.ylabel("Normalized frequency")
    plt.title("Angle Between Samples")
    plt.legend()


def autocorrelation(gaze, max_len=7):
    coords = gaze[:2, :]
    amplitudes = np.linalg.norm(coords, axis=0)
    signal = amplitudes - np.mean(amplitudes)
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    autocorr = autocorr / autocorr[0]
    return autocorr[:max_len]


def plot_autocorrelation(gaze_list, label_list, min_len=7):
    for gaze, label in zip(gaze_list, label_list):
        autocorr = np.vstack([autocorrelation(gaze_sample, min_len)
                             for gaze_sample in gaze if gaze_sample.shape[1] > min_len])
        mean_autocorr = autocorr.mean(axis=0)
        std_autocorr = autocorr.std(axis=0)
        line_plot = plt.plot(mean_autocorr, label=label)
        line_color = line_plot[0].get_color()
        plt.fill_between(
            range(len(mean_autocorr)),
            mean_autocorr - std_autocorr,
            mean_autocorr + std_autocorr,
            color=line_color,
            alpha=0.2,
            label='Standard Deviation'
        )
    plt.title('Autocorrelation of Amplitudes')
    plt.xlabel('Lag (k)')
    plt.ylabel('Normalized Correlation')
    plt.grid(True)
    plt.ylim(-1.1, 1.1)
    plt.legend()
    plt.hlines([0], xmin=0, xmax=min_len - 1, colors='red')


def compute_vector_dist(vec_list, ptoa=None,  hist_min = -10, hist_max = 10, hist_bins = 30):
    if ptoa is None:
        ptoa = 1/16
    vec = np.concatenate(vec_list, axis=1)
    vec = vec * ptoa
    edges = np.linspace(hist_min, hist_max, hist_bins)

    counts, x_edges, y_edges = np.histogram2d(vec[0], vec[1], bins=edges)
    return counts, x_edges, y_edges


def plot_vector_dist(gaze_list, ptoa_list, label_list, hist_min= -20, hist_max=20, hist_bins=50, arrow = False):
    """
    Plots the 2D distribution of vectors for multiple datasets.

    gaze_list, ptoa_list, and label_list must be the same length.
    """
    fig, axis = plt.subplots(1, len(gaze_list), figsize=(16, 8), squeeze=False)
    axis = axis.flatten() # Ensure axis is always an iterable array

    for i, (gaze, ptoa, label) in enumerate(zip(gaze_list, ptoa_list, label_list)):
        counts, x_edges, y_edges = compute_vector_dist(gaze, ptoa, hist_min, hist_max, hist_bins)
        
        # Avoid division by zero if counts is all zero
        if counts.sum() > 0:
            counts = counts / counts.sum()

        blurred_counts = counts
        im = axis[i].imshow(
            blurred_counts.T,  # Transpose for conventional X/Y axis alignment
            cmap='gray_r',
            origin='lower',  # Put the origin (0,0) at the bottom-left
            extent=[x_edges.min(), x_edges.max(), y_edges.min(), y_edges.max()]
        )
        
        fig.colorbar(im, ax=axis[i], label='Probability Density') 
        
        if arrow:
            # CORRECTED LINE: dx=1, dy=0 to go from (-1,0) to (0,0)
            axis[i].arrow(-1, 0, 1, 0, head_width=0.1, head_length=0.1, fc='red', ec='red', length_includes_head=True)

        axis[i].set_title(f'{label}')
        axis[i].set_xlabel('Scaled X Difference')
        axis[i].set_ylabel('Scaled Y Difference')
    plt.suptitle('2D Histogram of Vectors')
    plt.tight_layout() # Adjust subplots to fit in figure area
    plt.show()

def calculate_relative_vectors(vec):
    """
    Transforms movement vectors to be relative to the preceding vector.

    For each pair of consecutive movement vectors (v1, v2), it transforms v2
    into a new coordinate system where v1 is the vector (1, 0).

    Args:
        gaze_trajectory (np.ndarray): An array of shape (2, N) with x and y coordinates.

    Returns:
        np.ndarray: An array of shape (2, M) containing the transformed vectors.
    """

    v1 = vec[:, :-1]
    v2 = vec[:, 1:]

    norm_v1 = np.linalg.norm(v1, axis=0)

    valid_indices = norm_v1 > 1e-9
    v1 = v1[:, valid_indices]
    v2 = v2[:, valid_indices]
    norm_v1 = norm_v1[valid_indices]

    angle_v1 = np.arctan2(v1[1, :], v1[0, :])

    cos_angle = np.cos(-angle_v1)
    sin_angle = np.sin(-angle_v1)

    new_x = (v2[0, :] * cos_angle - v2[1, :] * sin_angle) / norm_v1
    new_y = (v2[0, :] * sin_angle + v2[1, :] * cos_angle) / norm_v1

    return np.vstack((new_x, new_y))

def concat_reg(output):
    if 'reg' in output:
        return output['reg']
    else:
        return torch.concat([output['coord'], output['dur']], dim=2)

def eval_autoregressive(model, inputs, out_len, only_last = False):
    model.eval()
    output = None
    inputs['tgt_mask'] = None
    with torch.no_grad():
        for _ in range(out_len):
            inputs['tgt'] = output            
            current_output = model(**inputs)
            
            reg = concat_reg(current_output)
            if only_last:
                if output is None:
                    output = reg
                else:
                    output = torch.concat([only_last, reg[:,-1:,:]], dim=1)
            else:
                output = reg
    current_output['reg'] = output
    return current_output