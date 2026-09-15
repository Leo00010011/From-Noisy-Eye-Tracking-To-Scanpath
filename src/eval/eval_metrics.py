import torch
import matplotlib.pyplot as plt


def create_cls_targets(cls_out, fixation_len):
    batch_idx = torch.arange(cls_out.size()[0])
    cls_targets = torch.zeros(cls_out.size(), 
                              dtype = torch.float32,
                              device = cls_out.device)
    cls_targets[batch_idx,fixation_len] = 1.0
    return cls_targets

def accuracy(cls_out, attn_mask, cls_targets):
    cls_preds = torch.sigmoid(cls_out) >= 0.5
    attn_mask = attn_mask.unsqueeze(-1)
    correct = (cls_preds == cls_targets) & attn_mask
    accuracy = correct.sum().item() / attn_mask.sum().item()
    return accuracy

def precision(cls_out, attn_mask, cls_targets, cls = 1):
    cls_preds = torch.sigmoid(cls_out) >= 0.5
    attn_mask = attn_mask.unsqueeze(-1)
    true_positives = ((cls_preds == cls) & (cls_targets == cls) & attn_mask).sum().item()
    predicted_positives = ((cls_preds == cls) & attn_mask).sum().item()
    precision = true_positives / predicted_positives if predicted_positives > 0 else 0.0
    return precision

def recall(cls_out, attn_mask, cls_targets, cls = 1):
    cls_preds = torch.sigmoid(cls_out) >= 0.5
    attn_mask = attn_mask.unsqueeze(-1)
    true_positives = ((cls_preds == cls) & (cls_targets == cls) & attn_mask).sum().item()
    actual_positives = ((cls_targets == cls) & attn_mask).sum().item()
    recall = true_positives / actual_positives if actual_positives > 0 else 0.0
    return recall

def eval_reg(reg, y, y_mask):
    y_mask = y_mask.unsqueeze(-1)[:,1:,:]
    count = y_mask.sum().item()
    diff = torch.where(y_mask, reg[:,:-1,:3] - y, torch.tensor(0.0, device=y.device))
    diff_xy = diff[:,:,:2]
    reg_error = torch.sqrt(torch.sum(diff_xy**2, dim=-1))
    dur_error = torch.abs(diff[:,:,2])
    reg_error = reg_error.sum().item() / count
    dur_error = dur_error.sum().item() / count
    return reg_error, dur_error

def eval_denoise(denoise, clean_x):
    diff = denoise - clean_x[:, :, :2]
    denoise_error = torch.sqrt(torch.sum(diff**2, dim=-1))
    denoise_error = denoise_error.sum().item() / denoise_error.numel()
    return denoise_error


def nearest_centroid_offsets(token_centers, image_centroids, centroid_mask):
    """Per-token offset to its nearest **valid** fixation centroid (the alignment target).

    Args:
        token_centers:   ``(1, S, 2)`` or ``(B, S, 2)`` — per-token anchors (reference grids).
        image_centroids: ``(B, C, 2)`` — per-image centroids (already gathered per batch row).
        centroid_mask:   ``(B, C)`` bool — True where the centroid is real (not padding).

    Returns ``(B, S, 2)`` offsets ``nearest_centroid - token_center``, computed under
    ``torch.no_grad`` (image-intrinsic target; no gradient flows through it).
    """
    with torch.no_grad():
        B = image_centroids.size(0)
        c = token_centers.expand(B, -1, -1) if token_centers.size(0) != B else token_centers
        d = torch.cdist(c, image_centroids)                        # (B, S, C)
        d = d.masked_fill(~centroid_mask.unsqueeze(1), float("inf"))
        nn_idx = d.argmin(dim=-1)                                  # (B, S)
        nearest = torch.gather(image_centroids, 1,
                               nn_idx.unsqueeze(-1).expand(-1, -1, 2))
        return nearest - c                                         # (B, S, 2)


def eval_align(align_out, token_centers, image_centroids, centroid_mask, pixel_scale=None):
    """Mean ``‖pred - target‖₂`` over valid tokens/rows (FR16).

    In normalized units by default. When ``pixel_scale`` is given (a length-2 ``[W, H]`` tensor /
    sequence), each axis of the ``(pred - target)`` offset is scaled by its corresponding image
    dimension **before** the norm — the anisotropic conversion to pixels (x·W, y·H). Rows whose
    image has no centroid are dropped; returns ``0.0`` when no row is valid (so ``validate``'s
    ``> 0`` guard never appends spuriously)."""
    row = centroid_mask.any(dim=1)                                 # (B,)
    if not bool(row.any()):
        return 0.0
    target = nearest_centroid_offsets(token_centers, image_centroids, centroid_mask)
    diff = align_out - target                                      # (B, S, 2), normalized
    if pixel_scale is not None:
        if not torch.is_tensor(pixel_scale):
            pixel_scale = torch.as_tensor(pixel_scale)
        diff = diff * pixel_scale.to(device=diff.device, dtype=diff.dtype)
    r = row.view(-1, 1, 1).expand_as(align_out)
    return float(diff[r].view(-1, 2).norm(dim=-1).mean().item())