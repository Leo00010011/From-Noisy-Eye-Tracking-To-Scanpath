import torch
from torch.utils.data import Subset
import os
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Any, Dict
from omegaconf import OmegaConf
from src.model.path_model import PathModel
from src.training.pipeline_builder import PipelineBuilder

def save_checkpoint(
    model: nn.Module,
    filepath: str,
    save_full_state: bool = True,
    epoch: Optional[int] = None,
    loss: Optional[float] = None,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    log: bool = False
) -> None:
    if log:
        print(f"\n--- Saving Checkpoint to: {filepath} ---")

    if save_full_state:
        # Create a dictionary for the full training state
        checkpoint: Dict[str, Any] = {
            'model_state_dict': model.state_dict(),
            'save_full_state': True,
        }

        # Conditionally add other training components
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if loss is not None:
            checkpoint['loss'] = loss
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            if hasattr(scheduler, 'state_dict'):
                 checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            else:
                 if log:
                    print("Warning: Scheduler provided does not have a state_dict property.")

        if log:
            print("Mode: Saving FULL training state (Model, Optimizer, Epoch, etc.)")

    else:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_name': model.name,
            'save_full_state': False, 
        }
    torch.save(checkpoint, filepath)
    if log:
        print(f"Checkpoint saved successfully at: {filepath}\n")


def load_checkpoint(
    model: nn.Module,
    filepath: str,
    load_full_state: bool = True,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    log: bool = False
) -> Dict[str, Any]:
    if log:
        print(f"\n--- Loading Checkpoint from: {filepath} ---")

    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    if checkpoint.get('save_full_state', False) and load_full_state:
        if log:
            print("Mode: Loaded FULL training state (Model, Optimizer, Epoch, etc.)")

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            if hasattr(scheduler, 'load_state_dict'):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            else:
                if log:
                    print("Warning: Scheduler provided does not have a load_state_dict method.")

    else:
        if log:
            print("Mode: Loaded Model state only.")

    if log:
        print(f"Checkpoint loaded successfully from: {filepath}\n")

    return checkpoint

def load_models_with_data(path_list):
    pipe = None
    for path in path_list:
        print(f"Loading pipeline from {path}")
        pipe = load_pipeline(path, pipe)
        pipe.load_dataset()
        train, val, test = load_test_data(pipe, path)
        model = load_model_for_eval(pipe, path)
        scheduled_sampling = pipe.build_scheduled_sampling()
        if scheduled_sampling is not None:
            print('>>>>>Loading scheduled sampling')
            model.set_scheduled_sampling(scheduled_sampling)
        yield (model, train, val, test)

def load_pipeline(path, pipe=None):
    model_config = OmegaConf.load(os.path.join(path, '.hydra', 'config.yaml'))
    return PipelineBuilder(model_config)

def load_test_data(pipe: PipelineBuilder, path: str):
    path = os.path.join(path, 'split.pth')
    index_dict = None
    if os.path.exists(path):
        print(f"Loading splits from {path}")
        index_dict = torch.load(path)
    else:
        print(f"Making new splits")
        train_idx, val_idx, test_idx = pipe.make_splits()
        index_dict = {
            'train': train_idx,
            'val': val_idx,
            'test': test_idx
        }
    train, val, test = pipe.build_dataloader(index_dict['train'], index_dict['val'], index_dict['test'])
    return train, val, test

def load_model_for_eval(pipe, path):
    weight_path = os.path.join(path, 'model.pth')
    model = pipe.build_model()
    state_dict = torch.load(weight_path, map_location = 'cpu')
    model_state_dict = state_dict['model_state_dict']
    # remove the _orig_mod prefix from the state dict keys
    new_state_dict = {}
    for k, v in model_state_dict.items():
        # Check if the key starts with the problematic prefix
        if k.startswith('_orig_mod.'):
            # Remove the prefix
            new_key = k[len('_orig_mod.'):]
            new_state_dict[new_key] = v
        else:
            # Keep other keys as they are
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    return model


def save_splits(train_idx, val_idx, test_idx, file_path):
    # replace the subsets in the split dict with their indices
    torch.save({
        'train': train_idx,
        'val': val_idx,
        'test': test_idx
    }, file_path)

def load_splits(path, dataset):
    index_dict = torch.load(os.path.join(path, 'split.pth'))
    train_set = Subset(dataset,index_dict['train'])
    val_set = Subset(dataset,index_dict['val'])
    test_set = Subset(dataset,index_dict['test'])
    return train_set, val_set, test_set