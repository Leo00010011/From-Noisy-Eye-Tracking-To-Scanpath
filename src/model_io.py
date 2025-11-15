import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Any, Dict

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