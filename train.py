import os
import torch
from src.pipeline_builder import PipelineBuilder
from src.pipeline import train
import hydra
from omegaconf import DictConfig, open_dict, OmegaConf

def add_metric_and_checkpoint_paths(config: DictConfig):
    hydra_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    metric_path = os.path.join(hydra_path, "metrics.json")
    checkpoint_path = os.path.join(hydra_path, "model.pth")
    with open_dict(config):
        config.training.metric_file = metric_path
        config.training.checkpoint_file = checkpoint_path

def half_dim(dim_value):
    """Calculates half of the input dimension."""
    # Ensure the result is an integer for layer dimensions
    return int(dim_value / 2)

def double_dim(dim_value):
    """Calculates double the input dimension."""
    # Ensure the result is an integer for layer dimensions
    return int(dim_value * 2)

# 2. Register the resolver
OmegaConf.register_new_resolver("half", half_dim)
OmegaConf.register_new_resolver("double", double_dim)

@hydra.main(config_path="./configs", config_name="main", version_base=None)
def main(config: DictConfig):
    torch.set_float32_matmul_precision('high')
    add_metric_and_checkpoint_paths(config)
    builder = PipelineBuilder(config)
    train(builder)
# fixation_len
if __name__ == "__main__":
    main()

