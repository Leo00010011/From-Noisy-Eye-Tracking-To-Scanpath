import torch
from tqdm import tqdm
from src.training.training_utils import compute_loss, validate, move_data_to_device, MetricsStorage
from src.training.pipeline_builder import PipelineBuilder
from src.model.model_io import save_checkpoint, save_splits
import os
import torch
from src.training.pipeline import PipelineBuilder
from src.training.pipeline import train
import hydra
from omegaconf import DictConfig, open_dict, OmegaConf

def train(builder:PipelineBuilder):
        builder.load_dataset()
        train_idx, val_idx, test_idx = builder.make_splits()
        train_dataloader, val_dataloader, _ = builder.build_dataloader(train_idx, val_idx, test_idx)
        builder.clear_dataframe()
        for batch in tqdm(train_dataloader):#
            input
            
            
            
def add_metric_and_checkpoint_paths(config: DictConfig):
    hydra_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    metric_path = os.path.join(hydra_path, "metrics.json")
    checkpoint_path = os.path.join(hydra_path, "model.pth")
    splits_path = os.path.join(hydra_path, "split.pth")
    with open_dict(config):
        config.training.metric_file = metric_path
        config.training.checkpoint_file = checkpoint_path
        config.training.splits_file = splits_path

@hydra.main(config_path="./configs", config_name="main", version_base=None)
def main(config: DictConfig):
    torch.set_float32_matmul_precision('high')
    add_metric_and_checkpoint_paths(config)
    builder = PipelineBuilder(config)
    train(builder)
# fixation_len
if __name__ == "__main__":
    main()
