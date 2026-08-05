import os
import torch
from src.training.pipeline import PipelineBuilder
from src.training.pipeline import train
import hydra
from omegaconf import DictConfig, open_dict, OmegaConf

def add_metric_and_checkpoint_paths(config: DictConfig):
    hydra_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    metric_path = os.path.join(hydra_path, "metrics.json")
    checkpoint_path = os.path.join(hydra_path, "model.pth")
    splits_path = os.path.join(hydra_path, "split.pth")
    with open_dict(config):
        config.training.metric_file = metric_path
        config.training.checkpoint_file = checkpoint_path
        config.training.splits_file = splits_path
        if hasattr(config.training, "inference_recorder"):
            config.training.inference_recorder.output_dir = os.path.join(hydra_path, "inference_records")

def init_wandb(config: DictConfig):
    """Initialise a W&B run for a plain ``train.py`` run when ``training.wandb.enabled``.

    ``train()`` only ever logs to an *already-active* run (the hp_search driver owns
    init/finish for its trials); on the plain training path nothing else does, so this
    owns ``wandb.init``/``finish``. Returns the run (or ``None`` when W&B is off) so
    ``main`` can finish it. Mirrors the driver's ``define_metric`` setup so ``epoch`` is
    the x-axis for every ``train/*`` and ``val/*`` curve.
    """
    wcfg = config.training.get("wandb", None)
    if wcfg is None or not wcfg.get("enabled", False):
        return None
    import wandb
    run = wandb.init(
        project=wcfg.get("project", "noisy-eye-scanpath"),
        entity=wcfg.get("entity", None),
        mode=wcfg.get("mode", "online"),
        group=wcfg.get("group", None),
        name=wcfg.get("name", None),
        config=OmegaConf.to_container(config, resolve=True),
    )
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    return run

@hydra.main(config_path="./configs", config_name="main", version_base=None)
def main(config: DictConfig):
    torch.set_float32_matmul_precision('high')
    add_metric_and_checkpoint_paths(config)
    run = init_wandb(config)
    try:
        builder = PipelineBuilder(config)
        train(builder)
    finally:
        if run is not None:
            import wandb
            wandb.finish()
# fixation_len
if __name__ == "__main__":
    main()

