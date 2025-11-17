import os
import json
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import hydra
from omegaconf import DictConfig, open_dict
from src.datasets import FreeViewInMemory, seq2seq_padded_collate_fn
from src.model import PathModel
from src.training_utils import compute_loss, validate, move_data_to_device
from src.model_io import save_checkpoint

class MetricsStorage:
    def __init__(self, filepath: str = None, decisive_metric: str = 'reg_loss_val'):
        self.metrics = {
            'epoch': [],
            'reg_loss_train': [],
            'reg_loss_val': [],
            'cls_loss_train': [],
            'cls_loss_val': [],
            'accuracy': [],
            'precision_pos': [],
            'recall_pos': [],
            'precision_neg': [],
            'recall_neg': []
        }
        self.total_reg_loss = 0
        self.total_cls_loss = 0
        self.num_batches = 0
        self.filepath = filepath
        self.decisive_metric = decisive_metric
        self.best_metric_value = np.inf
    
    def init_epoch(self):
        self.total_reg_loss = 0
        self.total_cls_loss = 0
        self.num_batches = 0

    def update_batch_loss(self, reg_loss, cls_loss):
        self.total_reg_loss += reg_loss.item()
        self.total_cls_loss += cls_loss.item()
        self.num_batches += 1
    
    def finalize_epoch(self):
        avg_reg_loss = self.total_reg_loss / self.num_batches
        avg_cls_loss = self.total_cls_loss / self.num_batches
        self.metrics['reg_loss_train'].append(avg_reg_loss)
        self.metrics['cls_loss_train'].append(avg_cls_loss)
        return avg_reg_loss, avg_cls_loss
    
    def update_best(self):
        if self.metrics[self.decisive_metric][-1] < self.best_metric_value:
            self.best_metric_value = self.metrics[self.decisive_metric][-1]
            return True
        return False

    def save_metrics(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.metrics, f)


class Pipeline:
    def __init__(self, config: DictConfig):
        # dataset parameters
        self.config = config
        if self.config.model.device.startswith('cuda') and torch.cuda.is_available():
            self.device = torch.device(self.config.model.device)
        else:
            if self.config.model.device.startswith('cuda'):
                print("CUDA not available, using CPU.")
            self.device = torch.device('cpu')

    def build_dataloader(self) -> DataLoader:

        datasetv2 = FreeViewInMemory(sample_size= self.config.data.sample_size,
                                     log = self.config.data.log, 
                                     start_index=self.config.data.start_index)
        total_size = len(datasetv2)
        train_size = int(self.config.data.train_per * total_size)
        val_size = int(self.config.data.val_per * total_size)
        test_size = total_size - train_size - val_size
        train_set, val_set, test_set = random_split(datasetv2, [train_size, val_size, test_size])
        train_dataloader = DataLoader(train_set, batch_size=self.config.training.batch_size, shuffle=True, num_workers=0, collate_fn= seq2seq_padded_collate_fn)
        val_dataloader = DataLoader(val_set, batch_size=self.config.training.batch_size, shuffle=False, num_workers=0, collate_fn= seq2seq_padded_collate_fn)
        test_dataloader = DataLoader(test_set, batch_size=self.config.training.batch_size, shuffle=False, num_workers=0, collate_fn= seq2seq_padded_collate_fn)
        return train_dataloader, val_dataloader, test_dataloader
    
    def build_model(self) -> PathModel:
        activation = None
        if self.config.model.activation == "relu":
            activation = torch.nn.ReLU()
        elif self.config.model.activation == "gelu":
            activation = torch.nn.GELU()

        model = PathModel(input_dim = self.config.model.input_dim,
                          output_dim = self.config.model.output_dim,
                          n_encoder = self.config.model.n_encoder,
                          n_decoder = self.config.model.n_decoder,
                          model_dim = self.config.model.model_dim,
                          total_dim = self.config.model.total_dim,
                          n_heads = self.config.model.n_heads,
                          ff_dim = self.config.model.ff_dim,
                          max_pos_enc = self.config.model.max_pos_enc,
                          max_pos_dec = self.config.model.max_pos_dec,
                          norm_first = self.config.model.norm_first,
                          dropout_p= self.config.model.dropout_p,
                          activation = activation,
                          device = self.device)
        return model
    
    def build_optimizer(self, model: PathModel):
        optimizer = torch.optim.Adam(model.parameters(), 
                                     lr = self.config.training.learning_rate)
        return optimizer
    
    def build_scheduler(self, optimizer: torch.optim.Optimizer, train_dataloader: DataLoader):
        sample_count = len(train_dataloader.dataset)
        steps_per_epoch = int(np.ceil(sample_count / train_dataloader.batch_size))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.scheduler.max_lr,
            total_steps=None,
            epochs=self.config.training.num_epochs,
            steps_per_epoch= steps_per_epoch,
            pct_start=self.config.scheduler.pct_start,
            div_factor=self.config.scheduler.div_factor,
            final_div_factor=self.config.scheduler.final_div_factor,
        )
        return scheduler

    def train(self):
        cls_weight = self.config.training.cls_weight    
        num_epochs = self.config.training.num_epochs
        needs_validate = self.config.training.validate
        val_interval = self.config.training.val_interval
        device = self.device
        model = self.build_model()
        if self.config.model.compilate:
            model = torch.compile(model) 
        optimizer = self.build_optimizer(model)
        train_dataloader, val_dataloader, _ = self.build_dataloader()
        scheduler = self.build_scheduler(optimizer, train_dataloader)
        first_time = True
        metrics_storage = MetricsStorage(filepath=self.config.training.metric_file, 
                                         decisive_metric=self.config.training.decisive_metric)
        for epoch in range(num_epochs):
            model.train()  # Set the model to training mode
            metrics_storage.init_epoch()
            if first_time and self.config.training.log:
                print('starting data loading')
            for batch in tqdm(train_dataloader):#
                # LOAD DATA TO DEVICE
                x,x_mask,y, y_mask, fixation_len = move_data_to_device(batch, device)
                optimizer.zero_grad()  # Zero the gradients
                if first_time and self.config.training.log and self.config.model.compilate:
                    print('model compilation')
                    first_time = False
                # FORWARD PASS AND LOSS COMPUTATION
                reg_out, cls_out = model(x,y, src_mask = x_mask, tgt_mask = y_mask)  # Forward pass
                cls_loss, reg_loss = compute_loss(reg_out,cls_out, y, y_mask, fixation_len) # Compute loss
                total_loss = (1-cls_weight)*reg_loss + cls_weight*cls_loss
                # BACKWARD PASS AND OPTIMIZATION
                total_loss.backward()
                optimizer.step()
                scheduler.step()
                metrics_storage.update_batch_loss(reg_loss, cls_loss)
            avg_reg_loss, avg_cls_loss = metrics_storage.finalize_epoch()
            print(f"Epoch {epoch+1}/{num_epochs}, Avg Regression Loss: {avg_reg_loss:.4f}, Avg Classification Loss: {avg_cls_loss:.4f}")

            if needs_validate and ((epoch + 1) % val_interval == 0):
                validate(model, val_dataloader, epoch, device, metrics_storage.metrics, log = self.config.training.log)
                metrics_storage.save_metrics()
                is_best = metrics_storage.update_best()
                if is_best:
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        filepath=self.config.training.checkpoint_file,
                        epoch=epoch,
                        log=self.config.training.log,
                        save_full_state= self.config.training.save_full_state
                    )

        print("Training finished!")

def add_metric_and_checkpoint_paths(config: DictConfig):
    hydra_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    metric_path = os.path.join(hydra_path, "metrics.json")
    checkpoint_path = os.path.join(hydra_path, "model.pth")
    with open_dict(config):
        config.training.metric_file = metric_path
        config.training.checkpoint_file = checkpoint_path

@hydra.main(config_path="./configs", config_name="main", version_base=None)
def main(config: DictConfig):
    add_metric_and_checkpoint_paths(config)
    builder = Pipeline(config)
    builder.train()

# fixation_len
if __name__ == "__main__":
    main()

