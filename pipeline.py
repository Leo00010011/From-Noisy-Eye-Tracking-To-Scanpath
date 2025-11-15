

import os
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from src.datasets import FreeViewInMemory, seq2seq_jagged_collate_fn, seq2seq_padded_collate_fn
from src.model import PathModel
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf


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
        train_dataloader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0, collate_fn= seq2seq_padded_collate_fn)
        val_dataloader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=0, collate_fn= seq2seq_padded_collate_fn)
        test_dataloader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=0, collate_fn= seq2seq_padded_collate_fn)
        return train_dataloader, val_dataloader, test_dataloader
    
    def build_model(self) -> PathModel:
        activation = None
        if self.config.model.activation == "relu":
            activation = torch.nn.ReLU()


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

    def train(self):
        cls_weight = self.config.training.cls_weight    
        num_epochs = self.config.training.num_epochs
        needs_validate = self.config.training.validate
        val_interval = self.config.training.val_interval
        device = self.device
        optimizer = torch.optim.Adam(model.parameters(), 
                                     lr = self.config.training.learning_rate)
        if self.config.model.compile:
            if self.config.training.log:
                print('starting model compilation')
            model = torch.compile(model) 
        
        self.build_dataloader()
        train_dataloader, val_dataloader, _ = self.build_dataloader()
        reg_loss_list = []
        cls_loss_list = []
        metrics = {'accuracy': [], 'precision_pos': [], 'recall_pos': [], 'precision_neg': [], 'recall_neg': [], 'epoch': []}
        first_time = True
        for epoch in range(num_epochs):
            model.train()  # Set the model to training mode
            total_reg_loss = 0
            total_cls_loss = 0
            num_batches = 0
            if first_time and self.config.training.log:
                print('starting data loading')
            for batch in tqdm(train_dataloader):
                x,x_mask,y, y_mask, fixation_len = batch
                x = x.to(device=device)
                y = y.to(device=device)
                if x_mask is not None:
                    x_mask = x_mask.to(device = device)
                if y_mask is not None:
                    y_mask = y_mask.to(device = device)
                fixation_len = fixation_len.to(device = device)

                optimizer.zero_grad()  # Zero the gradients
                if first_time and self.config.training.log:
                    print('model compilation')
                first_time = False
                reg_out, cls_out = model(x,y, src_mask = x_mask, tgt_mask = y_mask)  # Forward pass
                cls_loss, reg_loss = compute_loss(reg_out,cls_out, y, y_mask, fixation_len) # Compute loss
                total_loss = (1-cls_weight)*reg_loss + cls_weight*cls_loss
                total_loss.backward()
                optimizer.step()
                total_reg_loss += reg_loss.item()
                total_cls_loss += cls_loss.item()
                num_batches += 1

            if needs_validate and ((epoch + 1) % val_interval == 0):
                validate(model, val_dataloader, epoch, device, metrics)
            
            avg_reg_loss = total_reg_loss / num_batches
            avg_cls_loss = total_cls_loss / num_batches
            reg_loss_list.append(avg_reg_loss)
            cls_loss_list.append(avg_cls_loss)
            print(f"Epoch {epoch+1}/{num_epochs}, Avg Regression Loss: {avg_reg_loss:.4f}, Avg Classification Loss: {avg_cls_loss:.4f}")

        print("Training finished!")

class Pipeline:
    pass


@hydra.main(config_path="./configs", config_name="main", version_base=None)
def main(config):
    builder = Pipeline(config)
    print(builder.training_config())

# compute_loss(reg_out,cls_out, y, y_mask, fixation_len)

# fixation_len
if __name__ == "__main__":
    main()

