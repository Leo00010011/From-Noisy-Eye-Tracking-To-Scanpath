import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
from src.data.datasets import FreeViewInMemory, seq2seq_padded_collate_fn
from src.model.model import PathModel



class PipelineBuilder:
    def __init__(self, config):
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
                          head_type = self.config.model.get('head_type', 'linear'),
                          mlp_head_hidden_dim = self.config.model.get('mlp_head_hidden_dim', None),
                          activation = activation,
                          device = self.device)
        return model
    
    def build_optimizer(self, model: PathModel):
        optimizer = torch.optim.Adam(model.parameters(), 
                                     lr = self.config.training.learning_rate)
        return optimizer
    
    def build_scheduler(self, optimizer: torch.optim.Optimizer, train_dataloader: DataLoader):
        if self.config.scheduler.type == 'one_cycle':
            if self.config.training.log:
                print("Using One Cycle Learning Rate Scheduler")
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
        elif self.config.scheduler.type == 'multistep_lr':
            if self.config.training.log:
                print("Using MultiStep Learning Rate Scheduler")
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.config.scheduler.milestones,
                gamma=self.config.scheduler.gamma
            )
        else:
            raise ValueError(f"Scheduler type {self.config.scheduler.type} not supported.")
        return scheduler

    def training_summary(self, n_samples):
        print(""" Traning Summary:
                Number of epochs: {}
                Classification Loss Weight: {}
                Validation every {} epochs
                Training Percentage: {}
                Training Samples: {}
                """.format(self.config.training.num_epochs, 
                           self.config.training.cls_weight, 
                           self.config.training.val_interval, 
                           self.config.data.train_per, 
                           n_samples))

    
