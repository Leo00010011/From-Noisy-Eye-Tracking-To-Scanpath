import torch

class WeightsScheduler:
    def __init__(self,init_b, end_b, epochs, loss_function, device, dtype = torch.float32):
        self.init_b = init_b
        self.end_b = end_b
        self.epochs = epochs
        self.loss_function = loss_function
        self.device = device
        self.dtype = dtype
        self.epoch = 0
        self.x = torch.arange(15, device = device, dtype = dtype)
        self.b = torch.linspace(0.8,0.999,epochs, device = device, dtype = dtype)
    def compute_weights(self):
        return (self.b[self.epoch]**self.x) + 1
    
    def start(self):
        self.loss_function.set_weights(self.compute_weights())
    
    def update_epoch(self):
        if self.epoch < self.epochs:
            self.epoch += 1
            self.loss_function.set_weights(self.compute_weights())
        
        