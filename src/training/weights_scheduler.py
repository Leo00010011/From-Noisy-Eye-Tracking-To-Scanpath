import torch

class WeightsScheduler:
    def __init__(self,init_b, end_b, epochs, alpha, loss_function, device, dtype = torch.float32):
        self.init_b = init_b
        self.end_b = end_b
        self.epochs = epochs
        self.alpha = alpha
        self.loss_function = loss_function
        self.device = device
        self.dtype = dtype
        self.epoch = 0
        self.x = torch.arange(26, device = device, dtype = dtype)
        self.b = torch.linspace(0.8,0.999,epochs, device = device, dtype = dtype)
        self.loss_function.set_weights(self.compute_weights())
    def compute_weights(self):
        if self.epoch < self.epochs:
            return (self.alpha* (self.b[self.epoch]**self.x) + 1)/(1 + self.alpha)
        else:
            return None
    
    def update_epoch(self):
        self.epoch += 1
        if self.epoch < self.epochs:
            self.loss_function.set_weights(self.compute_weights())
        
        