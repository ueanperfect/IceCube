import torch
from ..utils import progress_bar
class Validator:

    def __init__(self,
                 model,
                 loss,
                 device,
                 logger):
        self.model = model
        self.loss = loss
        self.device = device
        self.logger = logger

    def val(self,val_loader):
        self.model.eval()
        print("\ntesting process")
        total_loss_val = 0
        with torch.no_grad():
            for index, sample_batched in enumerate(val_loader):
                progress_bar(index + 1, len(val_loader), bar_length=30)
                sample_batched = sample_batched.to(self.device)
                outputs = self.model(sample_batched)
                label = sample_batched.y.reshape(-1, 2).to(self.device)
                loss = self.loss(outputs, label)
                total_loss_val += loss.cpu().item()