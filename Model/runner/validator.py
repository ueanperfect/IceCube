import torch
from ..utils import progress_bar
import numpy as np


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
        total_loss_val = 0
        with torch.no_grad():
            for index, sample_batched in enumerate(val_loader):
                sample_batched = sample_batched.to(self.device)
                outputs = self.model(sample_batched)
                label = sample_batched.y.reshape(-1, 2).to(self.device)
                loss = self.loss(outputs, label)
                total_loss_val += loss.cpu().item()
                # calculate score
                az_true = label[:, 0].cpu().numpy()
                zen_true = label[:, 1].cpu().numpy()
                az_pred = outputs[:, 0].cpu().numpy()
                zen_pred = outputs[:, 1].cpu().numpy()
                if not (np.all(np.isfinite(az_true)) and
                        np.all(np.isfinite(zen_true)) and
                        np.all(np.isfinite(az_pred)) and
                        np.all(np.isfinite(zen_pred))):
                    raise ValueError("All arguments must be finite")
                sz1 = np.sin(zen_true)
                cz1 = np.cos(zen_true)
                sz2 = np.sin(zen_pred)
                cz2 = np.cos(zen_pred)

                scalar_prod = np.clip(sz1 * sz2 * (np.cos(az_true - az_pred)) + (cz1 * cz2), -1, 1)
                score = np.average(np.abs(np.arccos(scalar_prod)))
        self.logger.print_evaluate_information(total_loss_val/len(val_loader),score)
