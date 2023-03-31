import pandas as pd
from pathlib import Path
from ..dataset import IceCubeDataset
from ..utils import prepare_sensors, progress_bar
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import torch


class Trainner:
    def __init__(self,
                 model,
                 loss,
                 optimzer,
                 device,
                 logger):
        self.model = model
        self.loss = loss
        self.device = device
        self.opt = optimzer
        self.model.to(device)
        self.loss.to(device)
        self.logger = logger

    def train(self, dataloader):
        iter_number = len(dataloader)
        record_number = int(iter_number / 5)

        for index, sample_batched in enumerate(dataloader):
            self.opt.zero_grad()
            sample_batched = sample_batched.to(self.device)
            outputs = self.model(sample_batched)
            label = sample_batched.y.reshape(-1, 2).to(self.device)
            loss = self.loss(outputs, label)
            loss.backward()
            self.opt.step()
            if index % record_number == 0:
                self.logger.print_training_information(loss.cpu().item(), index, len(dataloader))
