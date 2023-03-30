import pandas as pd
from pathlib import Path
from ..dataset import IceCubeDataset
from ..utils import prepare_sensors
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import torch

class basic_gnn_trainner:
    def __init__(self,
                 model,
                 loss,
                 optimzer,
                 epochs,
                 batchsize,
                 device):
        self.model = model
        self.loss = loss
        self.epochs = epochs
        self.batchsize = batchsize
        self.device = device
        self.opt = optimzer
        self.model.to(device)
        self.loss.to(device)

    def run(self):
        _dtype = {
            "batch_id": "int16",
            "event_id": "int64",
        }
        COMP_NAME = "icecube-neutrinos-in-deep-ice"
        INPUT_PATH = Path(f"data/{COMP_NAME}")
        meta = pd.read_parquet(
            INPUT_PATH / f"train_meta.parquet", columns=["batch_id", "event_id", "azimuth", "zenith"]
        ).astype(_dtype)
        batch_ids = meta["batch_id"].unique()
        sensors = prepare_sensors()

        for i in range(self.epochs):
            torch.save(self.model.state_dict(), 'checkpoints/model_checkpoint{}.pth'.format(i))
            for i, b in enumerate(batch_ids):
                event_ids = meta[meta["batch_id"] == b]["event_id"].tolist()
                y = meta[meta["batch_id"] == b][['zenith', 'azimuth']].reset_index(drop=True)
                dataset = IceCubeDataset(b, event_ids, sensors, mode='train', y=y, )
                train_len = int(0.7 * len(dataset[0:3000]))
                train_loader = DataLoader(dataset[0:train_len], batch_size=self.batchsize)
                val_loader = DataLoader(dataset[train_len:3000], batch_size=self.batchsize)

                print(f'batch {i}/{len(train_loader)}')
                total_loss_train = 0
                self.model.train()
                for sample_batched in tqdm(train_loader, desc='train'):
                    self.opt.zero_grad()
                    sample_batched = sample_batched.to(self.device)
                    outputs = self.model(sample_batched)
                    label = sample_batched.y.reshape(-1, 2).to(self.device)
                    loss = self.loss(outputs, label)
                    total_loss_train += loss.cpu().item()
                    loss.backward()
                    self.opt.step()

                total_loss_val = 0
                self.model.eval()
                with torch.no_grad():
                    for sample_batched in tqdm(val_loader, desc='val'):
                        sample_batched = sample_batched.to(self.device)
                        outputs = self.model(sample_batched)
                        label = sample_batched.y.reshape(-1, 2).to(self.device)
                        loss = self.loss(outputs, label)
                        total_loss_val += loss.cpu().item()
                print("trainning loss"+"testing loss")
                print(total_loss_train / train_len,total_loss_train / (len(dataset[0:3000]) - train_len))