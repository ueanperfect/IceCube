from .validator import Validator
from .trainner import Trainner
from .logger import Logger
from pathlib import Path
import pandas as pd
from ..dataset import IceCubeDataset
from ..utils import prepare_sensors,progress_bar
from torch_geometric.loader import DataLoader
import torch


class Runner():
    def __init__(self,
                 model,
                 loss,
                 optimzer,
                 device,
                 logger,
                 max_epoch=12,
                 batchsize = 32,
                 ):

        self.validator = Validator(model,loss,device,logger)
        self.trainner = Trainner(model,loss,optimzer,device,logger)
        self.model = model
        self.max_epoch = max_epoch
        self.batchsize = batchsize
        self.logger = logger

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

        for i, b in enumerate(batch_ids):

            event_ids = meta[meta["batch_id"] == b]["event_id"].tolist()
            y = meta[meta["batch_id"] == b][['zenith', 'azimuth']].reset_index(drop=True)
            dataset = IceCubeDataset(b, event_ids, sensors, mode='train', y=y,)
            train_len = int(0.7 * len(dataset[:10000]))
            train_loader = DataLoader(dataset[0:train_len], batch_size=self.batchsize)
            val_loader = DataLoader(dataset[train_len:10000], batch_size=self.batchsize)
            print(f'data_batch {i}/{len(batch_ids)}')
            self.trainner.train(train_loader)
            self.validator.val(val_loader)
            if (i + 1) % 5 == 0:
                self.logger.save_checkpoint(self.model, i)