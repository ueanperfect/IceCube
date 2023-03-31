from Model import IceCubeDataset,prepare_sensors,MyGNN,progress_bar
import torch
import pandas as pd
from pathlib import Path
from torch_geometric.loader import DataLoader
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

event_ids = meta[meta["batch_id"] == 660]["event_id"].tolist()
y = meta[meta["batch_id"] == 660][['zenith', 'azimuth']].reset_index(drop=True)
dataset = IceCubeDataset(660, event_ids, sensors, mode='train', y=y)
val_loader = DataLoader(dataset, batch_size=96)

model = MyGNN(8, 256, 2, 3).to(device)

model.load_state_dict(torch.load('checkpoints/basic_gnn/2023-03-31_12-36-14/checkpoints/basic_gnn_10_.pth'))

score = 0
with torch.no_grad():
    for index, sample_batched in enumerate(val_loader):
        progress_bar("testing..... ",index,len(val_loader))
        sample_batched = sample_batched.to(device)
        outputs = model(sample_batched)
        label = sample_batched.y.reshape(-1, 2).to(device)
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
        score = score + np.average(np.abs(np.arccos(scalar_prod)))
    print(f'testing score is {score/len(val_loader)}')