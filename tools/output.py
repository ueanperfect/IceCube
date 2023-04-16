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
COMP_NAME = "../../icecube-neutrinos-in-deep-ice"
INPUT_PATH = Path(f"{COMP_NAME}")

meta = pd.read_parquet(
    INPUT_PATH / f"test_meta.parquet", columns=["batch_id", "event_id"]
).astype(_dtype)

batch_ids = meta["batch_id"].unique().tolist()
sensors = prepare_sensors()

#list of event_ids
# event_ids = [2092, 7344, 9482]
event_ids = meta[meta["batch_id"] == batch_ids[0]]["event_id"].tolist()
dataset = IceCubeDataset(batch_ids[0], event_ids, sensors, mode='test')
val_loader = DataLoader(dataset, batch_size=96)
model = MyGNN(8, 256, 2, 3).to(device)
model.load_state_dict(torch.load('checkpoints/basic_gnn_5_.pth'))

out = []
with torch.no_grad():
    for index, sample_batched in enumerate(val_loader):
        progress_bar("testing..... ",index,len(val_loader))
        sample_batched = sample_batched.to(device)
        outputs = model(sample_batched)
        if device == torch.device('cuda'):
            outputs = outputs.cpu()
        out.append(outputs)

out = np.vstack(out)
az_pred = out[:, 0]
zen_pred = out[:, 1]
# save to csv
df = pd.DataFrame({'event_id': event_ids, 'azimuth': az_pred, 'zenith': zen_pred})
df.to_csv( f"submission.csv", index=False)