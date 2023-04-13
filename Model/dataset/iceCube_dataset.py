from torch_geometric.data import Data, Dataset
import pandas as pd
import torch
from pathlib import Path
import numpy as np
from ..utils import ice_transparency
from torch_geometric.nn import knn_graph


class IceCubeDataset(Dataset):
    def __init__(
            self,
            batch_id,
            event_ids,
            sensor_df,
            mode="test",
            y=None,
            pulse_limit=300,
            transform=None,
            pre_transform=None,
            pre_filter=None,
    ):
        super().__init__(transform, pre_transform, pre_filter)

        COMP_NAME = "icecube-neutrinos-in-deep-ice"
        INPUT_PATH = Path(f"data/{COMP_NAME}")
        OUTPUT_PATH = Path(f"data/{COMP_NAME}")
        MODEL_CACHE = Path("/mnt/storage/model_cache/torch")
        TRANSPARENCY_PATH = INPUT_PATH / "ice_transparency.txt"

        self.y = y
        self.event_ids = event_ids
        self.batch_df = pd.read_parquet(INPUT_PATH / mode / f"batch_{batch_id}.parquet")
        self.sensor_df = sensor_df
        self.pulse_limit = pulse_limit
        self.f_scattering, self.f_absorption = ice_transparency(TRANSPARENCY_PATH)

        self.batch_df["time"] = (self.batch_df["time"] - 1.0e04) / 3.0e4
        self.batch_df["charge"] = np.log10(self.batch_df["charge"]) / 3.0
        self.batch_df["auxiliary"] = self.batch_df["auxiliary"].astype(int) - 0.5

    def len(self):
        return len(self.event_ids)

    def get(self, idx):
        event_id = self.event_ids[idx]
        event = self.batch_df.loc[event_id]

        # represent each event by a single graph
        event = pd.merge(event, self.sensor_df, on="sensor_id")
        col = ["x", "y", "z", "time", "charge", "qe", "auxiliary"]

        x = event[col].values
        x = torch.tensor(x, dtype=torch.float32)
        data = Data(x=x, n_pulses=torch.tensor(x.shape[0], dtype=torch.int32))

        # Add ice transparency data
        z = data.x[:, 2].numpy()
        scattering = torch.tensor(self.f_scattering(z), dtype=torch.float32).view(-1, 1)
        # absorption = torch.tensor(self.f_absorption(z), dtype=torch.float32).view(-1, 1)

        data.x = torch.cat([data.x, scattering], dim=1)

        # Downsample the large events
        if data.n_pulses > self.pulse_limit:
            data.x = data.x[np.random.choice(data.n_pulses, self.pulse_limit)]
            data.n_pulses = torch.tensor(self.pulse_limit, dtype=torch.int32)

        # Builds graph from the k-nearest neighbours.
        data.edge_index = knn_graph(
            data.x[:, [0, 1, 2]],
            k=8,
            batch=None,
            loop=False
        )

        if self.y is not None:
            y = self.y.loc[idx, :].values
            y = torch.tensor(y, dtype=torch.float32)
            data.y = y

        return data


class IceCubeDatasetLstm(Dataset):
    def __init__(
            self,
            batch_ids,
            mode="test",
            transform=None,
            pre_transform=None,
            pre_filter=None,
    ):
        super().__init__(transform, pre_transform, pre_filter)

        if len(batch_ids) > 50:
            raise ValueError(" your input file is too much, please load less and seperate batch")
        self.train_x = None
        self.train_y = None
        INPUT_PATH = "data/LSTM_data/"
        for batch_id in batch_ids:
            try:
                data = np.load(INPUT_PATH + f"pointpicker_mpc128_n9_batch_{batch_id}.npz")
                if self.train_x is None:
                    self.train_x = data["x"]
                    self.train_y = data["y"]
                else:
                    self.train_x = np.append(self.train_x, data["x"], axis=0)
                    self.train_y = np.append(self.train_y, data["y"], axis=0)
            except:
                data = np.load(INPUT_PATH + f"pointpicker_mpc256_n9_batch_{batch_id}.npz")
                if self.train_x is None:
                    self.train_x = data["x"][:,:128,:]
                    self.train_y = data["y"]
                else:
                    self.train_x = np.append(self.train_x, data["x"][:,:128,:], axis=0)
                    self.train_y = np.append(self.train_y, data["y"], axis=0)


        self.train_x[:, :, 0] /= self.train_x[:, :, 0].max()
        self.train_x[:, :, 1] /= self.train_x[:, :, 1].max()
        self.train_x[:, :, 3:] /= 600  # space
        self.true_label = torch.from_numpy(self.train_y)
        self.train_y = self.y_to_code(self.train_y, 16)

        self.train_x = torch.from_numpy(self.train_x[:,:,:8])
        self.train_x = self.train_x.to(torch.float32)
        self.train_y = torch.from_numpy(self.train_y)

        # 对数据顺序进行打乱
        # indices = torch.randperm(len(self.train_x))
        # self.train_x = self.train_x[indices]
        # self.train_y = self.train_y[indices]

    def y_to_code(self, batch_y, bin_num=16):
        azimuth_edges = np.linspace(0, 2 * np.pi, bin_num + 1)
        zenith_edges_flat = np.linspace(0, np.pi, bin_num + 1)
        zenith_edges = list()
        zenith_edges.append(0)
        for bin_idx in range(1, bin_num):
            zen_now = np.arccos(np.cos(zenith_edges[-1]) - 2 / (bin_num))
            zenith_edges.append(zen_now)
        zenith_edges.append(np.pi)
        zenith_edges = np.array(zenith_edges)
        # evaluate bin code
        azimuth_code = (batch_y[:, 0] > azimuth_edges[1:].reshape((-1, 1))).sum(axis=0)
        zenith_code = (batch_y[:, 1] > zenith_edges[1:].reshape((-1, 1))).sum(axis=0)
        angle_code = bin_num * azimuth_code + zenith_code
        return angle_code

    def len(self):
        return len(self.train_x)

    def get(self, idx):
        x = self.train_x[idx]
        y = self.train_y[idx]
        true_label = self.true_label[idx]
        return x, y, true_label