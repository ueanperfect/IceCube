import numpy as np
from torch_geometric.data import Dataset
import torch

class IceCubeDatasetClassification(Dataset):
    def __init__(
            self,
            batch_ids,
            bin_number = 10,
    ):
        super().__init__()
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
        self.train_y = self.y_to_code(self.train_y, bin_number)

        self.train_x = torch.from_numpy(self.train_x[:,:,:8])
        self.train_x = self.train_x.to(torch.float32)
        self.train_y = torch.from_numpy(self.train_y)

    def y_to_code(self, batch_y, bin_num=12):
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