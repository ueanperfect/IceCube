from Model import IceCubeDatasetForSubmission, prepare_sensors, IceCubeDatasetClassification, TransformerEncoderV1
import torch
import pandas as pd
from pathlib import Path
from torch_geometric.loader import DataLoader
import numpy as np


def caculate_angle_bin_vector(bin_num=12):
    azimuth_edges = np.linspace(0, 2 * np.pi, bin_num + 1)
    zenith_edges_flat = np.linspace(0, np.pi, bin_num + 1)
    zenith_edges = list()
    zenith_edges.append(0)
    for bin_idx in range(1, bin_num):
        # cos(zen_before) - cos(zen_now) = 2 / bin_num
        zen_now = np.arccos(np.cos(zenith_edges[-1]) - 2 / (bin_num))
        zenith_edges.append(zen_now)
    zenith_edges.append(np.pi)
    zenith_edges = np.array(zenith_edges)

    angle_bin_zenith0 = np.tile(zenith_edges[:-1], bin_num)
    angle_bin_zenith1 = np.tile(zenith_edges[1:], bin_num)
    angle_bin_azimuth0 = np.repeat(azimuth_edges[:-1], bin_num)
    angle_bin_azimuth1 = np.repeat(azimuth_edges[1:], bin_num)

    angle_bin_area = (angle_bin_azimuth1 - angle_bin_azimuth0) * (
            np.cos(angle_bin_zenith0) - np.cos(angle_bin_zenith1))
    angle_bin_vector_sum_x = (np.sin(angle_bin_azimuth1) - np.sin(angle_bin_azimuth0)) * (
            (angle_bin_zenith1 - angle_bin_zenith0) / 2 - (
            np.sin(2 * angle_bin_zenith1) - np.sin(2 * angle_bin_zenith0)) / 4)
    angle_bin_vector_sum_y = (np.cos(angle_bin_azimuth0) - np.cos(angle_bin_azimuth1)) * (
            (angle_bin_zenith1 - angle_bin_zenith0) / 2 - (
            np.sin(2 * angle_bin_zenith1) - np.sin(2 * angle_bin_zenith0)) / 4)
    angle_bin_vector_sum_z = (angle_bin_azimuth1 - angle_bin_azimuth0) * (
            (np.cos(2 * angle_bin_zenith0) - np.cos(2 * angle_bin_zenith1)) / 4)

    angle_bin_vector_mean_x = angle_bin_vector_sum_x / angle_bin_area
    angle_bin_vector_mean_y = angle_bin_vector_sum_y / angle_bin_area
    angle_bin_vector_mean_z = angle_bin_vector_sum_z / angle_bin_area

    angle_bin_vector = np.zeros((1, bin_num * bin_num, 3))
    angle_bin_vector[:, :, 0] = angle_bin_vector_mean_x
    angle_bin_vector[:, :, 1] = angle_bin_vector_mean_y
    angle_bin_vector[:, :, 2] = angle_bin_vector_mean_z
    return angle_bin_vector


def pred_to_angle(pred, epsilon=1e-8, bin_num=12):
    # convert prediction to vector
    pred_vector = (pred.reshape((-1, bin_num * bin_num, 1)) * angle_bin_vector).sum(axis=1)

    # normalize
    pred_vector_norm = np.sqrt((pred_vector ** 2).sum(axis=1))
    mask = pred_vector_norm < epsilon
    pred_vector_norm[mask] = 1

    # assign <1, 0, 0> to very small vectors (badly predicted)
    pred_vector /= pred_vector_norm.reshape((-1, 1))
    pred_vector[mask] = np.array([1., 0., 0.])

    # convert to angle
    azimuth = np.arctan2(pred_vector[:, 1], pred_vector[:, 0])
    azimuth[azimuth < 0] += 2 * np.pi
    zenith = np.arccos(pred_vector[:, 2])

    return azimuth, zenith


bin_number = 10
d_model = 8  # 模型维度
num_heads = 2  # 多头注意力的头数
d_ff = 1024  # 前馈神经网络的隐藏层维度
num_layers = 8  # Transformer Encoder层数

angle_bin_vector = caculate_angle_bin_vector(bin_num=bin_number)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = IceCubeDatasetClassification([661], bin_number=bin_number)
test_dataloader = DataLoader(dataset, batch_size=96)
model = TransformerEncoderV1(d_model, num_heads, d_ff, num_layers, bin_number=bin_number)

with torch.no_grad():
    for index, (x, y, true_label) in enumerate(test_dataloader):
        outputs = model(x)
        az_true = true_label[:, 0].cpu().numpy()
        zen_true = true_label[:, 1].cpu().numpy()
        az_pred, zen_pred = pred_to_angle(outputs.cpu().numpy(),bin_num = bin_number)


_dtype = {
            "batch_id": "int16",
            "event_id": "int64",
        }
COMP_NAME = "data/icecube-neutrinos-in-deep-ice"
INPUT_PATH = Path(f"{COMP_NAME}")

meta = pd.read_parquet(
    INPUT_PATH / f"test_meta.parquet", columns=["batch_id", "event_id"]
).astype(_dtype)

batch_ids = meta["batch_id"].unique().tolist()

event_ids = meta[meta["batch_id"] == batch_ids[0]]["event_id"].tolist()

df = pd.DataFrame({'event_id': event_ids, 'azimuth': az_pred, 'zenith': zen_pred})
df.to_csv( f"submission.csv", index=False)

# _dtype = {
#             "batch_id": "int16",
#             "event_id": "int64",
#         }
# COMP_NAME = "data/icecube-neutrinos-in-deep-ice"
# INPUT_PATH = Path(f"{COMP_NAME}")
#
# meta = pd.read_parquet(
#     INPUT_PATH / f"test_meta.parquet", columns=["batch_id", "event_id"]
# ).astype(_dtype)
#
# batch_ids = meta["batch_id"].unique().tolist()
# sensors = prepare_sensors()
#
# event_ids = meta[meta["batch_id"] == batch_ids[0]]["event_id"].tolist()
# dataset = IceCubeDatasetForSubmission(batch_ids[0], event_ids, sensors, mode='test')
# val_loader = DataLoader(dataset, batch_size=96)
# data = dataset.get(0)


# model.load_state_dict(torch.load('checkpoints/basic_gnn_5_.pth'))

# out = []
# with torch.no_grad():
#     for index, sample_batched in enumerate(val_loader):
#         progress_bar("testing..... ",index,len(val_loader))
#         sample_batched = sample_batched.to(device)
#         outputs = model(sample_batched)
#         if device == torch.device('cuda'):
#             outputs = outputs.cpu()
#         out.append(outputs)
#
# out = np.vstack(out)
# az_pred = out[:, 0]
# zen_pred = out[:, 1]
# # save to csv

