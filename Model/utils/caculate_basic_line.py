import numpy as np
from sklearn.decomposition import PCA
from ..dataset import IceCubeDataset
from pathlib import Path
import pandas as pd
from .data_preprocess import prepare_sensors


def initial_data(batch_id):
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

    event_ids = meta[meta["batch_id"] == batch_id]["event_id"].tolist()
    y = meta[meta["batch_id"] == batch_id][['zenith', 'azimuth']].reset_index(drop=True)
    dataset = IceCubeDataset(batch_id, event_ids, sensors, mode='train', y=y)
    return dataset


def calculate_angles(direction):
    azimuth = np.arctan2(direction[1], direction[0])
    if azimuth < 0:
        azimuth += 2 * np.pi
    elevation = np.arctan2(direction[2], np.sqrt(direction[0] ** 2 + direction[1] ** 2))
    return azimuth, elevation


def extract_direction(batch_id, event_id):
    dataset = initial_data(batch_id)[event_id]
    points = dataset.x.numpy()[:, :3]
    pca = PCA(n_components=1)
    pca.fit(points)
    direction = pca.components_[0]
    center = np.mean(points, axis=0)
    azimuth_rad, elevation_rad = calculate_angles(direction)
    return azimuth_rad, elevation_rad, center
