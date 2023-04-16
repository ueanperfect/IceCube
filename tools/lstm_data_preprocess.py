import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import time
import os
import gc
from tqdm.notebook import tqdm

# multiprocessing
import multiprocessing

# %% [code] {"execution":{"iopub.status.busy":"2023-03-15T22:10:36.105388Z","iopub.execute_input":"2023-03-15T22:10:36.105808Z","iopub.status.idle":"2023-03-15T22:10:36.127331Z","shell.execute_reply.started":"2023-03-15T22:10:36.105779Z","shell.execute_reply":"2023-03-15T22:10:36.125591Z"}}
# Data setting
train_batch_id_first = 1
train_batch_id_last = 659
train_batch_ids = range(train_batch_id_first, train_batch_id_last + 1)

# Feature Settings
max_pulse_count = 96
n_features = 7  # time, charge, aux, x, y, z, rank

# Directories
home_dir = "data/icecube-neutrinos-in-deep-ice/"
train_format = home_dir + 'train/batch_{batch_id:d}.parquet'
point_picker_format = 'data/lstm_format/pp_mpc96_n7_batch_{batch_id:d}.npz'

# %% [code] {"execution":{"iopub.status.busy":"2023-03-15T22:10:36.12884Z","iopub.execute_input":"2023-03-15T22:10:36.129875Z","iopub.status.idle":"2023-03-15T22:10:36.160276Z","shell.execute_reply.started":"2023-03-15T22:10:36.129838Z","shell.execute_reply":"2023-03-15T22:10:36.159298Z"}}
# Sensor Geometry Data
sensor_geometry_df = pd.read_csv(home_dir + "sensor_geometry.csv")

# X, Y, Z coordinates
sensor_x = sensor_geometry_df.x
sensor_y = sensor_geometry_df.y
sensor_z = sensor_geometry_df.z

# Detector constants
c_const = 0.299792458  # speed of light [m/ns]

# Min / Max information
x_min = sensor_x.min()
x_max = sensor_x.max()
y_min = sensor_y.min()
y_max = sensor_y.max()
z_min = sensor_z.min()
z_max = sensor_z.max()

# Detector Valid Length
detector_length = np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2 + (z_max - z_min) ** 2)
t_valid_length = detector_length / c_const


"""

## Single event reader function

- Pick-up important data points first
    - Rank 3 (First)
        - not aux, in valid time window
    - Rank 2
        - not aux, out of valid time window
    - Rank 1
        - aux, in valid time window
    - Rank 0 (Last)
        - aux, out of valid time window
    - In each ranks, take pulses from highest charge

"""


# read single event from batch_meta_df
def read_event(event_idx, batch_meta_df, max_pulse_count, batch_df, train=True):
    # read metadata
    batch_id, first_pulse_index, last_pulse_index = batch_meta_df.iloc[event_idx][
        ["batch_id", "first_pulse_index", "last_pulse_index"]].astype("int")

    # read event
    event_feature = batch_df[first_pulse_index:last_pulse_index + 1]
    sensor_id = event_feature.sensor_id

    # merge features into single structured array
    dtype = [("time", "float16"),
             ("charge", "float16"),
             ("auxiliary", "float16"),
             ("x", "float16"),
             ("y", "float16"),
             ("z", "float16"),
             ("rank", "short")]
    event_x = np.zeros(last_pulse_index - first_pulse_index + 1, dtype)

    event_x["time"] = event_feature.time.values - event_feature.time.min()
    event_x["charge"] = event_feature.charge.values
    event_x["auxiliary"] = event_feature.auxiliary.values

    event_x["r_err"] = sensor_geometry_df.r_err[sensor_id].values
    event_x["z_err"] = sensor_geometry_df.z_err[sensor_id].values

    event_x["x"] = sensor_geometry_df.x[sensor_id].values
    event_x["y"] = sensor_geometry_df.y[sensor_id].values
    event_x["z"] = sensor_geometry_df.z[sensor_id].values

    # For long event, pick-up
    if len(event_x) > max_pulse_count:
        # Find valid time window
        t_peak = event_x["time"][event_x["charge"].argmax()]
        t_valid_min = t_peak - t_valid_length
        t_valid_max = t_peak + t_valid_length

        t_valid = (event_x["time"] > t_valid_min) * (event_x["time"] < t_valid_max)

        # rank
        event_x["rank"] = 2 * (1 - event_x["auxiliary"]) + (t_valid)

        # sort by rank and charge (important goes to backward)
        event_x = np.sort(event_x, order=["rank", "charge"])

        # pick-up from backward
        event_x = event_x[-max_pulse_count:]

        # resort by time
        event_x = np.sort(event_x, order="time")

    # resort by time
    event_x = np.sort(event_x, order="time")

    # for train data, give angles together
    azimuth, zenith = batch_meta_df.iloc[event_idx][["azimuth", "zenith"]].astype("float16")
    event_y = np.array([azimuth, zenith], dtype="float16")

    return event_idx, len(event_x), event_x, event_y

# Read Train Meta Data
train_meta_df = pd.read_parquet(home_dir + 'train_meta.parquet')

batch_counts = train_meta_df.batch_id.value_counts().sort_index()

batch_max_index = batch_counts.cumsum()
batch_max_index[train_meta_df.batch_id.min() - 1] = 0
batch_max_index = batch_max_index.sort_index()


def train_meta_df_spliter(batch_id):
    return train_meta_df.loc[batch_max_index[batch_id - 1]:batch_max_index[batch_id] - 1]


for batch_id in train_batch_ids:
    print("Reading batch ", batch_id, end="")
    # get batch meta data and data
    batch_meta_df = train_meta_df_spliter(batch_id)
    batch_df = pd.read_parquet(train_format.format(batch_id=batch_id))

    # register pulses
    batch_x = np.zeros((len(batch_meta_df), max_pulse_count, n_features), dtype="float16")
    batch_y = np.zeros((len(batch_meta_df), 2), dtype="float16")

    batch_x[:, :, 2] = -1


    def read_event_local(event_idx):
        return read_event(event_idx, batch_meta_df, max_pulse_count, batch_df, train=True)

    # Proces Events
    iterator = range(len(batch_meta_df))
    event_idx, pulse_count, event_x, event_y  = read_event_local(1)
    batch_x[event_idx, :pulse_count, 0] = event_x["time"]
    batch_x[event_idx, :pulse_count, 1] = event_x["charge"]
    batch_x[event_idx, :pulse_count, 2] = event_x["auxiliary"]
    batch_x[event_idx, :pulse_count, 3] = event_x["x"]
    batch_x[event_idx, :pulse_count, 4] = event_x["y"]
    batch_x[event_idx, :pulse_count, 5] = event_x["z"]
    batch_x[event_idx, :pulse_count, 6] = event_x["r_err"]
    batch_x[event_idx, :pulse_count, 7] = event_x["z_err"]
    batch_x[event_idx, :pulse_count, 8] = event_x["rank"]
    batch_y[event_idx] = event_y

    del batch_meta_df, batch_df

    # Save
    print(" DONE! Saving...")
    np.savez(point_picker_format.format(batch_id=batch_id), x=batch_x, y=batch_y)