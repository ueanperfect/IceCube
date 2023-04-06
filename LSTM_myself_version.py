import numpy as np

import time
import os
import gc

# Graphic
import matplotlib.pyplot as plt
from tqdm import tqdm

icecube_dir = "data/icecube-neutrinos-in-deep-ice"
model_dir = "checkpoints/LSTM"
data_dir = "data/LSTM_data"

bin_num = 16

train_batch_id_min = 412
train_batch_id_max = 415

train_batch_ids = [412, 413, 414, 415]  # range(train_batch_id_min, train_batch_id_max + 1)

# model
LSTM_width = 160
DENSE_width = 0

# training
validation_split = 0.05
seed = 220242
epochs = 20
batch_size = 128
fit_verbose = 1

# %% [code] {"execution":{"iopub.status.busy":"2023-02-01T16:16:05.292604Z","iopub.execute_input":"2023-02-01T16:16:05.293176Z","iopub.status.idle":"2023-02-01T16:16:05.297653Z","shell.execute_reply.started":"2023-02-01T16:16:05.293143Z","shell.execute_reply":"2023-02-01T16:16:05.29666Z"}}
# data preprocessing
point_picker_format = data_dir + '/pointpicker_mpc128_n9_batch_{batch_id:d}.npz'

# model
model_output_path = "checkpoints/LSTM_output/" + "PointPicker_mpc128bin16_LSTM160DENSE0"

def angular_dist_score(az_true, zen_true, az_pred, zen_pred):
    '''
    calculate the MAE of the angular distance between two directions.
    The two vectors are first converted to cartesian unit vectors,
    and then their scalar product is computed, which is equal to
    the cosine of the angle between the two vectors. The inverse
    cosine (arccos) thereof is then the angle between the two input vectors

    Parameters:
    -----------

    az_true : float (or array thereof)
        true azimuth value(s) in radian
    zen_true : float (or array thereof)
        true zenith value(s) in radian
    az_pred : float (or array thereof)
        predicted azimuth value(s) in radian
    zen_pred : float (or array thereof)
        predicted zenith value(s) in radian

    Returns:
    --------

    dist : float
        mean over the angular distance(s) in radian
    '''

    if not (np.all(np.isfinite(az_true)) and
            np.all(np.isfinite(zen_true)) and
            np.all(np.isfinite(az_pred)) and
            np.all(np.isfinite(zen_pred))):
        raise ValueError("All arguments must be finite")

    # pre-compute all sine and cosine values
    sa1 = np.sin(az_true)
    ca1 = np.cos(az_true)
    sz1 = np.sin(zen_true)
    cz1 = np.cos(zen_true)

    sa2 = np.sin(az_pred)
    ca2 = np.cos(az_pred)
    sz2 = np.sin(zen_pred)
    cz2 = np.cos(zen_pred)

    # scalar product of the two cartesian vectors (x = sz*ca, y = sz*sa, z = cz)
    scalar_prod = sz1 * sz2 * (ca1 * ca2 + sa1 * sa2) + (cz1 * cz2)

    # scalar product of two unit vectors is always between -1 and 1, this is against nummerical instability
    # that might otherwise occure from the finite precision of the sine and cosine functions
    scalar_prod = np.clip(scalar_prod, -1, 1)

    # convert back to an angle (in radian)
    return np.average(np.abs(np.arccos(scalar_prod)))

# %% [code] {"execution":{"iopub.status.busy":"2023-02-01T16:16:05.369059Z","iopub.execute_input":"2023-02-01T16:16:05.36991Z","iopub.status.idle":"2023-02-01T16:16:05.37853Z","shell.execute_reply.started":"2023-02-01T16:16:05.369876Z","shell.execute_reply":"2023-02-01T16:16:05.377405Z"}}
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


# %% [code] {"execution":{"iopub.status.busy":"2023-02-01T16:16:05.46674Z","iopub.execute_input":"2023-02-01T16:16:05.467006Z","iopub.status.idle":"2023-02-01T16:16:05.473244Z","shell.execute_reply.started":"2023-02-01T16:16:05.466982Z","shell.execute_reply":"2023-02-01T16:16:05.472309Z"}}
def y_to_onehot(batch_y):
    # evaluate bin code
    azimuth_code = (batch_y[:, 0] > azimuth_edges[1:].reshape((-1, 1))).sum(axis=0)
    zenith_code = (batch_y[:, 1] > zenith_edges[1:].reshape((-1, 1))).sum(axis=0)
    angle_code = bin_num * azimuth_code + zenith_code

    # one-hot
    batch_y_onehot = np.zeros((angle_code.size, bin_num * bin_num))
    batch_y_onehot[np.arange(angle_code.size), angle_code] = 1

    return batch_y_onehot


# %% [code] {"execution":{"iopub.status.busy":"2023-02-01T16:16:05.494831Z","iopub.execute_input":"2023-02-01T16:16:05.496866Z","iopub.status.idle":"2023-02-01T16:16:05.506008Z","shell.execute_reply.started":"2023-02-01T16:16:05.496821Z","shell.execute_reply":"2023-02-01T16:16:05.505022Z"}}
angle_bin_zenith0 = np.tile(zenith_edges[:-1], bin_num)
angle_bin_zenith1 = np.tile(zenith_edges[1:], bin_num)
angle_bin_azimuth0 = np.repeat(azimuth_edges[:-1], bin_num)
angle_bin_azimuth1 = np.repeat(azimuth_edges[1:], bin_num)

angle_bin_area = (angle_bin_azimuth1 - angle_bin_azimuth0) * (np.cos(angle_bin_zenith0) - np.cos(angle_bin_zenith1))
angle_bin_vector_sum_x = (np.sin(angle_bin_azimuth1) - np.sin(angle_bin_azimuth0)) * ((angle_bin_zenith1 - angle_bin_zenith0) / 2 - (np.sin(2 * angle_bin_zenith1) - np.sin(2 * angle_bin_zenith0)) / 4)
angle_bin_vector_sum_y = (np.cos(angle_bin_azimuth0) - np.cos(angle_bin_azimuth1)) * ((angle_bin_zenith1 - angle_bin_zenith0) / 2 - (np.sin(2 * angle_bin_zenith1) - np.sin(2 * angle_bin_zenith0)) / 4)
angle_bin_vector_sum_z = (angle_bin_azimuth1 - angle_bin_azimuth0) * ((np.cos(2 * angle_bin_zenith0) - np.cos(2 * angle_bin_zenith1)) / 4)

angle_bin_vector_mean_x = angle_bin_vector_sum_x / angle_bin_area
angle_bin_vector_mean_y = angle_bin_vector_sum_y / angle_bin_area
angle_bin_vector_mean_z = angle_bin_vector_sum_z / angle_bin_area

angle_bin_vector = np.zeros((1, bin_num * bin_num, 3))
angle_bin_vector[:, :, 0] = angle_bin_vector_mean_x
angle_bin_vector[:, :, 1] = angle_bin_vector_mean_y
angle_bin_vector[:, :, 2] = angle_bin_vector_mean_z


def pred_to_angle(pred, epsilon=1e-8):
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

print("Reading training data...")


for batch_id in tqdm(train_batch_ids):
    train_data_file = np.load(point_picker_format.format(batch_id=batch_id))

    if train_x is None:
        train_x = train_data_file["x"]
        train_y = train_data_file["y"]
    else:
        train_x = np.append(train_x, train_data_file["x"], axis=0)
        train_y = np.append(train_y, train_data_file["y"], axis=0)

    train_data_file.close()
    del train_data_file
    _ = gc.collect()

train_x[:, :, 0] /= 1000  # time
train_x[:, :, 1] /= 300  # charge
train_x[:, :, 3:] /= 600  # space

train_y_onehot = y_to_onehot(train_y)

num_valid = int(validation_split * len(train_x))

valid_x = train_x[-num_valid:]
valid_y = train_y[-num_valid:]
valid_y_onehot = train_y_onehot[-num_valid:]

train_x = train_x[:-num_valid]
train_y = train_y[:-num_valid]
train_y_onehot = train_y_onehot[:-num_valid]