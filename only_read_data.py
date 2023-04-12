import numpy as np

data = np.load("data/LSTM_data/pointpicker_mpc128_n9_batch_101.npz")

array_list = list(data.keys())
data_x = data['x']