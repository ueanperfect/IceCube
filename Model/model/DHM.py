import torch.nn as nn
import torch
import math
import numpy as np


class DHMNeck(nn.Module):
    def __init__(self, d_model, sequence_length, neck_output):
        super(DHMNeck, self).__init__()
        self.mlp = nn.Linear(d_model * sequence_length, neck_output)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.mlp(x)
        x = self.relu(x)
        return x


class DHMClassesHead(nn.Module):
    def __init__(self, neck_output, hidden_size, output_class):
        super(DHMClassesHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(neck_output, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.classifier = nn.Linear(hidden_size, output_class)

    def forward(self, x):
        x = self.mlp(x)
        x = self.classifier(x)
        return x


class DHMRegressionHead(nn.Module):
    def __init__(self, neck_output, hidden_size):
        super(DHMRegressionHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(neck_output, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.Regression = nn.Linear(hidden_size, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.mlp(x)
        x = self.Regression(x)
        x = self.sigmoid(x)
        return x


class DHM(nn.Module):
    def __init__(self, transformer_encoder_layer_nums, transformer_encoder, neck_head, cls_head, regression_head):
        super(DHM, self).__init__()
        # self.pos_encoder = PositionalEncoding(d_model)
        self.transformer_encoder = nn.ModuleList(
            [nn.TransformerEncoderLayer(**transformer_encoder) for i in range(transformer_encoder_layer_nums)])
        self.neck_head = DHMNeck(**neck_head)
        self.cls_head = DHMClassesHead(**cls_head)
        self.regression_head = DHMRegressionHead(**regression_head)
        self.bin_number = int(math.sqrt(cls_head['output_class']))
        self.angle_bin_vector = self.caculate_angle_bin_vector(bin_num=self.bin_number).float().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    def caculate_angle_bin_vector(self, bin_num=12):
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
        angle_bin_vector = torch.from_numpy(angle_bin_vector)
        return angle_bin_vector

    def pred_to_angle(self, pred, epsilon=1e-8, bin_num=12):
        # convert prediction to vector
        pred_vector = (pred.reshape((-1, bin_num * bin_num, 1)) * self.angle_bin_vector).sum(axis=1)

        # normalize
        pred_vector_norm = torch.sqrt((pred_vector ** 2).sum(axis=1))
        mask = pred_vector_norm < epsilon
        pred_vector_norm[mask] = 1

        # assign <1, 0, 0> to very small vectors (badly predicted)
        pred_vector /= pred_vector_norm.reshape((-1, 1))
        pred_vector[mask] = torch.Tensor([1., 0., 0.]).to(torch.float32).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # convert to angle
        azimuth = torch.arctan2(pred_vector[:, 1], pred_vector[:, 0])
        azimuth[azimuth < 0] += 2 * torch.pi
        zenith = torch.arccos(pred_vector[:, 2])

        return azimuth, zenith

    def forward(self, src):
        # mask = torch.sum(src, dim=2) == 0
        for layer in self.transformer_encoder:
            src = layer(src)
        re = self.neck_head(src)
        class_result = self.cls_head(re)
        # regression_result = self.regression_head(re)
        # az_pred, zen_pred = self.pred_to_angle(class_result,bin_num=self.bin_number)
        # az_pred = regression_result[:, 0] * 2 * torch.pi
        # zen_pred = regression_result[:, 1] * 2 * torch.pi
        return class_result
