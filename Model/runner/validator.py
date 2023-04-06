import torch
from ..utils import progress_bar
import numpy as np


class Validator:
    def __init__(self,
                 model,
                 loss,
                 device,
                 logger):
        self.model = model
        self.loss = loss
        self.device = device
        self.logger = logger

    def val(self, val_loader):
        self.model.eval()
        total_loss_val = 0
        score = 0
        with torch.no_grad():
            for index, sample_batched in enumerate(val_loader):
                sample_batched = sample_batched.to(self.device)
                outputs = self.model(sample_batched)
                label = sample_batched.y.reshape(-1, 2).to(self.device)
                loss = self.loss(outputs, label)
                total_loss_val += loss.cpu().item()
                # calculate score
                az_true = label[:, 0].cpu().numpy()
                zen_true = label[:, 1].cpu().numpy()
                az_pred = outputs[:, 0].cpu().numpy()
                zen_pred = outputs[:, 1].cpu().numpy()
                # if not (np.all(np.isfinite(az_true)) and
                #         np.all(np.isfinite(zen_true)) and
                #         np.all(np.isfinite(az_pred)) and
                #         np.all(np.isfinite(zen_pred))):
                #     raise ValueError("All arguments must be finite")
                sz1 = np.sin(zen_true)
                cz1 = np.cos(zen_true)
                sz2 = np.sin(zen_pred)
                cz2 = np.cos(zen_pred)

                scalar_prod = np.clip(sz1 * sz2 * (np.cos(az_true - az_pred)) + (cz1 * cz2), -1, 1)
                score += np.average(np.abs(np.arccos(scalar_prod)))
        self.logger.print_evaluate_information(total_loss_val / len(val_loader), score / len(val_loader))


class ValidatorLSTM:
    def __init__(self,
                 model,
                 loss,
                 device,
                 logger):
        self.model = model
        self.loss = loss
        self.device = device
        self.logger = logger
        self.angle_bin_vector = self.caculate_angle_bin_vector(bin_num=16)
        a = 1

    def caculate_angle_bin_vector(self, bin_num=16):
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

    def pred_to_angle(self, pred, epsilon=1e-8, bin_num=16):
        # convert prediction to vector
        pred_vector = (pred.reshape((-1, bin_num * bin_num, 1)) * self.angle_bin_vector).sum(axis=1)

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

    def val(self, val_loader):
        self.model.eval()
        total_loss_val = 0
        score = 0
        with torch.no_grad():
            for index, (x, y, true_label) in enumerate(val_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                y = y.to(torch.int64)
                outputs = self.model(x)
                loss = self.loss(outputs, y)
                total_loss_val += loss.cpu().item()
                # calculate score
                az_true = true_label[:, 0].cpu().numpy()
                zen_true = true_label[:, 1].cpu().numpy()
                az_pred, zen_pred = self.pred_to_angle(outputs.cpu().numpy())
                sz1 = np.sin(zen_true)
                cz1 = np.cos(zen_true)
                sz2 = np.sin(zen_pred)
                cz2 = np.cos(zen_pred)
                scalar_prod = np.clip(sz1 * sz2 * (np.cos(az_true - az_pred)) + (cz1 * cz2), -1, 1)
                score += np.average(np.abs(np.arccos(scalar_prod)))
        self.logger.print_evaluate_information(total_loss_val / len(val_loader), score / len(val_loader))
