import pandas as pd
from pathlib import Path
from ..dataset import IceCubeDataset
from ..utils import prepare_sensors, progress_bar
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import torch


class Trainner:
    def __init__(self,
                 model,
                 loss,
                 optimzer,
                 device,
                 logger):
        self.model = model
        self.loss = loss
        self.device = device
        self.opt = optimzer
        self.model.to(device)
        self.loss.to(device)
        self.logger = logger

    def train(self, dataloader):
        iter_number = len(dataloader)
        record_number = int(iter_number / 5)

        for index, sample_batched in enumerate(dataloader):
            self.opt.zero_grad()
            sample_batched = sample_batched.to(self.device)
            outputs = self.model(sample_batched)
            label = sample_batched.y.reshape(-1, 2).to(self.device)
            loss = self.loss(label, outputs)
            loss.backward()
            self.opt.step()
            if index % record_number == 0:
                self.logger.print_training_information(loss.cpu().item(), index, len(dataloader))


class TrainnerLSTM:
    def __init__(self,
                 model,
                 loss,
                 optimzer,
                 device,
                 logger):
        self.model = model
        self.loss = loss
        self.device = device
        self.opt = optimzer
        self.model.to(device)
        self.loss.to(device)
        self.logger = logger

    def accuracy(self, y_pred, y_true):
        """
        计算模型的准确率
        y_pred: 模型预测的结果，大小为[N, C]，其中N为样本数，C为类别数
        y_true: 样本的真实标签，大小为[N]
        """
        # 将预测结果转换为类别id
        y_pred = torch.argmax(y_pred, dim=1)
        # 计算准确率
        correct = (y_pred == y_true).sum().item()
        total = y_true.size(0)
        acc = correct / total
        return acc

    def train(self, dataloader):
        iter_number = len(dataloader)
        record_number = int(iter_number / 20)
        for index, (x, y, true_label) in enumerate(dataloader):
            self.opt.zero_grad()
            x = x.to(self.device)
            y = y.to(self.device)
            y = y.to(torch.int64)
            outputs = self.model(x)
            loss = self.loss(outputs, y)
            loss.backward()
            self.opt.step()
            if index % record_number == 0:
                acc = self.accuracy(outputs, y)
                print(f"acc : {acc:.4f}")
                self.logger.print_training_information(loss.cpu().item(), index, len(dataloader))


class TrainnerDHM:
    def __init__(self,
                 model,
                 loss,
                 optimzer,
                 device,
                 logger):
        self.model = model
        self.loss = torch.nn.MSELoss()
        self.device = device
        self.opt = optimzer
        self.model.to(device)
        self.loss.to(device)
        self.logger = logger

    def accuracy(self, y_pred, y_true):
        """
        计算模型的准确率
        y_pred: 模型预测的结果，大小为[N, C]，其中N为样本数，C为类别数
        y_true: 样本的真实标签，大小为[N]
        """
        # 将预测结果转换为类别id
        y_pred = torch.argmax(y_pred, dim=1)
        # 计算准确率
        correct = (y_pred == y_true).sum().item()
        total = y_true.size(0)
        acc = correct / total
        return acc

    def train(self, dataloader):
        iter_number = len(dataloader)
        record_number = int(iter_number / 20)
        for index, (x, y, true_label) in enumerate(dataloader):
            self.opt.zero_grad()
            x = x.to(self.device)
            y = y.to(self.device)
            # y = y.to(torch.int64)
            regression_result= self.model(x)
            # az_true = true_label[:, 0].to(self.device)
            # zen_true = true_label[:, 1].to(self.device)
            true_label = true_label.to(self.device).to(torch.float32)
            loss = self.loss(regression_result,true_label)
            loss.backward()
            self.opt.step()
            if index % record_number == 0:
                # acc = self.accuracy(class_result, y)
                # print(f"acc : {acc:.4f}")
                self.logger.print_training_information(loss.cpu().item(), index, len(dataloader))
