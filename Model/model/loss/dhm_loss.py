import torch.nn as nn
import torch


class DHMLoss(nn.Module):
    def __init__(self, lambda1, lambda2, lambda3):
        super(DHMLoss, self).__init__()
        self.lambda1 = nn.Parameter(torch.Tensor([lambda1]))
        self.lambda2 = nn.Parameter(torch.Tensor([lambda2]))
        self.lambda3 = nn.Parameter(torch.Tensor([lambda3]))
        self.class_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()

    def forward(self, az_true, zen_true, az_pred, zen_pred, pre_class, true_class, regression_result):
        loss1 = self.class_loss(pre_class, true_class)
        loss2_1 = self.regression_loss(az_true, az_pred)
        loss2_2 = self.regression_loss(zen_true, zen_pred)
        loss3 = self.regression_loss(regression_result,torch.zeros_like(regression_result, device=regression_result.device))
        return loss1 * self.lambda1[0] + self.lambda2[0] * (loss2_1 + loss2_2) + self.lambda3[0] * loss3
