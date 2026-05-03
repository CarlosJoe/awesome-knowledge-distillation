import torch
import torch.nn as nn


class DiffusionMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, noise_pred, noise_target):
        return self.mse(noise_pred, noise_target)


class DistillMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, x_student, x_teacher):
        return self.mse(x_student, x_teacher)
