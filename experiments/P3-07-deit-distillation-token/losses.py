import torch
import torch.nn as nn
import torch.nn.functional as F


class DeiTDistillationLoss(nn.Module):
    def __init__(self, T=4.0, alpha=0.5):
        super().__init__()
        self.T = T
        self.alpha = alpha

    def forward(self, cls_out, dist_out, teacher_logits, labels):
        loss_hard = F.cross_entropy(cls_out, labels)
        dist_soft = F.log_softmax(dist_out / self.T, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.T, dim=1)
        loss_soft = F.kl_div(dist_soft, teacher_soft, reduction='batchmean') * (self.T ** 2)
        return (1 - self.alpha) * loss_hard + self.alpha * loss_soft
