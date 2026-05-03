import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    def __init__(self, T=4.0, alpha=0.7):
        super().__init__()
        self.T = T
        self.alpha = alpha

    def forward(self, student_logits, teacher_logits, labels):
        student_soft = F.log_softmax(student_logits / self.T, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.T, dim=1)
        loss_soft = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (self.T ** 2)
        loss_hard = F.cross_entropy(student_logits, labels)
        return self.alpha * loss_soft + (1 - self.alpha) * loss_hard
