import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitStandardizedKD(nn.Module):
    def __init__(self, temperature, alpha):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    @staticmethod
    def standardize(logits):
        mean = logits.mean(dim=1, keepdim=True)
        std = logits.std(dim=1, keepdim=True)
        return (logits - mean) / (std + 1e-6)

    def forward(self, student_logits, teacher_logits, labels):
        z_student = self.standardize(student_logits)
        z_teacher = self.standardize(teacher_logits)

        soft_student = F.log_softmax(z_student / self.temperature, dim=1)
        soft_teacher = F.softmax(z_teacher / self.temperature, dim=1)

        kd_loss = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)
        ce_loss = self.ce_loss(student_logits, labels)

        loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss
        return loss, kd_loss, ce_loss


class DistillationLoss(nn.Module):
    def __init__(self, temperature, alpha):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits, labels):
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)

        kd_loss = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)
        ce_loss = self.ce_loss(student_logits, labels)

        loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss
        return loss, kd_loss, ce_loss
