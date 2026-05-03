import torch
import torch.nn as nn
import torch.nn.functional as F


class BornAgainLoss(nn.Module):
    def __init__(self, temperature, lambda_kl):
        super().__init__()
        self.temperature = temperature
        self.lambda_kl = lambda_kl
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits, labels):
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)

        kl_loss = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)
        ce_loss = self.ce_loss(student_logits, labels)

        loss = ce_loss + self.lambda_kl * kl_loss
        return loss, kl_loss, ce_loss
