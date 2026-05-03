import torch
import torch.nn as nn
import torch.nn.functional as F


def wasserstein_distance(p, q):
    p_sorted = torch.sort(p, dim=1)[0]
    q_sorted = torch.sort(q, dim=1)[0]
    return (p_sorted - q_sorted).abs().mean(dim=1)


class WassersteinKDLoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        p_student = F.softmax(student_logits, dim=1)
        p_teacher = F.softmax(teacher_logits, dim=1)

        w_dist = wasserstein_distance(p_student, p_teacher).mean()
        ce_loss = self.ce_loss(student_logits, labels)

        loss = self.alpha * w_dist + (1 - self.alpha) * ce_loss
        return loss, w_dist, ce_loss


class KLDistillationLoss(nn.Module):
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
