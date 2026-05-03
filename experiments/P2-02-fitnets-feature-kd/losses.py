import torch
import torch.nn as nn
import torch.nn.functional as F


class HintLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, student_hint, teacher_hint):
        return self.mse(student_hint, teacher_hint)


class FitNetsLoss(nn.Module):
    def __init__(self, temperature, alpha, beta):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.hint_loss = HintLoss()

    def forward(self, student_logits, teacher_logits, labels, student_hint=None, teacher_hint=None, regressor_hint=None):
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)

        kd_loss = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)
        ce_loss = self.ce_loss(student_logits, labels)

        loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss

        hint = torch.tensor(0.0, device=student_logits.device)
        if student_hint is not None and teacher_hint is not None and regressor_hint is not None:
            hint = self.hint_loss(regressor_hint, teacher_hint)
            loss = loss + self.beta * hint

        return loss, kd_loss, ce_loss, hint
