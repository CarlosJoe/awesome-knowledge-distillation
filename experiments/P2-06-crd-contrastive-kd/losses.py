import torch
import torch.nn as nn
import torch.nn.functional as F


class CRDLoss(nn.Module):
    def __init__(self, temperature, alpha, crd_temp):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.crd_temp = crd_temp
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def _infonce_loss(self, student_proj, teacher_proj):
        student_proj = F.normalize(student_proj, dim=1)
        teacher_proj = F.normalize(teacher_proj, dim=1)

        sim_matrix = torch.matmul(student_proj, teacher_proj.t()) / self.crd_temp

        batch_size = sim_matrix.size(0)
        labels = torch.arange(batch_size, device=sim_matrix.device)

        loss = F.cross_entropy(sim_matrix, labels)
        return loss

    def forward(self, student_logits, teacher_logits, student_proj, teacher_proj, labels):
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)

        kd_loss = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)
        ce_loss = self.ce_loss(student_logits, labels)
        crd_loss = self._infonce_loss(student_proj, teacher_proj)

        total_loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss + crd_loss
        return total_loss, kd_loss, ce_loss, crd_loss
