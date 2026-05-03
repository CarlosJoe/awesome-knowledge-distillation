import torch
import torch.nn as nn
import torch.nn.functional as F


class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, x):
        abs_x = torch.abs(x)
        quadratic = torch.min(abs_x, torch.tensor(self.delta, device=x.device))
        linear = abs_x - quadratic
        return 0.5 * quadratic ** 2 + self.delta * linear


def pairwise_distance(features):
    eps = 1e-6
    diff = features.unsqueeze(1) - features.unsqueeze(0)
    dist = torch.sqrt(torch.sum(diff ** 2, dim=-1) + eps)
    mean_dist = dist.mean()
    norm_dist = dist / (mean_dist + eps)
    return norm_dist


def pairwise_angle(features):
    eps = 1e-6
    normed = F.normalize(features, p=2, dim=1)
    sim_matrix = torch.mm(normed, normed.t())
    sim_matrix = sim_matrix.clamp(-1.0 + eps, 1.0 - eps)
    angle_matrix = torch.acos(sim_matrix)
    return angle_matrix


class DistanceWiseRKD(nn.Module):
    def __init__(self):
        super().__init__()
        self.huber = HuberLoss(delta=1.0)

    def forward(self, student_features, teacher_features):
        student_dist = pairwise_distance(student_features)
        teacher_dist = pairwise_distance(teacher_features)
        loss = self.huber(student_dist - teacher_dist).mean()
        return loss


class AngleWiseRKD(nn.Module):
    def __init__(self):
        super().__init__()
        self.huber = HuberLoss(delta=1.0)

    def forward(self, student_features, teacher_features):
        student_angle = pairwise_angle(student_features)
        teacher_angle = pairwise_angle(teacher_features)
        loss = self.huber(student_angle - teacher_angle).mean()
        return loss


class RKDLoss(nn.Module):
    def __init__(self, temperature, alpha, w_dist, w_angle):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.w_dist = w_dist
        self.w_angle = w_angle
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.dist_loss = DistanceWiseRKD()
        self.angle_loss = AngleWiseRKD()

    def forward(self, student_logits, teacher_logits, labels, student_features, teacher_features):
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)

        kd_loss = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)
        ce_loss = self.ce_loss(student_logits, labels)

        dist_loss = self.dist_loss(student_features, teacher_features)
        angle_loss = self.angle_loss(student_features, teacher_features)

        loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss + self.w_dist * dist_loss + self.w_angle * angle_loss

        return loss, kd_loss, ce_loss, dist_loss, angle_loss
