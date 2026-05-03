import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginReLU(nn.Module):
    def __init__(self, margin=-0.2):
        super().__init__()
        self.margin = margin

    def forward(self, x):
        return torch.clamp(x + self.margin, min=0)


class OverhaulDistillationLoss(nn.Module):
    def __init__(self, temperature, alpha, beta, margin):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.margin_relu = MarginReLU(margin)
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def _channel_attention(self, teacher_feat):
        mean_c = teacher_feat.mean(dim=[2, 3], keepdim=True)
        std_c = teacher_feat.std(dim=[2, 3], keepdim=True)
        return mean_c + std_c

    def _align_features(self, student_feat, teacher_feat):
        if student_feat.shape[2:] != teacher_feat.shape[2:]:
            student_feat = F.adaptive_avg_pool2d(student_feat, teacher_feat.shape[2:])
        if student_feat.shape[1] != teacher_feat.shape[1]:
            student_feat = F.adaptive_avg_pool2d(student_feat, (teacher_feat.shape[2], teacher_feat.shape[3]))
            align_conv = nn.Conv2d(
                student_feat.shape[1], teacher_feat.shape[1],
                kernel_size=1, bias=False
            ).to(student_feat.device)
            student_feat = align_conv(student_feat)
        return student_feat

    def _feature_loss(self, student_features, teacher_features):
        total_loss = 0.0
        num_stages = len(teacher_features)

        for c in range(num_stages):
            s_feat = student_features[c]
            t_feat = teacher_features[c]

            s_feat = self._align_features(s_feat, t_feat)

            t_feat_transformed = self.margin_relu(t_feat)

            w_c = self._channel_attention(t_feat)

            diff = torch.abs(t_feat_transformed - s_feat)
            weighted_diff = w_c * diff
            num_elements = s_feat.shape[1] * s_feat.shape[2] * s_feat.shape[3]
            stage_loss = weighted_diff.sum(dim=[1, 2, 3]).mean() / num_elements
            total_loss += stage_loss

        return total_loss / num_stages

    def forward(self, student_logits, teacher_logits, labels, student_features, teacher_features):
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)

        kd_loss = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)
        ce_loss = self.ce_loss(student_logits, labels)
        feat_loss = self._feature_loss(student_features, teacher_features)

        total_loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss + self.beta * feat_loss

        return total_loss, kd_loss, ce_loss, feat_loss
