import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedDistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.5, beta=0.1, gamma=1e3, at_p=2):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.at_p = at_p

    def _soft_loss(self, student_logits, teacher_logits):
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        return F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (self.temperature ** 2)

    def _hard_loss(self, student_logits, labels):
        return F.cross_entropy(student_logits, labels)

    def _feature_loss(self, student_features, teacher_features):
        loss = 0.0
        n_stages = 0
        for s_feat, t_feat in zip(student_features, teacher_features):
            if s_feat.shape[2:] != t_feat.shape[2:]:
                t_feat = F.adaptive_avg_pool2d(t_feat, s_feat.shape[2:])
            loss += F.l1_loss(s_feat, t_feat)
            n_stages += 1
        if n_stages > 0:
            loss = loss / n_stages
        return loss

    def _attention_loss(self, student_features, teacher_features):
        loss = 0.0
        n_stages = 0
        for s_feat, t_feat in zip(student_features, teacher_features):
            s_attn = (s_feat.abs() ** self.at_p).sum(dim=1, keepdim=True)
            t_attn = (t_feat.abs() ** self.at_p).sum(dim=1, keepdim=True)

            if s_attn.shape[2:] != t_attn.shape[2:]:
                t_attn = F.adaptive_avg_pool2d(t_attn, s_attn.shape[2:])

            s_attn = self._normalize(s_attn)
            t_attn = self._normalize(t_attn)

            loss += F.mse_loss(s_attn, t_attn)
            n_stages += 1

        if n_stages > 0:
            loss = loss / n_stages
        return loss

    @staticmethod
    def _normalize(attention):
        norm = attention.norm(p=2, dim=(2, 3), keepdim=True)
        norm = norm.clamp(min=1e-8)
        return attention / norm

    def forward(self, student_logits, teacher_logits, student_features, teacher_features, labels):
        l_soft = self._soft_loss(student_logits, teacher_logits)
        l_hard = self._hard_loss(student_logits, labels)
        l_feature = self._feature_loss(student_features, teacher_features)
        l_attention = self._attention_loss(student_features, teacher_features)

        total = self.alpha * l_soft + (1 - self.alpha) * l_hard + self.beta * l_feature + self.gamma * l_attention
        return total, l_soft, l_hard, l_feature, l_attention
