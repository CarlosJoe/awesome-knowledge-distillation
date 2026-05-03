import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionTransferLoss(nn.Module):
    def __init__(self, p=2):
        super().__init__()
        self.p = p

    def attention_map(self, feature):
        return (feature.abs() ** self.p).sum(dim=1, keepdim=True)

    def normalize(self, attention):
        norm = attention.norm(p=2, dim=(2, 3), keepdim=True)
        norm = norm.clamp(min=1e-8)
        return attention / norm

    def forward(self, student_features, teacher_features):
        loss = 0.0
        n_layers = 0
        layer_names = sorted(set(student_features.keys()) & set(teacher_features.keys()))

        for name in layer_names:
            s_feat = student_features[name]
            t_feat = teacher_features[name]

            s_attn = self.attention_map(s_feat)
            t_attn = self.attention_map(t_feat)

            if s_attn.shape[2:] != t_attn.shape[2:]:
                t_attn = F.adaptive_avg_pool2d(t_attn, s_attn.shape[2:])

            s_attn = self.normalize(s_attn)
            t_attn = self.normalize(t_attn)

            loss += F.mse_loss(s_attn, t_attn)
            n_layers += 1

        if n_layers > 0:
            loss = loss / n_layers

        return loss


class ATDistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.7, at_p=2, at_beta=1e3):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.at_beta = at_beta
        self.at_loss_fn = AttentionTransferLoss(p=at_p)

    def forward(self, student_logits, teacher_logits, student_features, teacher_features, labels):
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1),
            reduction="batchmean",
        ) * (self.temperature ** 2)

        hard_loss = F.cross_entropy(student_logits, labels)

        at_loss = self.at_loss_fn(student_features, teacher_features)

        total = self.alpha * soft_loss + (1 - self.alpha) * hard_loss + self.at_beta * at_loss

        return total, soft_loss, hard_loss, at_loss
