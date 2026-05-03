import torch
import torch.nn as nn
import torch.nn.functional as F


class BNStatisticsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn_stats = {}

    def _hook(self, name):
        def hook(module, input, output):
            mean = input[0].mean(dim=[0, 2, 3])
            var = input[0].var(dim=[0, 2, 3], unbiased=False)
            self.bn_stats[name] = (mean, var)
        return hook

    def register_hooks(self, model):
        self.handles = []
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                handle = module.register_forward_hook(self._hook(name))
                self.handles.append(handle)

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def forward(self, model):
        loss = 0.0
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                if name in self.bn_stats and module.running_mean is not None and module.running_var is not None:
                    batch_mean, batch_var = self.bn_stats[name]
                    loss += (module.running_mean.detach() - batch_mean).pow(2).sum()
                    loss += (module.running_var.detach() - batch_var).pow(2).sum()
        return loss


class TotalVariationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        diff_h = x[:, :, 1:, :] - x[:, :, :-1, :]
        diff_w = x[:, :, :, 1:] - x[:, :, :, :-1]
        return diff_h.pow(2).sum() + diff_w.pow(2).sum()


class L2Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.pow(2).sum()


class DeepInversionLoss(nn.Module):
    def __init__(self, bn_weight, tv_weight, l2_weight):
        super().__init__()
        self.bn_weight = bn_weight
        self.tv_weight = tv_weight
        self.l2_weight = l2_weight
        self.bn_loss = BNStatisticsLoss()
        self.tv_loss = TotalVariationLoss()
        self.l2_loss = L2Loss()

    def register_hooks(self, model):
        self.bn_loss.register_hooks(model)

    def remove_hooks(self):
        self.bn_loss.remove_hooks()

    def forward(self, logits, targets, images, model):
        ce_loss = F.cross_entropy(logits, targets)
        bn_loss = self.bn_loss(model)
        tv_loss = self.tv_loss(images)
        l2_loss = self.l2_loss(images)
        total = ce_loss + self.bn_weight * bn_loss + self.tv_weight * tv_loss + self.l2_weight * l2_loss
        return total, ce_loss, bn_loss, tv_loss, l2_loss


class KDLoss(nn.Module):
    def __init__(self, temperature, alpha):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_logits, teacher_logits, targets):
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1),
            reduction="batchmean",
        ) * (self.temperature ** 2)
        hard_loss = F.cross_entropy(student_logits, targets)
        return self.alpha * soft_loss + (1.0 - self.alpha) * hard_loss
