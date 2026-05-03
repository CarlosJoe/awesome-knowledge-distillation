import torch
import torch.nn as nn
import torch.nn.functional as F


class PseudoLabelLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha

    def forward(self, logits, targets, is_labeled):
        labeled_mask = is_labeled
        unlabeled_mask = ~is_labeled
        loss = torch.tensor(0.0, device=logits.device)
        if labeled_mask.any():
            loss = loss + self.alpha * F.cross_entropy(logits[labeled_mask], targets[labeled_mask])
        if unlabeled_mask.any():
            loss = loss + (1 - self.alpha) * F.cross_entropy(logits[unlabeled_mask], targets[unlabeled_mask])
        return loss
