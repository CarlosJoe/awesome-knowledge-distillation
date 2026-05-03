import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfDistillationLoss(nn.Module):
    def __init__(self, T=4.0, alpha=0.5, beta=0.3, gamma=0.2):
        super().__init__()
        self.T = T
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, main_out, aux1_out, aux2_out, labels):
        loss_main = F.cross_entropy(main_out, labels)

        loss_aux1_ce = F.cross_entropy(aux1_out, labels)
        main_soft_detached = F.softmax(main_out.detach() / self.T, dim=1)
        aux1_soft = F.log_softmax(aux1_out / self.T, dim=1)
        loss_aux1_kd = F.kl_div(aux1_soft, main_soft_detached, reduction='batchmean') * (self.T ** 2)
        loss_aux1 = loss_aux1_ce + loss_aux1_kd

        loss_aux2_ce = F.cross_entropy(aux2_out, labels)
        aux2_soft = F.log_softmax(aux2_out / self.T, dim=1)
        loss_aux2_kd = F.kl_div(aux2_soft, main_soft_detached, reduction='batchmean') * (self.T ** 2)
        loss_aux2 = loss_aux2_ce + loss_aux2_kd

        total_loss = self.alpha * loss_main + self.beta * loss_aux1 + self.gamma * loss_aux2
        return total_loss
