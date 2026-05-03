import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class TinyBERTDistiller(nn.Module):
    def __init__(self, config: Config = Config()):
        super().__init__()
        self.T = config.T
        self.alpha_pred = config.alpha_pred
        self.alpha_attn = config.alpha_attn
        self.alpha_hidn = config.alpha_hidn
        self.alpha_embd = config.alpha_embd
        self.teacher_heads = config.teacher_num_attention_heads
        self.student_heads = config.student_num_attention_heads
        self.head_ratio = self.teacher_heads // self.student_heads

    def distillation_loss(self, student_logits, teacher_logits, labels):
        student_soft = F.log_softmax(student_logits / self.T, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.T, dim=1)
        loss_kl = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (self.T ** 2)
        loss_ce = F.cross_entropy(student_logits, labels)
        return loss_kl + loss_ce

    def attention_loss(self, student_attentions, teacher_attentions, layer_map):
        loss = 0.0
        for s_layer, t_layer in layer_map:
            s_attn = student_attentions[s_layer]
            t_attn = teacher_attentions[t_layer]
            batch = t_attn.size(0)
            seq_len = t_attn.size(2)
            t_attn_reshaped = t_attn.view(
                batch, self.student_heads, self.head_ratio, seq_len, seq_len
            )
            t_attn_avg = t_attn_reshaped.mean(dim=2)
            loss += F.mse_loss(s_attn, t_attn_avg)
        return loss / len(layer_map)

    def hidden_loss(self, student_hidden_states, teacher_hidden_states, W_h, layer_map):
        loss = 0.0
        for s_layer, t_layer in layer_map:
            s_hidden = student_hidden_states[s_layer + 1]
            t_hidden = teacher_hidden_states[t_layer + 1]
            s_hidden_proj = W_h(s_hidden)
            loss += F.mse_loss(s_hidden_proj, t_hidden)
        return loss / len(layer_map)

    def embedding_loss(self, student_hidden_states, teacher_hidden_states, W_e):
        s_embd = student_hidden_states[0]
        t_embd = teacher_hidden_states[0]
        s_embd_proj = W_e(s_embd)
        return F.mse_loss(s_embd_proj, t_embd)

    def compute_loss(self, student_outputs, teacher_outputs, labels, W_e, W_h, layer_map):
        loss_pred = self.distillation_loss(student_outputs.logits, teacher_outputs.logits, labels)
        loss_attn = self.attention_loss(student_outputs.attentions, teacher_outputs.attentions, layer_map)
        loss_hidn = self.hidden_loss(student_outputs.hidden_states, teacher_outputs.hidden_states, W_h, layer_map)
        loss_embd = self.embedding_loss(student_outputs.hidden_states, teacher_outputs.hidden_states, W_e)
        total_loss = (
            self.alpha_pred * loss_pred
            + self.alpha_attn * loss_attn
            + self.alpha_hidn * loss_hidn
            + self.alpha_embd * loss_embd
        )
        return total_loss
