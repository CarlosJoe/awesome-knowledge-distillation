import torch.nn as nn
from transformers import BertForSequenceClassification, BertConfig

from config import Config


def create_teacher_model(config: Config = Config()):
    model = BertForSequenceClassification.from_pretrained(
        config.teacher_name,
        num_labels=config.num_labels,
        attn_implementation='eager',
    )
    return model


def create_student_model(config: Config = Config()):
    student_config = BertConfig(
        hidden_size=config.student_hidden_size,
        num_hidden_layers=config.num_student_layers,
        num_attention_heads=config.student_num_attention_heads,
        intermediate_size=config.student_intermediate_size,
        num_labels=config.num_labels,
        attn_implementation='eager',
    )
    model = BertForSequenceClassification(student_config)
    return model


def create_projection_layers(config: Config = Config()):
    W_e = nn.Linear(config.student_hidden_size, config.teacher_hidden_size, bias=False)
    W_h = nn.Linear(config.student_hidden_size, config.teacher_hidden_size, bias=False)
    return W_e, W_h
