from dataclasses import dataclass


@dataclass
class Config:
    teacher_name: str = 'bert-base-uncased'
    num_student_layers: int = 4
    student_hidden_size: int = 384
    student_num_attention_heads: int = 6
    student_intermediate_size: int = 1536
    teacher_hidden_size: int = 768
    teacher_num_layers: int = 12
    teacher_num_attention_heads: int = 12
    T: float = 4.0
    alpha_pred: float = 0.7
    alpha_attn: float = 0.1
    alpha_hidn: float = 0.1
    alpha_embd: float = 0.1
    max_seq_length: int = 64
    teacher_lr: float = 2e-5
    student_lr: float = 5e-5
    teacher_epochs: int = 3
    student_epochs: int = 10
    batch_size: int = 32
    weight_decay: float = 0.01
    seed: int = 42
    num_labels: int = 2
