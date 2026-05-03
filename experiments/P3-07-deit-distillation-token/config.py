from dataclasses import dataclass


@dataclass
class Config:
    T: float = 4.0
    alpha: float = 0.5
    learning_rate: float = 5e-4
    teacher_epochs: int = 50
    student_epochs: int = 50
    batch_size: int = 128
    test_batch_size: int = 200
    dropout: float = 0.1
    data_dir: str = './data'
    image_size: int = 32
    patch_size: int = 4
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    num_classes: int = 100
    weight_decay: float = 5e-5
    lr_step_size: int = 20
    lr_gamma: float = 0.5
    warmup_epochs: int = 5
    seed: int = 42
