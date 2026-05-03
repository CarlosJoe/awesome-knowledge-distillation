from dataclasses import dataclass


@dataclass
class Config:
    T: float = 4.0
    alpha: float = 0.7
    learning_rate: float = 0.1
    teacher_epochs: int = 100
    student_epochs: int = 100
    batch_size: int = 128
    test_batch_size: int = 100
    dropout: float = 0.3
    stochastic_depth_p: float = 0.1
    num_iterations: int = 3
    data_dir: str = './data'
    weight_decay: float = 5e-4
    momentum: float = 0.9
    labeled_ratio: float = 0.5
    base_channels: int = 32
