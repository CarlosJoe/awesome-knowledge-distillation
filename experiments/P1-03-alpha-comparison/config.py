from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    T: float = 4.0
    alpha_values: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7, 0.9, 1.0])
    learning_rate: float = 1e-3
    teacher_epochs: int = 20
    student_epochs: int = 10
    batch_size: int = 64
    test_batch_size: int = 1000
    dropout: float = 0.2
    data_dir: str = './data'
    mnist_mean: float = 0.1307
    mnist_std: float = 0.3081
    lr_step_size: int = 10
    lr_gamma: float = 0.5
    seed: int = 42
    results_dir: str = './results'
