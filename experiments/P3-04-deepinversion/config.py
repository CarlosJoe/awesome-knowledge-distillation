from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DeepInversionConfig:
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 4

    dataset: str = "CIFAR10"
    num_classes: int = 10
    img_size: Tuple[int, int, int] = (3, 32, 32)
    imagenet_normalize: bool = False

    teacher_epochs: int = 200
    teacher_lr: float = 0.1
    teacher_weight_decay: float = 5e-4
    teacher_momentum: float = 0.9

    inv_iterations: int = 2000
    inv_lr: float = 0.1
    bn_weight: float = 5.0
    tv_weight: float = 1e-4
    l2_weight: float = 1e-3
    images_per_class: int = 50
    inv_batch_size: int = 25

    student_epochs: int = 200
    student_lr: float = 0.1
    student_weight_decay: float = 5e-4
    student_momentum: float = 0.9
    student_batch_size: int = 64

    temperature: float = 4.0
    alpha: float = 0.9

    output_dir: str = "outputs"
    teacher_path: str = "outputs/teacher_best.pth"
    generated_dir: str = "outputs/generated"
