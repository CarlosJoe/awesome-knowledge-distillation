from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    dataset_name: str = "CIFAR-10"
    num_classes: int = 10
    input_channels: int = 3
    image_size: int = 32

    teacher_blocks: tuple = (9, 9, 9)
    student_blocks: tuple = (3, 3, 3)

    temperature: float = 4.0
    alpha: float = 0.7

    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    epochs: int = 100
    batch_size: int = 128
    num_workers: int = 4

    seed: int = 42
    data_dir: str = "./data"
    checkpoint_dir: str = "./checkpoints"

    teacher_acc: float = 0.0
    standardized_kd_acc: float = 0.0
    standard_kd_acc: float = 0.0
    baseline_acc: float = 0.0

    mean: tuple = (0.4914, 0.4822, 0.4465)
    std: tuple = (0.2470, 0.2435, 0.2616)
