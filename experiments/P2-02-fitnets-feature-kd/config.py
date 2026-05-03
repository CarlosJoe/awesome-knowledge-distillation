from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    dataset_name: str = "CIFAR-10"
    num_classes: int = 10
    input_channels: int = 3
    image_size: int = 32

    teacher_channels: tuple = (32, 64, 128)
    student_channels: tuple = (16, 32, 128)
    hint_channels: int = 128

    temperature: float = 4.0
    alpha: float = 0.7
    beta: float = 0.1

    hint_epochs: int = 30
    kd_epochs: int = 100
    epochs: int = 100

    learning_rate: float = 0.1
    hint_learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    batch_size: int = 128
    num_workers: int = 4

    seed: int = 42
    data_dir: str = "./data"
    checkpoint_dir: str = "./checkpoints"

    teacher_acc: float = 0.0
    distilled_acc: float = 0.0
    baseline_acc: float = 0.0
    hint_acc: float = 0.0

    mean: tuple = (0.4914, 0.4822, 0.4465)
    std: tuple = (0.2470, 0.2435, 0.2616)
