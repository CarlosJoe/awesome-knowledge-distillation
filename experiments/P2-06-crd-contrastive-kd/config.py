from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    dataset_name: str = "CIFAR-100"
    num_classes: int = 100
    input_channels: int = 3
    image_size: int = 32

    teacher_blocks: tuple = (5, 5, 5)
    teacher_width: int = 4
    student_blocks: tuple = (1, 1, 1)
    student_width: int = 4

    temperature: float = 4.0
    alpha: float = 0.7
    crd_temp: float = 0.07
    projector_hidden: int = 128

    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    epochs: int = 100
    batch_size: int = 64
    num_workers: int = 4

    seed: int = 42
    data_dir: str = "./data"
    checkpoint_dir: str = "./checkpoints"

    teacher_acc: float = 0.0
    crd_acc: float = 0.0
    baseline_acc: float = 0.0
    response_kd_acc: float = 0.0

    mean: tuple = (0.5071, 0.4867, 0.4408)
    std: tuple = (0.2675, 0.2565, 0.2761)
