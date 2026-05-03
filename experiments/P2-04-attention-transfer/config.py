from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ATConfig:
    data_root: str = "./data"
    batch_size: int = 128
    num_workers: int = 4
    epochs: int = 200
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    lr_milestones: list = field(default_factory=lambda: [100, 150])
    lr_gamma: float = 0.1

    temperature: float = 4.0
    alpha: float = 0.7
    at_p: int = 2
    at_beta: float = 1e3

    teacher_depth: int = 56
    student_depth: int = 20
    num_classes: int = 10

    seed: int = 42
    device: str = "cuda"
    save_dir: str = "checkpoints"
    results_dir: str = "results"

    teacher_ckpt: str = field(default="")
    student_baseline_ckpt: str = field(default="")

    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.save_dir)

    @property
    def attention_map_dir(self) -> Path:
        return Path(self.results_dir) / "attention_maps"
