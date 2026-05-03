from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    image_size: int = 32
    in_channels: int = 3
    base_channels: int = 64
    channel_multipliers: List[int] = field(default_factory=lambda: [1, 2, 4])
    num_res_blocks: int = 2
    time_embed_dim: int = 256

    num_timesteps: int = 128
    beta_start: float = 1e-4
    beta_end: float = 0.02
    step_schedule: List[int] = field(default_factory=lambda: [128, 64, 32, 16, 8, 4])

    teacher_epochs: int = 50
    distill_epochs_per_stage: int = 50
    learning_rate: float = 1e-4
    batch_size: int = 128
    num_workers: int = 4
    data_dir: str = './data'
    seed: int = 42
    num_samples_fid: int = 1000
    save_dir: str = './checkpoints'
