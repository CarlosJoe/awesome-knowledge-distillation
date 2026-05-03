from config import Config
from models import SimpleUNet, DiffusionSampler
from dataset import get_dataloaders
from train import train_ddpm, progressive_distill
from evaluate import print_results
from utils import get_device, set_seed


def main():
    config = Config()
    device = get_device()
    set_seed(config.seed)
    print(f"使用设备: {device}")
    print(f"步数压缩计划: {config.step_schedule}")

    train_loader, test_loader = get_dataloaders(config)

    teacher = SimpleUNet(
        in_channels=config.in_channels,
        base_channels=config.base_channels,
        channel_multipliers=config.channel_multipliers,
        num_res_blocks=config.num_res_blocks,
        time_embed_dim=config.time_embed_dim,
    ).to(device)

    sampler = DiffusionSampler(
        num_timesteps=config.num_timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
    )

    total_params = sum(p.numel() for p in teacher.parameters())
    print(f"模型参数量: {total_params:,}")

    print("=== 阶段1: 训练教师DDPM (128步) ===")
    teacher = train_ddpm(teacher, sampler, train_loader, device, config)

    print("\n=== 阶段2: 渐进式蒸馏 ===")
    print(f"压缩路线: {' → '.join(str(s) for s in config.step_schedule)}")
    final_model, distilled_models = progressive_distill(
        teacher, sampler, train_loader, device, config
    )

    print("\n=== 阶段3: 评估 ===")
    results = print_results(teacher, distilled_models, sampler, test_loader, device, config)


if __name__ == '__main__':
    main()
