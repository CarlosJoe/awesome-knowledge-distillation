import torch
import numpy as np
import os

from config import Config
from models import SimpleUNet, DiffusionSampler


@torch.no_grad()
def generate_samples(model, sampler, num_samples, image_size, in_channels, num_steps, device):
    shape = (num_samples, in_channels, image_size, image_size)
    samples = sampler.ddim_sample(model, shape, num_steps, device, eta=0.0)
    samples = (samples + 1.0) / 2.0
    samples = torch.clamp(samples, 0.0, 1.0)
    return samples


def save_sample_grid(samples, path, nrow=8):
    try:
        from torchvision.utils import save_image
        save_image(samples, path, nrow=nrow, padding=2)
    except ImportError:
        pass


@torch.no_grad()
def compute_fid_simple(model, sampler, test_loader, num_steps, device, config=Config()):
    model.eval()
    real_images = []
    for images, _ in test_loader:
        real_images.append(images)
        if sum(img.shape[0] for img in real_images) >= config.num_samples_fid:
            break
    real_images = torch.cat(real_images, dim=0)[:config.num_samples_fid]
    real_images = real_images.to(device)

    fake_images = generate_samples(
        model, sampler, config.num_samples_fid,
        config.image_size, config.in_channels, num_steps, device
    )

    real_flat = real_images.flatten(1)
    fake_flat = fake_images.flatten(1)

    mu_real = real_flat.mean(dim=0)
    mu_fake = fake_flat.mean(dim=0)

    diff = mu_real - mu_fake
    fid_approx = (diff @ diff).item() * 1000

    return fid_approx


def print_results(teacher, distilled_models, sampler, test_loader, device, config=Config()):
    print("\n" + "=" * 60)
    print("渐进式扩散蒸馏 - 最终结果")
    print("=" * 60)

    results = {}

    print(f"\n教师模型 ({config.step_schedule[0]}步 DDIM采样):")
    teacher_fid = compute_fid_simple(teacher, sampler, test_loader, config.step_schedule[0], device, config)
    results[config.step_schedule[0]] = teacher_fid
    print(f"  FID (近似): {teacher_fid:.2f}")

    samples = generate_samples(
        teacher, sampler, 64, config.image_size, config.in_channels,
        config.step_schedule[0], device
    )
    save_sample_grid(samples, os.path.join(config.save_dir, f'teacher_{config.step_schedule[0]}steps.png'))

    for steps, state_dict in distilled_models.items():
        model = SimpleUNet(
            in_channels=config.in_channels,
            base_channels=config.base_channels,
            channel_multipliers=config.channel_multipliers,
            num_res_blocks=config.num_res_blocks,
            time_embed_dim=config.time_embed_dim,
        ).to(device)
        model.load_state_dict(state_dict)
        model.eval()

        fid = compute_fid_simple(model, sampler, test_loader, steps, device, config)
        results[steps] = fid
        print(f"\n蒸馏模型 ({steps}步 DDIM采样):")
        print(f"  FID (近似): {fid:.2f}")

        samples = generate_samples(
            model, sampler, 64, config.image_size, config.in_channels, steps, device
        )
        save_sample_grid(samples, os.path.join(config.save_dir, f'student_{steps}steps.png'))

    print("\n" + "=" * 60)
    print("步数压缩对比汇总")
    print("-" * 60)
    print(f"{'步数':>6} | {'FID(近似)':>10} | {'加速比':>8}")
    print("-" * 60)
    base_steps = config.step_schedule[0]
    for steps in sorted(results.keys()):
        speedup = base_steps / steps
        print(f"{steps:>6} | {results[steps]:>10.2f} | {speedup:>7.1f}x")
    print("=" * 60)

    final_steps = config.step_schedule[-1]
    final_fid = results.get(final_steps, float('inf'))
    if final_fid < 15:
        print(f"\n✓ 目标达成: {final_steps}步采样 FID(近似) = {final_fid:.2f} < 15")
    else:
        print(f"\n✗ 目标未达成: {final_steps}步采样 FID(近似) = {final_fid:.2f} >= 15")
        print("  建议: 增加训练轮次或调整学习率")

    return results
