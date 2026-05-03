import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from config import ATConfig
from models import ResNet
from utils import evaluate


def print_final_results(teacher, student_at, student_baseline, test_loader, device, config=ATConfig()):
    teacher_acc = evaluate(teacher, test_loader, device)
    at_acc = evaluate(student_at, test_loader, device)
    baseline_acc = evaluate(student_baseline, test_loader, device)

    print("\n" + "=" * 60)
    print("Attention Transfer 蒸馏实验结果")
    print("=" * 60)
    print(f"教师模型 (ResNet-56) 准确率:       {teacher_acc:.2f}%")
    print(f"学生+AT蒸馏 (ResNet-20) 准确率:    {at_acc:.2f}%")
    print(f"学生基线 (ResNet-20) 准确率:        {baseline_acc:.2f}%")
    print(f"AT蒸馏提升:                        +{at_acc - baseline_acc:.2f}%")
    print("=" * 60)

    return {
        "teacher_acc": teacher_acc,
        "at_acc": at_acc,
        "baseline_acc": baseline_acc,
        "improvement": at_acc - baseline_acc,
    }


def visualize_attention_maps(teacher, student, test_loader, device, config=ATConfig(), num_samples=4):
    teacher.eval()
    student.eval()
    teacher._register_hooks()
    student._register_hooks()

    save_dir = config.attention_map_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    images_batch, labels_batch = next(iter(test_loader))
    images_batch = images_batch[:num_samples].to(device)
    labels_batch = labels_batch[:num_samples]

    layer_names = ["layer1", "layer2", "layer3"]
    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]

    with torch.no_grad():
        teacher_logits, teacher_features = teacher(images_batch, return_features=True)
        student_logits, student_features = student(images_batch, return_features=True)

    teacher_attn = {}
    student_attn = {}
    for name in layer_names:
        t_feat = teacher_features[name]
        s_feat = student_features[name]
        teacher_attn[name] = (t_feat.abs() ** config.at_p).sum(dim=1, keepdim=True)
        student_attn[name] = (s_feat.abs() ** config.at_p).sum(dim=1, keepdim=True)

    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
    raw_images = images_batch.cpu() * std + mean
    raw_images = raw_images.clamp(0, 1)

    for sample_idx in range(num_samples):
        n_cols = 1 + len(layer_names)
        fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))

        img = raw_images[sample_idx].permute(1, 2, 0).numpy()
        axes[0, 0].imshow(img)
        axes[0, 0].set_title("Input", fontsize=10)
        axes[0, 0].axis("off")
        axes[1, 0].imshow(img)
        axes[1, 0].set_title("Input", fontsize=10)
        axes[1, 0].axis("off")

        for col, name in enumerate(layer_names, start=1):
            t_map = teacher_attn[name][sample_idx, 0].cpu().numpy()
            s_map = student_attn[name][sample_idx, 0].cpu().numpy()

            t_map = (t_map - t_map.min()) / (t_map.max() - t_map.min() + 1e-8)
            s_map = (s_map - s_map.min()) / (s_map.max() - s_map.min() + 1e-8)

            axes[0, col].imshow(t_map, cmap="jet", interpolation="bilinear")
            axes[0, col].set_title(f"Teacher {name}", fontsize=9)
            axes[0, col].axis("off")

            axes[1, col].imshow(s_map, cmap="jet", interpolation="bilinear")
            axes[1, col].set_title(f"Student {name}", fontsize=9)
            axes[1, col].axis("off")

        label = class_names[labels_batch[sample_idx].item()]
        fig.suptitle(f"Attention Maps - Sample {sample_idx+1} (label: {label})", fontsize=12)
        plt.tight_layout()
        plt.savefig(save_dir / f"attention_sample_{sample_idx+1}.png", dpi=150, bbox_inches="tight")
        plt.close()

    teacher.remove_hooks()
    student.remove_hooks()
    print(f"注意力图已保存到 {save_dir}/")
