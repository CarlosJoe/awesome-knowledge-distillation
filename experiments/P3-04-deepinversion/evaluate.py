import os
import torch
import torchvision
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import DeepInversionConfig
from models import ResNet56, ResNet20
from dataset import get_cifar10_testloader
from utils import evaluate, get_device, ensure_dir


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def visualize_generated_images(config, generated_images, generated_labels):
    ensure_dir(config.generated_dir)
    num_per_class = min(8, config.images_per_class)
    fig, axes = plt.subplots(config.num_classes, num_per_class, figsize=(num_per_class * 1.5, config.num_classes * 1.5))

    for cls_idx in range(config.num_classes):
        mask = generated_labels == cls_idx
        cls_images = generated_images[mask][:num_per_class]
        for j in range(num_per_class):
            ax = axes[cls_idx, j]
            img = cls_images[j].permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.axis("off")
            if j == 0:
                ax.set_ylabel(CIFAR10_CLASSES[cls_idx], fontsize=8, rotation=0, labelpad=40)

    plt.suptitle("DeepInversion 生成的图像", fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(config.generated_dir, "generated_images.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"生成图像可视化已保存至 {save_path}")


def visualize_single_class(config, generated_images, generated_labels, cls_idx=0):
    ensure_dir(config.generated_dir)
    mask = generated_labels == cls_idx
    cls_images = generated_images[mask][:16]

    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i, ax in enumerate(axes.flat):
        if i < len(cls_images):
            img = cls_images[i].permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            ax.imshow(img)
        ax.axis("off")

    plt.suptitle(f"类别 {cls_idx} ({CIFAR10_CLASSES[cls_idx]}) 的生成图像", fontsize=12)
    plt.tight_layout()
    save_path = os.path.join(config.generated_dir, f"class_{cls_idx}_{CIFAR10_CLASSES[cls_idx]}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"类别 {cls_idx} 生成图像已保存至 {save_path}")


def print_final_results(config, teacher_acc, student_acc):
    print("\n" + "=" * 60)
    print("实验结果总结 - DeepInversion 无数据蒸馏 (CIFAR-10)")
    print("=" * 60)
    print(f"教师模型 (ResNet-56):     {teacher_acc:.2f}%")
    print(f"学生模型 (ResNet-20):     {student_acc:.2f}%")
    print("-" * 60)

    if teacher_acc > 0:
        gap = teacher_acc - student_acc
        print(f"师生准确率差距:           {gap:.2f}%")

    if student_acc > 85.0:
        print(f"\n✓ 无数据蒸馏准确率 {student_acc:.2f}% > 85%, 达到预期目标!")
    else:
        print(f"\n✗ 无数据蒸馏准确率 {student_acc:.2f}% <= 85%, 未达预期目标")

    print("\n--- DeepInversion 关键信息 ---")
    print("本实验全程不使用原始训练数据，仅通过教师 BN 统计量反演生成图像")
    print(f"生成超参数: iterations={config.inv_iterations}, lr={config.inv_lr}")
    print(f"损失权重: bn_weight={config.bn_weight}, tv_weight={config.tv_weight}, l2_weight={config.l2_weight}")
    print(f"每类生成: {config.images_per_class} 张, 共 {config.num_classes * config.images_per_class} 张")
    print(f"蒸馏超参数: T={config.temperature}, α={config.alpha}")
    print("=" * 60)
