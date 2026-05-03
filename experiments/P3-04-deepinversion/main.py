import os
import sys
import torch

from config import DeepInversionConfig
from models import ResNet56, ResNet20
from dataset import get_cifar10_testloader
from train import train_teacher, generate_images, distill_datafree
from evaluate import visualize_generated_images, visualize_single_class, print_final_results
from utils import evaluate, get_device, set_seed, ensure_dir


def main():
    config = DeepInversionConfig()
    set_seed(config.seed)
    device = get_device(config.device)

    print("=" * 60)
    print("DeepInversion 无数据蒸馏实验 - CIFAR-10")
    print("=" * 60)
    print(f"设备: {device}")
    print(f"教师: ResNet-56 | 学生: ResNet-20")
    print(f"生成迭代: {config.inv_iterations} | 学习率: {config.inv_lr}")
    print(f"BN权重: {config.bn_weight} | TV权重: {config.tv_weight} | L2权重: {config.l2_weight}")
    print(f"每类图像: {config.images_per_class} | 温度: {config.temperature} | α: {config.alpha}")
    print("=" * 60)

    ensure_dir(config.output_dir)

    if os.path.exists(config.teacher_path):
        print(f"\n加载已有教师模型: {config.teacher_path}")
        teacher = ResNet56(num_classes=config.num_classes).to(device)
        teacher.load_state_dict(torch.load(config.teacher_path, map_location=device))
    else:
        print("\n未找到教师模型, 开始训练...")
        teacher = train_teacher(config)

    testloader = get_cifar10_testloader(config.student_batch_size, config.num_workers)
    teacher_acc = evaluate(teacher, testloader, device)
    print(f"\n教师模型测试准确率: {teacher_acc:.2f}%")

    gen_data_path = os.path.join(config.generated_dir, "generated_data.pth")
    if os.path.exists(gen_data_path):
        print(f"\n加载已有生成数据: {gen_data_path}")
        data = torch.load(gen_data_path, map_location="cpu")
        generated_images = data["images"]
        generated_labels = data["labels"]
    else:
        print("\n开始 DeepInversion 图像生成...")
        generated_images, generated_labels = generate_images(config, teacher)

    print(f"\n生成图像统计: {generated_images.shape[0]} 张, 形状 {generated_images.shape[1:]}")
    print(f"  像素范围: [{generated_images.min():.3f}, {generated_images.max():.3f}]")

    visualize_generated_images(config, generated_images, generated_labels)
    for cls_idx in [0, 1, 2]:
        visualize_single_class(config, generated_images, generated_labels, cls_idx)

    print("\n开始无数据知识蒸馏...")
    student, student_acc = distill_datafree(config, teacher, generated_images, generated_labels)

    print_final_results(config, teacher_acc, student_acc)


if __name__ == "__main__":
    main()
