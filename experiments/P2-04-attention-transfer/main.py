from config import ATConfig
from models import get_model
from dataset import get_cifar10_loaders
from train import train_teacher, distill_train_at, train_student_baseline
from evaluate import print_final_results, visualize_attention_maps
from utils import set_seed, get_device


def main():
    config = ATConfig()
    set_seed(config.seed)
    device = get_device(config.device)
    print(f"使用设备: {device}")
    print(f"配置: T={config.temperature}, α={config.alpha}, p={config.at_p}, β={config.at_beta}")

    train_loader, test_loader = get_cifar10_loaders(
        data_root=config.data_root, batch_size=config.batch_size, num_workers=config.num_workers
    )

    teacher = get_model(config.teacher_depth, config.num_classes).to(device)
    student_at = get_model(config.student_depth, config.num_classes).to(device)
    student_baseline = get_model(config.student_depth, config.num_classes).to(device)

    if config.teacher_ckpt:
        print(f"加载教师模型权重: {config.teacher_ckpt}")
        teacher.load_state_dict(torch.load(config.teacher_ckpt, map_location=device))
    else:
        print("=== 训练教师模型 (ResNet-56) ===")
        train_teacher(teacher, train_loader, test_loader, device, config)

    if config.student_baseline_ckpt:
        print(f"加载学生基线模型权重: {config.student_baseline_ckpt}")
        student_baseline.load_state_dict(torch.load(config.student_baseline_ckpt, map_location=device))
    else:
        print("\n=== 训练学生基线 (ResNet-20, 无蒸馏) ===")
        train_student_baseline(student_baseline, train_loader, test_loader, device, config)

    print("\n=== Attention Transfer 蒸馏训练 (ResNet-56 → ResNet-20) ===")
    distill_train_at(teacher, student_at, train_loader, test_loader, device, config)

    print("\n=== 最终结果对比 ===")
    results = print_final_results(teacher, student_at, student_baseline, test_loader, device, config)

    print("\n=== 生成注意力图可视化 ===")
    visualize_attention_maps(teacher, student_at, test_loader, device, config)


if __name__ == "__main__":
    import torch
    main()
