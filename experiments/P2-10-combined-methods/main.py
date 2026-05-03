from config import ExperimentConfig
from dataset import get_cifar100_loaders
from models import ResNet32x4, ResNet8x4
from train import (
    train_teacher,
    distill_combined,
    distill_response_only,
    distill_feature_only,
    distill_attention_only,
    train_student_baseline,
)
from evaluate import print_final_results
from utils import set_seed, get_device


def main():
    config = ExperimentConfig()

    print("=" * 60)
    print("CIFAR-100 多方法组合知识蒸馏实验")
    print("=" * 60)
    print(f"数据集: {config.dataset_name}")
    print(f"教师模型: ResNet-32x4 (blocks={config.teacher_blocks}, width={config.teacher_width})")
    print(f"学生模型: ResNet-8x4 (blocks={config.student_blocks}, width={config.student_width})")
    print(f"组合超参数: T={config.temperature}, α={config.alpha}, β={config.beta}, γ={config.gamma}")
    print(f"优化器: SGD(lr={config.learning_rate}, momentum={config.momentum}, wd={config.weight_decay})")
    print(f"调度器: CosineAnnealing, epochs={config.epochs}")
    print(f"设备: {get_device()}")

    set_seed(config.seed)
    device = get_device()

    train_loader, test_loader = get_cifar100_loaders(config)

    teacher_model = train_teacher(config, train_loader, test_loader, device)

    baseline_student = train_student_baseline(config, train_loader, test_loader, device)

    response_student = distill_response_only(config, teacher_model, train_loader, test_loader, device)

    feature_student = distill_feature_only(config, teacher_model, train_loader, test_loader, device)

    attention_student = distill_attention_only(config, teacher_model, train_loader, test_loader, device)

    combined_student = distill_combined(config, teacher_model, train_loader, test_loader, device)

    print_final_results(config)


if __name__ == "__main__":
    main()
