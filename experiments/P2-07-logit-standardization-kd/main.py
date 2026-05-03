from config import ExperimentConfig
from dataset import get_cifar10_loaders
from models import ResNet56, ResNet20
from train import train_teacher, distill_train_standardized, distill_train_standard, train_student_baseline
from evaluate import print_final_results
from utils import set_seed, get_device


def main():
    config = ExperimentConfig()

    print("=" * 60)
    print("CIFAR-10 Logit标准化知识蒸馏实验")
    print("=" * 60)
    print(f"数据集: {config.dataset_name}")
    print(f"教师模型: ResNet-56 (blocks={config.teacher_blocks})")
    print(f"学生模型: ResNet-20 (blocks={config.student_blocks})")
    print(f"温度 T={config.temperature}, α={config.alpha}")
    print(f"Logit标准化: ẑ = (z - μ) / σ")
    print(f"优化器: SGD(lr={config.learning_rate}, momentum={config.momentum}, wd={config.weight_decay})")
    print(f"调度器: CosineAnnealing, epochs={config.epochs}")
    print(f"设备: {get_device()}")

    set_seed(config.seed)
    device = get_device()

    train_loader, test_loader = get_cifar10_loaders(config)

    teacher_model = train_teacher(config, train_loader, test_loader, device)

    standardized_student = distill_train_standardized(config, teacher_model, train_loader, test_loader, device)

    standard_student = distill_train_standard(config, teacher_model, train_loader, test_loader, device)

    baseline_student = train_student_baseline(config, train_loader, test_loader, device)

    print_final_results(config)


if __name__ == "__main__":
    main()
