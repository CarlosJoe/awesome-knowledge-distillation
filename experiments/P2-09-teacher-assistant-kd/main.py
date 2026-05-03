from config import ExperimentConfig
from dataset import get_cifar10_loaders
from models import ResNet56, ResNet44, ResNet20
from train import (
    train_teacher,
    train_assistant_from_teacher,
    train_student_from_assistant,
    distill_direct,
    train_student_baseline,
)
from evaluate import print_final_results
from utils import set_seed, get_device


def main():
    config = ExperimentConfig()

    print("=" * 60)
    print("CIFAR-10 助教蒸馏 (TA-KD) 实验")
    print("=" * 60)
    print(f"数据集: {config.dataset_name}")
    print(f"教师模型: ResNet-56 (blocks={config.teacher_blocks})")
    print(f"助教模型: ResNet-44 (blocks={config.assistant_blocks})")
    print(f"学生模型: ResNet-20 (blocks={config.student_blocks})")
    print(f"温度 T={config.temperature}, α={config.alpha}")
    print(f"蒸馏路径: 教师→助教→学生 (两阶段渐进蒸馏)")
    print(f"优化器: SGD(lr={config.learning_rate}, momentum={config.momentum}, wd={config.weight_decay})")
    print(f"调度器: CosineAnnealing, epochs={config.epochs}")
    print(f"设备: {get_device()}")

    set_seed(config.seed)
    device = get_device()

    train_loader, test_loader = get_cifar10_loaders(config)

    teacher_model = train_teacher(config, train_loader, test_loader, device)

    assistant_model = train_assistant_from_teacher(config, teacher_model, train_loader, test_loader, device)

    ta_kd_student = train_student_from_assistant(config, assistant_model, train_loader, test_loader, device)

    direct_kd_student = distill_direct(config, teacher_model, train_loader, test_loader, device)

    baseline_student = train_student_baseline(config, train_loader, test_loader, device)

    print_final_results(config)


if __name__ == "__main__":
    main()
