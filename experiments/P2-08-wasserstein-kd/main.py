from config import ExperimentConfig
from dataset import get_cifar10_loaders
from models import ResNet56, ResNet20
from train import train_teacher, distill_train_wasserstein, distill_train_kl, train_student_baseline
from evaluate import print_final_results
from utils import set_seed, get_device


def main():
    config = ExperimentConfig()

    print("=" * 60)
    print("CIFAR-10 Wasserstein 知识蒸馏实验")
    print("=" * 60)
    print(f"数据集: {config.dataset_name}")
    print(f"教师模型: ResNet-56 (blocks={config.teacher_blocks})")
    print(f"学生模型: ResNet-20 (blocks={config.student_blocks})")
    print(f"Wasserstein 蒸馏: α={config.alpha}, 距离: W_1")
    print(f"KL 散度蒸馏: T={config.temperature}, α={config.alpha}")
    print(f"优化器: SGD(lr={config.learning_rate}, momentum={config.momentum}, wd={config.weight_decay})")
    print(f"调度器: CosineAnnealing, epochs={config.epochs}")
    print(f"设备: {get_device()}")

    set_seed(config.seed)
    device = get_device()

    train_loader, test_loader = get_cifar10_loaders(config)

    teacher_model = train_teacher(config, train_loader, test_loader, device)

    wasserstein_student = distill_train_wasserstein(config, teacher_model, train_loader, test_loader, device)

    kl_student = distill_train_kl(config, teacher_model, train_loader, test_loader, device)

    baseline_student = train_student_baseline(config, train_loader, test_loader, device)

    print_final_results(config)


if __name__ == "__main__":
    main()
