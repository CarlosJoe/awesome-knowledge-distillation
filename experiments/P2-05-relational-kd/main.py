from config import ExperimentConfig
from dataset import get_cifar100_loaders
from models import ResNet56, ResNet20
from train import train_teacher, distill_train_rkd, distill_train_response_kd, train_student_baseline
from evaluate import print_final_results
from utils import set_seed, get_device


def main():
    config = ExperimentConfig()

    print("=" * 60)
    print("CIFAR-100 Relational 知识蒸馏实验 (RKD)")
    print("=" * 60)
    print(f"数据集: {config.dataset_name}")
    print(f"教师模型: ResNet-56 (blocks={config.teacher_blocks})")
    print(f"学生模型: ResNet-20 (blocks={config.student_blocks})")
    print(f"温度 T={config.temperature}, α={config.alpha}")
    print(f"w_dist={config.w_dist}, w_angle={config.w_angle}")
    print(f"优化器: SGD(lr={config.learning_rate}, momentum={config.momentum}, wd={config.weight_decay})")
    print(f"调度器: CosineAnnealing, epochs={config.epochs}")
    print(f"设备: {get_device()}")

    set_seed(config.seed)
    device = get_device()

    train_loader, test_loader = get_cifar100_loaders(config)

    teacher_model = train_teacher(config, train_loader, test_loader, device)

    rkd_student = distill_train_rkd(config, teacher_model, train_loader, test_loader, device)

    response_kd_student = distill_train_response_kd(config, teacher_model, train_loader, test_loader, device)

    baseline_student = train_student_baseline(config, train_loader, test_loader, device)

    print_final_results(config)


if __name__ == "__main__":
    main()
