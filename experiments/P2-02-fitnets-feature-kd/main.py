from config import ExperimentConfig
from dataset import get_cifar10_loaders
from models import FitNetsTeacher, FitNetsStudent, HintRegressor
from train import train_teacher, hint_training, kd_training, train_student_baseline
from evaluate import print_final_results
from utils import set_seed, get_device


def main():
    config = ExperimentConfig()

    print("=" * 60)
    print("CIFAR-10 FitNets 特征蒸馏实验")
    print("=" * 60)
    print(f"数据集: {config.dataset_name}")
    print(f"教师模型: FitNetsTeacher (channels={config.teacher_channels})")
    print(f"学生模型: FitNetsStudent (channels={config.student_channels})")
    print(f"Hint通道: {config.hint_channels}")
    print(f"温度 T={config.temperature}, α={config.alpha}, β={config.beta}")
    print(f"Hint阶段: {config.hint_epochs} epochs, KD阶段: {config.kd_epochs} epochs")
    print(f"优化器: SGD(lr={config.learning_rate}, momentum={config.momentum}, wd={config.weight_decay})")
    print(f"调度器: CosineAnnealing")
    print(f"设备: {get_device()}")

    set_seed(config.seed)
    device = get_device()

    train_loader, test_loader = get_cifar10_loaders(config)

    teacher_model = train_teacher(config, train_loader, test_loader, device)

    student_model = FitNetsStudent(num_classes=config.num_classes).to(device)
    regressor = HintRegressor(
        student_hint_channels=config.hint_channels,
        teacher_hint_channels=config.hint_channels,
    ).to(device)

    student_model, regressor = hint_training(
        config, teacher_model, student_model, regressor,
        train_loader, test_loader, device,
    )

    student_model = kd_training(
        config, teacher_model, student_model, regressor,
        train_loader, test_loader, device,
    )

    baseline_student = train_student_baseline(config, train_loader, test_loader, device)

    print_final_results(config)


if __name__ == "__main__":
    main()
