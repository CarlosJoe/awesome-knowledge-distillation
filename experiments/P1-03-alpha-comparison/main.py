from config import Config
from models import TeacherNet
from dataset import get_dataloaders
from train import train_teacher, train_student_baseline, run_alpha_sweep
from evaluate import print_alpha_results, plot_alpha_curve
from utils import get_device, set_seed, evaluate


def main():
    config = Config()
    device = get_device()
    set_seed(config.seed)
    print(f"使用设备: {device}")
    print(f"实验参数: T={config.T}, α ∈ {config.alpha_values}, seed={config.seed}")

    train_loader, test_loader = get_dataloaders(config)

    print("=== 训练教师模型 ===")
    teacher = TeacherNet(dropout=config.dropout).to(device)
    train_teacher(teacher, train_loader, test_loader, device, config)
    teacher_acc = evaluate(teacher, test_loader, device)

    print("\n=== 训练学生基线模型 ===")
    _, baseline_acc = train_student_baseline(train_loader, test_loader, device, config)

    print("\n=== α 蒸馏权重扫描实验 ===")
    alpha_results = run_alpha_sweep(teacher, train_loader, test_loader, device, config)

    print_alpha_results(alpha_results, baseline_acc, teacher_acc, config)
    plot_alpha_curve(alpha_results, baseline_acc, teacher_acc, config)


if __name__ == '__main__':
    main()
