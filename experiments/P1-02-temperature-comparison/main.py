from config import Config
from models import TeacherNet, StudentNet
from dataset import get_dataloaders
from train import train_teacher, train_student_baseline, run_temperature_sweep
from evaluate import (print_temperature_results, plot_temperature_curve,
                      plot_convergence_curves, visualize_soft_labels)
from utils import get_device, set_seed, evaluate


def main():
    config = Config()
    set_seed(config.seed)
    device = get_device()
    print(f"使用设备: {device}")

    train_loader, test_loader = get_dataloaders(config)

    print("=== 训练教师模型 ===")
    teacher = TeacherNet(dropout=config.dropout).to(device)
    train_teacher(teacher, train_loader, test_loader, device, config)
    teacher_acc = evaluate(teacher, test_loader, device)
    print(f"教师模型最终准确率: {teacher_acc:.2f}%")

    print("\n=== 训练基线学生模型 ===")
    set_seed(config.seed)
    student_baseline = StudentNet().to(device)
    train_student_baseline(student_baseline, train_loader, test_loader, device, config)
    baseline_acc = evaluate(student_baseline, test_loader, device)
    print(f"基线学生模型准确率: {baseline_acc:.2f}%")

    print("\n=== 温度参数扫描实验 ===")
    results = run_temperature_sweep(teacher, train_loader, test_loader, device, config)

    print_temperature_results(results, baseline_acc, teacher_acc)

    plot_temperature_curve(results, baseline_acc, save_path='./results/temperature_curve.png')

    plot_convergence_curves(results, save_path='./results/convergence_curves.png')

    visualize_soft_labels(teacher, test_loader, device, config.T_values,
                          config.num_vis_samples, save_dir='./results/soft_labels')


if __name__ == '__main__':
    main()
