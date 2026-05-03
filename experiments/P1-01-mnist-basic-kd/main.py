from config import Config
from models import TeacherNet, StudentNet
from dataset import get_dataloaders
from train import train_teacher, distill_train, train_student_baseline
from evaluate import print_final_results
from utils import get_device


def main():
    config = Config()
    device = get_device()
    print(f"使用设备: {device}")

    train_loader, test_loader = get_dataloaders(config)

    teacher = TeacherNet(dropout=config.dropout).to(device)
    student_kd = StudentNet().to(device)
    student_baseline = StudentNet().to(device)

    print("=== 训练教师模型 ===")
    train_teacher(teacher, train_loader, test_loader, device, config)

    print("\n=== 蒸馏训练学生模型 (T=4, α=0.7) ===")
    distill_train(teacher, student_kd, train_loader, test_loader, device, config)

    print("\n=== 直接训练学生模型（基线） ===")
    train_student_baseline(student_baseline, train_loader, test_loader, device, config)

    print_final_results(teacher, student_kd, student_baseline, test_loader, device)


if __name__ == '__main__':
    main()
