from utils import evaluate


def print_final_results(teacher, student_kd, student_baseline, test_loader, device):
    teacher_acc = evaluate(teacher, test_loader, device)
    student_kd_acc = evaluate(student_kd, test_loader, device)
    student_baseline_acc = evaluate(student_baseline, test_loader, device)

    print("\n" + "=" * 50)
    print("最终结果对比")
    print("=" * 50)
    print(f"教师模型准确率:       {teacher_acc:.2f}%")
    print(f"蒸馏学生模型准确率:   {student_kd_acc:.2f}%")
    print(f"基线学生模型准确率:   {student_baseline_acc:.2f}%")
    print(f"蒸馏提升:            +{student_kd_acc - student_baseline_acc:.2f}%")
    print("=" * 50)
