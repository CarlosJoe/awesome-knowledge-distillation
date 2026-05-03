from utils import evaluate


def print_results(teacher, student_kd, student_baseline, val_loader, device):
    teacher_acc = evaluate(teacher, val_loader, device)
    student_kd_acc = evaluate(student_kd, val_loader, device)
    student_baseline_acc = evaluate(student_baseline, val_loader, device)

    print("\n" + "=" * 60)
    print("TinyBERT 三层蒸馏结果对比 (SST-2)")
    print("=" * 60)
    print(f"教师模型 (BERT-base, 12层, 110M) 准确率:   {teacher_acc:.2f}%")
    print(f"蒸馏学生模型 (TinyBERT, 4层, 14.5M) 准确率: {student_kd_acc:.2f}%")
    print(f"基线学生模型 (TinyBERT, 4层, 14.5M) 准确率: {student_baseline_acc:.2f}%")
    print(f"蒸馏提升:                                   +{student_kd_acc - student_baseline_acc:.2f}%")
    print("=" * 60)
