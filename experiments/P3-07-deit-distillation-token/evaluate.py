from utils import evaluate


def print_final_results(teacher, deit_distill, deit_baseline, test_loader, device):
    teacher_acc = evaluate(teacher, test_loader, device)
    deit_distill_acc = evaluate(deit_distill, test_loader, device, is_deit_distill=True)
    deit_baseline_acc = evaluate(deit_baseline, test_loader, device)

    print("\n" + "=" * 60)
    print("最终结果对比")
    print("=" * 60)
    print(f"教师模型 (ResNet-56) 准确率:       {teacher_acc:.2f}%")
    print(f"DeiT + 蒸馏Token 准确率:           {deit_distill_acc:.2f}%")
    print(f"DeiT 无蒸馏 (基线) 准确率:         {deit_baseline_acc:.2f}%")
    print(f"蒸馏Token提升:                     +{deit_distill_acc - deit_baseline_acc:.2f}%")
    print("=" * 60)
