def print_final_results(config):
    print("\n" + "=" * 60)
    print("实验结果总结 - CIFAR-10 Response-based 知识蒸馏")
    print("=" * 60)
    print(f"教师模型 (ResNet-56):      {config.teacher_acc:.2f}%")
    print(f"蒸馏学生模型 (ResNet-20):  {config.distilled_acc:.2f}%")
    print(f"基线学生模型 (ResNet-20):  {config.baseline_acc:.2f}%")
    print("-" * 60)

    if config.distilled_acc > config.baseline_acc:
        improvement = config.distilled_acc - config.baseline_acc
        print(f"蒸馏提升: +{improvement:.2f}%")
    else:
        diff = config.baseline_acc - config.distilled_acc
        print(f"蒸馏差距: -{diff:.2f}%")

    if config.teacher_acc > 0:
        gap_teacher = config.teacher_acc - config.distilled_acc
        print(f"与教师差距: {gap_teacher:.2f}%")

    print("=" * 60)
