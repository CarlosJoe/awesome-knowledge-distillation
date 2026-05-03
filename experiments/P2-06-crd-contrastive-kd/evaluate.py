def print_final_results(config):
    print("\n" + "=" * 60)
    print("实验结果总结 - CIFAR-100 CRD 对比知识蒸馏")
    print("=" * 60)
    print(f"教师模型 (ResNet-32x4):       {config.teacher_acc:.2f}%")
    print(f"CRD 蒸馏学生 (ResNet-8x4):    {config.crd_acc:.2f}%")
    print(f"基线学生模型 (ResNet-8x4):    {config.baseline_acc:.2f}%")
    if config.response_kd_acc > 0:
        print(f"Response KD 参考:             {config.response_kd_acc:.2f}%")
    print("-" * 60)

    if config.crd_acc > config.baseline_acc:
        improvement = config.crd_acc - config.baseline_acc
        print(f"CRD 相比基线提升: +{improvement:.2f}%")
    else:
        diff = config.baseline_acc - config.crd_acc
        print(f"CRD 相比基线差距: -{diff:.2f}%")

    if config.response_kd_acc > 0:
        if config.crd_acc > config.response_kd_acc:
            improvement = config.crd_acc - config.response_kd_acc
            print(f"CRD 相比 Response KD 提升: +{improvement:.2f}%")
        else:
            diff = config.response_kd_acc - config.crd_acc
            print(f"CRD 相比 Response KD 差距: -{diff:.2f}%")

    if config.teacher_acc > 0:
        gap_teacher = config.teacher_acc - config.crd_acc
        print(f"与教师差距: {gap_teacher:.2f}%")

    print("=" * 60)
