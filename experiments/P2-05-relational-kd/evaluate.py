def print_final_results(config):
    print("\n" + "=" * 60)
    print("实验结果总结 - CIFAR-100 Relational 知识蒸馏 (RKD)")
    print("=" * 60)
    print(f"教师模型 (ResNet-56):              {config.teacher_acc:.2f}%")
    print(f"RKD 蒸馏学生模型 (ResNet-20):      {config.rkd_acc:.2f}%")
    print(f"Response KD 蒸馏学生 (ResNet-20):  {config.response_kd_acc:.2f}%")
    print(f"基线学生模型 (ResNet-20):          {config.baseline_acc:.2f}%")
    print("-" * 60)

    if config.rkd_acc > config.baseline_acc:
        improvement = config.rkd_acc - config.baseline_acc
        print(f"RKD 蒸馏提升 (vs 基线): +{improvement:.2f}%")
    else:
        diff = config.baseline_acc - config.rkd_acc
        print(f"RKD 蒸馏差距 (vs 基线): -{diff:.2f}%")

    if config.response_kd_acc > config.baseline_acc:
        improvement = config.response_kd_acc - config.baseline_acc
        print(f"Response KD 蒸馏提升 (vs 基线): +{improvement:.2f}%")
    else:
        diff = config.baseline_acc - config.response_kd_acc
        print(f"Response KD 蒸馏差距 (vs 基线): -{diff:.2f}%")

    print("-" * 60)

    if config.rkd_acc > config.response_kd_acc:
        diff = config.rkd_acc - config.response_kd_acc
        print(f"RKD 优于 Response KD: +{diff:.2f}%")
    elif config.response_kd_acc > config.rkd_acc:
        diff = config.response_kd_acc - config.rkd_acc
        print(f"Response KD 优于 RKD: +{diff:.2f}%")
    else:
        print("RKD 与 Response KD 性能相同")

    if config.teacher_acc > 0:
        gap_rkd = config.teacher_acc - config.rkd_acc
        gap_response = config.teacher_acc - config.response_kd_acc
        print(f"RKD 与教师差距: {gap_rkd:.2f}%")
        print(f"Response KD 与教师差距: {gap_response:.2f}%")

    print("=" * 60)
