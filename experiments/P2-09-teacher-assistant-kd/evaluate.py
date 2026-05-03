def print_final_results(config):
    print("\n" + "=" * 60)
    print("实验结果总结 - CIFAR-10 助教蒸馏 (TA-KD)")
    print("=" * 60)
    print(f"教师模型 (ResNet-56):          {config.teacher_acc:.2f}%")
    print(f"助教模型 (ResNet-44):          {config.assistant_acc:.2f}%")
    print(f"TA-KD学生模型 (ResNet-20):     {config.ta_kd_acc:.2f}%")
    print(f"直接蒸馏学生模型 (ResNet-20):  {config.direct_kd_acc:.2f}%")
    print(f"基线学生模型 (ResNet-20):      {config.baseline_acc:.2f}%")
    print("-" * 60)

    if config.ta_kd_acc > config.direct_kd_acc:
        improvement = config.ta_kd_acc - config.direct_kd_acc
        print(f"TA-KD vs 直接蒸馏提升: +{improvement:.2f}%")
    else:
        diff = config.direct_kd_acc - config.ta_kd_acc
        print(f"TA-KD vs 直接蒸馏差距: -{diff:.2f}%")

    if config.ta_kd_acc > config.baseline_acc:
        improvement = config.ta_kd_acc - config.baseline_acc
        print(f"TA-KD vs 基线提升: +{improvement:.2f}%")
    else:
        diff = config.baseline_acc - config.ta_kd_acc
        print(f"TA-KD vs 基线差距: -{diff:.2f}%")

    if config.teacher_acc > 0:
        gap_ta = config.teacher_acc - config.ta_kd_acc
        gap_direct = config.teacher_acc - config.direct_kd_acc
        print(f"TA-KD与教师差距: {gap_ta:.2f}%")
        print(f"直接蒸馏与教师差距: {gap_direct:.2f}%")

    print("=" * 60)
