def print_final_results(config):
    print("\n" + "=" * 60)
    print("实验结果总结 - CIFAR-10 Logit标准化知识蒸馏")
    print("=" * 60)
    print(f"教师模型 (ResNet-56):            {config.teacher_acc:.2f}%")
    print(f"Logit标准化KD学生 (ResNet-20):   {config.standardized_kd_acc:.2f}%")
    print(f"标准KD学生 (ResNet-20):          {config.standard_kd_acc:.2f}%")
    print(f"基线学生模型 (ResNet-20):        {config.baseline_acc:.2f}%")
    print("-" * 60)

    if config.standardized_kd_acc > config.baseline_acc:
        improvement = config.standardized_kd_acc - config.baseline_acc
        print(f"Logit标准化KD vs 基线提升: +{improvement:.2f}%")
    else:
        diff = config.baseline_acc - config.standardized_kd_acc
        print(f"Logit标准化KD vs 基线差距: -{diff:.2f}%")

    if config.standard_kd_acc > config.baseline_acc:
        improvement = config.standard_kd_acc - config.baseline_acc
        print(f"标准KD vs 基线提升: +{improvement:.2f}%")
    else:
        diff = config.baseline_acc - config.standard_kd_acc
        print(f"标准KD vs 基线差距: -{diff:.2f}%")

    if config.standardized_kd_acc > config.standard_kd_acc:
        diff = config.standardized_kd_acc - config.standard_kd_acc
        print(f"Logit标准化KD vs 标准KD提升: +{diff:.2f}%")
    elif config.standardized_kd_acc < config.standard_kd_acc:
        diff = config.standard_kd_acc - config.standardized_kd_acc
        print(f"Logit标准化KD vs 标准KD差距: -{diff:.2f}%")
    else:
        print(f"Logit标准化KD vs 标准KD: 持平")

    if config.teacher_acc > 0:
        gap_std = config.teacher_acc - config.standardized_kd_acc
        gap_kd = config.teacher_acc - config.standard_kd_acc
        print(f"与教师差距 - Logit标准化KD: {gap_std:.2f}%, 标准KD: {gap_kd:.2f}%")

    print("=" * 60)
