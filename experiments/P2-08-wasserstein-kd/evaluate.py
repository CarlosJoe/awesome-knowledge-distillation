def print_final_results(config):
    print("\n" + "=" * 60)
    print("实验结果总结 - CIFAR-10 Wasserstein 知识蒸馏")
    print("=" * 60)
    print(f"教师模型 (ResNet-56):            {config.teacher_acc:.2f}%")
    print(f"Wasserstein 蒸馏学生 (ResNet-20): {config.wasserstein_acc:.2f}%")
    print(f"KL 散度蒸馏学生 (ResNet-20):      {config.kl_acc:.2f}%")
    print(f"基线学生模型 (ResNet-20):         {config.baseline_acc:.2f}%")
    print("-" * 60)

    if config.wasserstein_acc > config.baseline_acc:
        improvement = config.wasserstein_acc - config.baseline_acc
        print(f"Wasserstein 蒸馏提升: +{improvement:.2f}%")
    else:
        diff = config.baseline_acc - config.wasserstein_acc
        print(f"Wasserstein 蒸馏差距: -{diff:.2f}%")

    if config.kl_acc > config.baseline_acc:
        improvement = config.kl_acc - config.baseline_acc
        print(f"KL 散度蒸馏提升: +{improvement:.2f}%")
    else:
        diff = config.baseline_acc - config.kl_acc
        print(f"KL 散度蒸馏差距: -{diff:.2f}%")

    print("-" * 60)

    if config.wasserstein_acc > config.kl_acc:
        diff = config.wasserstein_acc - config.kl_acc
        print(f"Wasserstein 优于 KL 散度: +{diff:.2f}%")
    elif config.kl_acc > config.wasserstein_acc:
        diff = config.kl_acc - config.wasserstein_acc
        print(f"KL 散度优于 Wasserstein: +{diff:.2f}%")
    else:
        print("Wasserstein 与 KL 散度性能相同")

    if config.teacher_acc > 0:
        gap_w = config.teacher_acc - config.wasserstein_acc
        gap_kl = config.teacher_acc - config.kl_acc
        print(f"Wasserstein 与教师差距: {gap_w:.2f}%")
        print(f"KL 散度与教师差距: {gap_kl:.2f}%")

    print("=" * 60)
