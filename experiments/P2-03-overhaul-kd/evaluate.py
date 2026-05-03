def print_final_results(config):
    print("\n" + "=" * 60)
    print("实验结果总结 - CIFAR-10 Overhaul 特征蒸馏")
    print("=" * 60)
    print(f"教师模型 (ResNet-56):          {config.teacher_acc:.2f}%")
    print(f"Overhaul蒸馏学生 (ResNet-20):  {config.overhaul_acc:.2f}%")
    print(f"基线学生模型 (ResNet-20):      {config.baseline_acc:.2f}%")
    print("-" * 60)

    if config.overhaul_acc > config.baseline_acc:
        improvement = config.overhaul_acc - config.baseline_acc
        print(f"Overhaul蒸馏提升 (vs 基线): +{improvement:.2f}%")
    else:
        diff = config.baseline_acc - config.overhaul_acc
        print(f"Overhaul蒸馏差距 (vs 基线): -{diff:.2f}%")

    if config.teacher_acc > 0:
        gap_teacher = config.teacher_acc - config.overhaul_acc
        print(f"与教师差距: {gap_teacher:.2f}%")

    print("\n--- Overhaul 特征蒸馏 vs 纯 Response KD 对比 ---")
    print(f"纯 Response KD 准确率:         {config.response_kd_acc:.2f}%")
    print(f"Overhaul 特征蒸馏准确率:       {config.overhaul_acc:.2f}%")

    if config.overhaul_acc > config.response_kd_acc:
        improvement = config.overhaul_acc - config.response_kd_acc
        print(f"Overhaul vs Response KD 提升: +{improvement:.2f}%")
    else:
        diff = config.response_kd_acc - config.overhaul_acc
        print(f"Overhaul vs Response KD 差距: -{diff:.2f}%")

    print(f"\n超参数: T={config.temperature}, α={config.alpha}, β={config.beta}, margin={config.margin}")
    print("核心方法: Margin ReLU + 通道注意力 + L1 特征损失")
    print("预期提升: 相比纯 Response KD 提升 +1.5~2.0%")
    print("=" * 60)
