def print_final_results(config):
    print("\n" + "=" * 60)
    print("实验结果总结 - CIFAR-10 FitNets 特征蒸馏")
    print("=" * 60)
    print(f"教师模型 (FitNetsTeacher):       {config.teacher_acc:.2f}%")
    print(f"Hint训练后学生模型:              {config.hint_acc:.2f}%")
    print(f"FitNets蒸馏学生模型:             {config.distilled_acc:.2f}%")
    print(f"基线学生模型 (FitNetsStudent):   {config.baseline_acc:.2f}%")
    print("-" * 60)

    if config.distilled_acc > config.baseline_acc:
        improvement = config.distilled_acc - config.baseline_acc
        print(f"蒸馏提升 (vs 基线): +{improvement:.2f}%")
    else:
        diff = config.baseline_acc - config.distilled_acc
        print(f"蒸馏差距 (vs 基线): -{diff:.2f}%")

    if config.teacher_acc > 0:
        gap_teacher = config.teacher_acc - config.distilled_acc
        print(f"与教师差距: {gap_teacher:.2f}%")

    print("\n--- FitNets 特征蒸馏 vs 纯 Response KD 对比 ---")
    print("FitNets 通过 Hint Training 让学生学习教师中间层特征表示，")
    print("再结合 Response KD 进行联合训练，预期提升 +0.2~0.5%")
    print(f"超参数: T={config.temperature}, α={config.alpha}, β={config.beta}")
    print(f"Hint阶段: {config.hint_epochs} epochs, KD阶段: {config.kd_epochs} epochs")
    print("=" * 60)
