def print_final_results(config):
    print("\n" + "=" * 60)
    print("实验结果总结 - CIFAR-100 多方法组合知识蒸馏")
    print("=" * 60)
    print(f"教师模型 (ResNet-32x4):          {config.teacher_acc:.2f}%")
    print(f"基线学生模型 (ResNet-8x4):       {config.baseline_acc:.2f}%")
    print("-" * 60)
    print(f"组合蒸馏 (R+F+A):                {config.combined_acc:.2f}%")
    print(f"Response-only 蒸馏:              {config.response_only_acc:.2f}%")
    print(f"Feature-only 蒸馏:               {config.feature_only_acc:.2f}%")
    print(f"Attention-only 蒸馏:             {config.attention_only_acc:.2f}%")
    print("-" * 60)

    if config.combined_acc > config.baseline_acc:
        improvement = config.combined_acc - config.baseline_acc
        print(f"组合蒸馏相比基线提升: +{improvement:.2f}%")
    else:
        diff = config.baseline_acc - config.combined_acc
        print(f"组合蒸馏相比基线差距: -{diff:.2f}%")

    best_single = max(config.response_only_acc, config.feature_only_acc, config.attention_only_acc)
    best_name = "Response"
    if config.feature_only_acc == best_single:
        best_name = "Feature"
    elif config.attention_only_acc == best_single:
        best_name = "Attention"

    if config.combined_acc > best_single:
        improvement = config.combined_acc - best_single
        print(f"组合蒸馏相比最佳单一方法({best_name})提升: +{improvement:.2f}%")
    else:
        diff = best_single - config.combined_acc
        print(f"组合蒸馏相比最佳单一方法({best_name})差距: -{diff:.2f}%")

    for name, acc in [("Response", config.response_only_acc), ("Feature", config.feature_only_acc), ("Attention", config.attention_only_acc)]:
        if acc > config.baseline_acc:
            improvement = acc - config.baseline_acc
            print(f"{name}-only 相比基线提升: +{improvement:.2f}%")
        else:
            diff = config.baseline_acc - acc
            print(f"{name}-only 相比基线差距: -{diff:.2f}%")

    if config.teacher_acc > 0:
        gap_teacher = config.teacher_acc - config.combined_acc
        print(f"组合蒸馏与教师差距: {gap_teacher:.2f}%")

    print("=" * 60)
