def print_generation_results(config):
    print("\n" + "=" * 60)
    print("Born Again 迭代自蒸馏实验结果")
    print("=" * 60)

    header = f"{'代数':^10}{'准确率 (%)':^15}{'提升 (%)':^15}"
    print(header)
    print("-" * 40)

    prev_acc = 0.0
    for gen, acc in enumerate(config.generation_accs):
        if gen == 0:
            delta_str = "-"
        else:
            delta = acc - prev_acc
            delta_str = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
        print(f"Gen {gen:^5}{acc:^15.2f}{delta_str:^15}")
        prev_acc = acc

    print("-" * 40)

    if len(config.generation_accs) >= 2:
        total_improvement = config.generation_accs[-1] - config.generation_accs[0]
        print(f"总提升 (Gen 0 → Gen {len(config.generation_accs) - 1}): {total_improvement:+.2f}%")

    if len(config.generation_accs) >= 3:
        last_delta = config.generation_accs[-1] - config.generation_accs[-2]
        print(f"最后一代提升: {last_delta:+.2f}%")
        if abs(last_delta) < 0.5:
            print("→ 准确率已趋于收敛")

    print("=" * 60)
