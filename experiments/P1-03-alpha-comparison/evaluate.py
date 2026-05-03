import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import Config


def print_alpha_results(alpha_results, baseline_acc, teacher_acc, config=Config()):
    print("\n" + "=" * 60)
    print("α 蒸馏权重对比实验结果")
    print("=" * 60)
    print(f"教师模型准确率:       {teacher_acc:.2f}%")
    print(f"学生基线准确率:       {baseline_acc:.2f}%")
    print("-" * 60)
    print(f"{'α':>6s}  {'蒸馏准确率':>10s}  {'vs 基线':>10s}")
    print("-" * 60)
    for alpha in config.alpha_values:
        acc = alpha_results[alpha]
        diff = acc - baseline_acc
        sign = '+' if diff >= 0 else ''
        print(f"{alpha:>6.1f}  {acc:>10.2f}%  {sign}{diff:>9.2f}%")
    print("=" * 60)

    best_alpha = max(alpha_results, key=alpha_results.get)
    print(f"最佳 α 值: {best_alpha} (准确率: {alpha_results[best_alpha]:.2f}%)")


def plot_alpha_curve(alpha_results, baseline_acc, teacher_acc, config=Config()):
    os.makedirs(config.results_dir, exist_ok=True)

    alphas = config.alpha_values
    accs = [alpha_results[a] for a in alphas]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, accs, 'bo-', linewidth=2, markersize=8, label='蒸馏学生模型')
    ax.axhline(y=baseline_acc, color='r', linestyle='--', linewidth=1.5, label=f'学生基线 ({baseline_acc:.2f}%)')
    ax.axhline(y=teacher_acc, color='g', linestyle='--', linewidth=1.5, label=f'教师模型 ({teacher_acc:.2f}%)')

    ax.set_xlabel('α (蒸馏损失权重)', fontsize=12)
    ax.set_ylabel('准确率 (%)', fontsize=12)
    ax.set_title(f'α 对蒸馏准确率的影响 (T={config.T})', fontsize=14)
    ax.set_xticks(alphas)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    for alpha, acc in zip(alphas, accs):
        ax.annotate(f'{acc:.2f}%', (alpha, acc), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(config.results_dir, 'alpha_curve.png')
    fig.savefig(save_path, dpi=150)
    print(f"曲线图已保存至: {save_path}")
    plt.close(fig)
