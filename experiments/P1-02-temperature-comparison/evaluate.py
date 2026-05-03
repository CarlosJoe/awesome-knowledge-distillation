import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def print_temperature_results(results, baseline_acc, teacher_acc):
    print("\n" + "=" * 60)
    print("温度参数对比实验结果")
    print("=" * 60)
    print(f"{'T':>6} | {'准确率 (%)':>10} | {'vs 基线':>10}")
    print("-" * 60)
    for T in sorted(results.keys()):
        acc = results[T]['final_acc']
        diff = acc - baseline_acc
        sign = '+' if diff >= 0 else ''
        print(f"{T:>6} | {acc:>10.2f} | {sign}{diff:>9.2f}")
    print("-" * 60)
    print(f"{'基线':>6} | {baseline_acc:>10.2f} | {'---':>10}")
    print(f"{'教师':>6} | {teacher_acc:>10.2f} | {'---':>10}")
    print("=" * 60)

    best_T = max(results, key=lambda t: results[t]['final_acc'])
    best_acc = results[best_T]['final_acc']
    print(f"\n最佳温度: T={best_T}, 准确率={best_acc:.2f}%")
    print(f"蒸馏提升: +{best_acc - baseline_acc:.2f}% (vs 基线)")


def plot_temperature_curve(results, baseline_acc, save_path='./results/temperature_curve.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    T_values = sorted(results.keys())
    accs = [results[T]['final_acc'] for T in T_values]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(T_values, accs, 'bo-', linewidth=2, markersize=8, label='蒸馏学生模型')
    ax.axhline(y=baseline_acc, color='r', linestyle='--', linewidth=1.5, label=f'基线学生 ({baseline_acc:.2f}%)')

    for T, acc in zip(T_values, accs):
        ax.annotate(f'{acc:.2f}%', (T, acc), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9)

    ax.set_xscale('log')
    ax.set_xticks(T_values)
    ax.set_xticklabels([str(t) for t in T_values])
    ax.set_xlabel('温度 T', fontsize=12)
    ax.set_ylabel('准确率 (%)', fontsize=12)
    ax.set_title('温度参数 T 对蒸馏效果的影响', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"T-准确率曲线已保存至: {save_path}")


def plot_convergence_curves(results, save_path='./results/convergence_curves.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(torch.linspace(0, 1, len(results)).tolist())

    for idx, T in enumerate(sorted(results.keys())):
        epoch_accs = results[T]['epoch_accs']
        epochs = list(range(1, len(epoch_accs) + 1))
        ax.plot(epochs, epoch_accs, '-o', color=colors[idx], linewidth=1.5,
                markersize=4, label=f'T={T}')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('准确率 (%)', fontsize=12)
    ax.set_title('不同温度 T 下的训练收敛曲线', fontsize=14)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"收敛曲线已保存至: {save_path}")


def visualize_soft_labels(teacher, dataloader, device, T_values, num_samples=5,
                          save_dir='./results/soft_labels'):
    os.makedirs(save_dir, exist_ok=True)
    teacher.eval()

    samples = []
    labels_true = []
    for images, labels in dataloader:
        samples.append(images)
        labels_true.append(labels)
        if len(samples) * images.size(0) >= num_samples:
            break

    images_batch = samples[0][:num_samples].to(device)
    labels_batch = labels_true[0][:num_samples]

    with torch.no_grad():
        logits = teacher(images_batch)

    class_names = [str(i) for i in range(10)]

    for sample_idx in range(num_samples):
        true_label = labels_batch[sample_idx].item()
        sample_logits = logits[sample_idx]

        fig, axes = plt.subplots(1, len(T_values), figsize=(4 * len(T_values), 3.5))
        if len(T_values) == 1:
            axes = [axes]

        for ax_idx, T in enumerate(T_values):
            probs = F.softmax(sample_logits / T, dim=0).cpu().numpy()
            bars = axes[ax_idx].bar(class_names, probs, color='steelblue', alpha=0.8)
            bars[true_label].set_color('red')
            axes[ax_idx].set_ylim(0, 1.0)
            axes[ax_idx].set_title(f'T={T}', fontsize=12)
            axes[ax_idx].set_xlabel('类别', fontsize=10)
            if ax_idx == 0:
                axes[ax_idx].set_ylabel('概率', fontsize=10)

        fig.suptitle(f'样本 #{sample_idx+1} (真实标签: {true_label}) 不同温度下的软标签分布',
                     fontsize=13, y=1.02)
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'soft_labels_sample_{sample_idx+1}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"软标签可视化已保存至: {save_dir}/")
