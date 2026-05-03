from utils import evaluate


def print_results(sd_model, baseline_model, test_loader, device):
    sd_acc = evaluate(sd_model, test_loader, device)
    baseline_acc = evaluate(baseline_model, test_loader, device)

    print("\n" + "=" * 50)
    print("BYOT 自蒸馏实验结果对比")
    print("=" * 50)
    print(f"自蒸馏模型准确率:     {sd_acc:.2f}%")
    print(f"直接训练模型准确率:   {baseline_acc:.2f}%")
    print(f"自蒸馏提升:          +{sd_acc - baseline_acc:.2f}%")
    print("=" * 50)
