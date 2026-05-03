from config import Config
from models import SelfDistillationResNet
from dataset import get_dataloaders
from train import train_self_distillation, train_baseline
from evaluate import print_results
from utils import get_device, set_seed


def main():
    config = Config()
    set_seed(config.seed)
    device = get_device()
    print(f"使用设备: {device}")

    train_loader, test_loader = get_dataloaders(config)

    sd_model = SelfDistillationResNet(num_classes=config.num_classes).to(device)
    baseline_model = SelfDistillationResNet(num_classes=config.num_classes).to(device)

    print("=== BYOT 自蒸馏训练 (T=4.0, α=0.5, β=0.3, γ=0.2) ===")
    train_self_distillation(sd_model, train_loader, test_loader, device, config)

    print("\n=== 直接训练基线 ===")
    train_baseline(baseline_model, train_loader, test_loader, device, config)

    print_results(sd_model, baseline_model, test_loader, device)


if __name__ == '__main__':
    main()
