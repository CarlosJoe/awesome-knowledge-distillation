from config import Config
from models import ResNet56, DistillationVisionTransformer, VisionTransformer
from dataset import get_dataloaders
from train import train_teacher, train_deit_distill, train_deit_baseline
from evaluate import print_final_results
from utils import get_device, set_seed


def main():
    config = Config()
    set_seed(config.seed)
    device = get_device()
    print(f"使用设备: {device}")

    train_loader, test_loader = get_dataloaders(config)

    teacher = ResNet56(num_classes=config.num_classes).to(device)
    deit_distill = DistillationVisionTransformer(
        img_size=config.image_size,
        patch_size=config.patch_size,
        embed_dim=config.embed_dim,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        num_classes=config.num_classes,
        dropout=config.dropout,
    ).to(device)
    deit_baseline = VisionTransformer(
        img_size=config.image_size,
        patch_size=config.patch_size,
        embed_dim=config.embed_dim,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        num_classes=config.num_classes,
        dropout=config.dropout,
    ).to(device)

    print("=== 训练教师模型 (ResNet-56) ===")
    train_teacher(teacher, train_loader, test_loader, device, config)

    print("\n=== 蒸馏训练 DeiT-Small + 蒸馏Token (T=4, α=0.5) ===")
    train_deit_distill(teacher, deit_distill, train_loader, test_loader, device, config)

    print("\n=== 直接训练 DeiT-Small 无蒸馏（基线） ===")
    train_deit_baseline(deit_baseline, train_loader, test_loader, device, config)

    print_final_results(teacher, deit_distill, deit_baseline, test_loader, device)


if __name__ == '__main__':
    main()
