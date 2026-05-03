from config import BornAgainConfig
from dataset import get_cifar10_loaders
from models import ResNet20
from train import train_generation_0, train_generation_k
from evaluate import print_generation_results
from utils import set_seed, get_device


def main():
    config = BornAgainConfig()

    print("=" * 60)
    print("Born Again 迭代自蒸馏实验")
    print("=" * 60)
    print(f"数据集: {config.dataset_name}")
    print(f"模型架构: ResNet-20 (同架构自蒸馏)")
    print(f"迭代代数: {config.num_generations}")
    print(f"温度 T={config.temperature}, lambda={config.lambda_kl}")
    print(f"损失函数: L = CE(y, p_student) + lambda * KL(p_teacher || p_student)")
    print(f"优化器: SGD(lr={config.learning_rate}, momentum={config.momentum}, wd={config.weight_decay})")
    print(f"调度器: CosineAnnealing, epochs={config.epochs}")
    print(f"设备: {get_device()}")

    set_seed(config.seed)
    device = get_device()

    train_loader, test_loader = get_cifar10_loaders(config)

    model, acc = train_generation_0(config, train_loader, test_loader, device)
    config.generation_accs.append(acc)

    for gen in range(1, config.num_generations):
        model, acc = train_generation_k(config, model, gen, train_loader, test_loader, device)
        config.generation_accs.append(acc)

    print_generation_results(config)


if __name__ == "__main__":
    main()
