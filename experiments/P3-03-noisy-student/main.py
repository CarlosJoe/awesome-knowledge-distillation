from config import Config
from dataset import load_and_split_data
from train import iterative_training
from evaluate import print_iteration_results
from utils import get_device, set_seed


def main():
    config = Config()
    device = get_device()
    set_seed(42)
    print(f"使用设备: {device}")
    print(f"超参数: T={config.T}, α={config.alpha}, dropout={config.dropout}, "
          f"stochastic_depth_p={config.stochastic_depth_p}, iterations={config.num_iterations}")

    labeled_data, labeled_targets, unlabeled_data, unlabeled_targets, test_loader = load_and_split_data(config)
    print(f"有标签数据: {len(labeled_data)}, 无标签数据: {len(unlabeled_data)}")

    results = iterative_training(
        config, labeled_data, labeled_targets,
        unlabeled_data, unlabeled_targets, test_loader, device
    )

    print_iteration_results(results)


if __name__ == '__main__':
    main()
