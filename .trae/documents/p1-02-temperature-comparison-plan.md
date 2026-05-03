# P1-02 温度参数对比实验 - 实施计划

## 实验概述

根据 `knowledge-distillation-practice-checklist.md` 中 P1-02 的定义，本实验系统对比不同温度参数 T 对知识蒸馏效果的影响。

**核心目标**：在 MNIST 数据集上，使用 P1-01 相同的教师/学生模型，固定 α=0.7，扫描 T ∈ {1, 2, 4, 8, 16, 20}，记录每个 T 值下的学生模型准确率，绘制 T-准确率曲线，并可视化不同 T 值下教师软标签的分布形态。

**预期结果**：T=4\~8 效果最佳；T=1 接近无蒸馏；T>20 效果下降。

***

## 文件结构

```
experiments/P1-02-temperature-comparison/
├── config.py          # 配置：温度扫描参数
├── models.py          # 复用 P1-01 的 TeacherNet / StudentNet
├── dataset.py         # 复用 P1-01 的 MNIST 数据加载
├── losses.py          # 复用 P1-01 的 DistillationLoss
├── utils.py           # 复用 P1-01 的 evaluate / get_device
├── train.py           # 温度扫描训练逻辑
├── evaluate.py        # 结果对比、绘图、软标签可视化
└── main.py            # 入口文件
```

***

## 实施步骤

### 步骤 1：创建文件夹 `experiments/P1-02-temperature-comparison/`

新建实验目录。

### 步骤 2：创建 `config.py`

基于 P1-01 的 Config，新增温度扫描相关配置：

```python
from dataclasses import dataclass, field

@dataclass
class Config:
    T_values: list = field(default_factory=lambda: [1, 2, 4, 8, 16, 20])
    alpha: float = 0.7
    learning_rate: float = 1e-3
    teacher_epochs: int = 20
    student_epochs: int = 10
    batch_size: int = 64
    test_batch_size: int = 1000
    dropout: float = 0.2
    data_dir: str = './data'
    mnist_mean: float = 0.1307
    mnist_std: float = 0.3081
    lr_step_size: int = 10
    lr_gamma: float = 0.5
    num_vis_samples: int = 5        # 软标签可视化采样数
    seed: int = 42                  # 可复现性
```

关键变化：

* 移除单一 `T` 字段，新增 `T_values` 列表用于温度扫描

* 新增 `num_vis_samples` 控制软标签可视化采样数

* 新增 `seed` 保证实验可复现

### 步骤 3：创建 `models.py`

直接复用 P1-01 的 `TeacherNet` 和 `StudentNet`，代码完全一致。

### 步骤 4：创建 `dataset.py`

直接复用 P1-01 的 `get_dataloaders`，代码完全一致。

### 步骤 5：创建 `losses.py`

直接复用 P1-01 的 `DistillationLoss`，代码完全一致。

### 步骤 6：创建 `utils.py`

复用 P1-01 的 `get_device` 和 `evaluate`，新增 `set_seed` 函数：

```python
def set_seed(seed):
    import torch, random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

### 步骤 7：创建 `train.py`

核心改动文件。包含以下函数：

1. **`train_teacher(model, train_loader, test_loader, device, config)`** — 与 P1-01 一致，训练教师模型
2. **`distill_with_temperature(teacher, train_loader, test_loader, device, T, config)`** — 对单个 T 值进行蒸馏训练，返回训练过程中每个 epoch 的准确率列表
3. **`train_student_baseline(model, train_loader, test_loader, device, config)`** — 与 P1-01 一致，训练基线学生模型
4. **`run_temperature_sweep(teacher, train_loader, test_loader, device, config)`** — 核心函数，遍历 `config.T_values` 中的每个 T，调用 `distill_with_temperature`，收集所有结果

关键设计：

* 每次蒸馏前重新初始化学生模型（保证公平对比）

* 记录每个 T 值下每个 epoch 的准确率（用于收敛速度分析）

* 教师模型只训练一次，所有温度实验共享

### 步骤 8：创建 `evaluate.py`

包含结果展示和可视化功能：

1. **`print_temperature_results(results, baseline_acc, teacher_acc)`** — 打印温度对比表格
2. **`plot_temperature_curve(results, baseline_acc, save_path)`** — 绘制 T-准确率曲线图
3. **`plot_convergence_curves(results, save_path)`** — 绘制不同 T 值下的训练收敛曲线（epoch-准确率）
4. **`visualize_soft_labels(teacher, dataloader, device, T_values, num_samples, save_dir)`** — 可视化不同 T 值下教师软标签的分布形态（选取若干样本，展示 softmax 输出的概率分布柱状图）

可视化细节：

* T-准确率曲线：x 轴为 T 值（对数刻度），y 轴为准确率，同时绘制基线水平线

* 收敛曲线：每条线对应一个 T 值，x 轴为 epoch，y 轴为准确率

* 软标签可视化：对每个样本，绘制多个子图，每个子图对应一个 T 值，展示 10 个类别的概率分布

### 步骤 9：创建 `main.py`

入口文件，编排完整实验流程：

```python
def main():
    config = Config()
    set_seed(config.seed)
    device = get_device()

    # 1. 加载数据
    train_loader, test_loader = get_dataloaders(config)

    # 2. 训练教师模型（只训练一次）
    teacher = TeacherNet(dropout=config.dropout).to(device)
    train_teacher(teacher, train_loader, test_loader, device, config)
    teacher_acc = evaluate(teacher, test_loader, device)

    # 3. 训练基线学生模型
    student_baseline = StudentNet().to(device)
    train_student_baseline(student_baseline, train_loader, test_loader, device, config)
    baseline_acc = evaluate(student_baseline, test_loader, device)

    # 4. 温度扫描实验
    results = run_temperature_sweep(teacher, train_loader, test_loader, device, config)

    # 5. 打印结果
    print_temperature_results(results, baseline_acc, teacher_acc)

    # 6. 绘制 T-准确率曲线
    plot_temperature_curve(results, baseline_acc, save_path='./results/temperature_curve.png')

    # 7. 绘制收敛曲线
    plot_convergence_curves(results, save_path='./results/convergence_curves.png')

    # 8. 可视化软标签分布
    visualize_soft_labels(teacher, test_loader, device, config.T_values,
                          config.num_vis_samples, save_dir='./results/soft_labels')
```

***

## 依赖关系

* PyTorch + torchvision（训练和数据处理）

* matplotlib（绘图）

* numpy（数值计算）

无需额外安装，与 P1-01 环境一致。

***

## 预期输出

1. **控制台输出**：每个 T 值的训练日志 + 最终对比表格
2. **`results/temperature_curve.png`**：T-准确率曲线
3. **`results/convergence_curves.png`**：不同 T 的收敛速度对比
4. **`results/soft_labels/`**：不同 T 值下教师软标签分布可视化

