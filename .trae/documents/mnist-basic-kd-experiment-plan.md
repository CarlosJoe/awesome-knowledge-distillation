# MNIST 基础蒸馏实验 (P1-01) 实施计划

## 概述

根据 `knowledge-distillation-practice-checklist.md` 中 P1-01 的规格，新建文件夹并实现 MNIST 基础知识蒸馏实验。核心方法为 Response-based KD (Hinton 2015)，使用全连接网络在 MNIST 数据集上进行蒸馏。

## 实验规格（来自清单）

| 字段 | 内容 |
|------|------|
| 教师模型 | TeacherNet：FC(784→1200) → FC(1200→1200) → FC(1200→10)，带 Dropout(0.2) |
| 学生模型 | StudentNet：FC(784→800) → FC(800→10) |
| 核心蒸馏方法 | Response-based KD（Hinton 2015） |
| 损失函数 | L = α·T²·KL(p_t^T \|\| p_s^T) + (1-α)·CE(y, p_s^1) |
| 关键超参数 | T=4.0, α=0.7, optimizer=Adam(lr=1e-3), epochs=10 |
| 评估指标 | Top-1 准确率 (%) |
| 预期结果 | 教师 ~99.3%，学生(蒸馏) ~99.1%，学生(基线) ~98.5%，蒸馏提升 +0.6% |

## 文件结构

```
experiments/P1-01-mnist-basic-kd/
├── models.py          # 教师模型和学生模型定义
├── losses.py          # 蒸馏损失函数
├── dataset.py         # MNIST 数据集加载
├── train.py           # 训练逻辑（教师训练、学生蒸馏训练、学生基线训练）
├── evaluate.py        # 评估与结果对比
├── utils.py           # 工具函数（设备检测、日志等）
├── main.py            # 主入口，串联完整实验流程
└── config.py          # 超参数配置
```

## 实施步骤

### 步骤 1：创建文件夹 `experiments/P1-01-mnist-basic-kd/`

在项目根目录下创建 `experiments/P1-01-mnist-basic-kd/` 文件夹。

### 步骤 2：实现 `config.py` — 超参数配置

定义实验所有超参数的配置类：

- `T = 4.0`（温度参数）
- `alpha = 0.7`（蒸馏权重）
- `learning_rate = 1e-3`
- `epochs = 10`
- `batch_size = 64`
- `test_batch_size = 1000`
- `dropout = 0.2`
- 数据路径 `data_dir = './data'`
- MNIST 标准化参数 `mean=0.1307, std=0.3081`

### 步骤 3：实现 `models.py` — 模型定义

按照清单规格实现两个模型：

**TeacherNet**：
- `fc1`: Linear(784, 1200) + ReLU + Dropout(0.2)
- `fc2`: Linear(1200, 1200) + ReLU + Dropout(0.2)
- `fc3`: Linear(1200, 10)

**StudentNet**：
- `fc1`: Linear(784, 800) + ReLU
- `fc2`: Linear(800, 10)

### 步骤 4：实现 `losses.py` — 蒸馏损失函数

实现 `DistillationLoss` 类：

- `forward(student_logits, teacher_logits, labels)`：
  - 学生 soft：`F.log_softmax(student_logits / T, dim=1)`
  - 教师 soft：`F.softmax(teacher_logits / T, dim=1)`
  - 软损失：`F.kl_div(student_soft, teacher_soft, reduction='batchmean') * T²`
  - 硬损失：`F.cross_entropy(student_logits, labels)`
  - 总损失：`alpha * loss_soft + (1 - alpha) * loss_hard`

### 步骤 5：实现 `dataset.py` — 数据集加载

- 使用 `torchvision.datasets.MNIST` 加载数据
- 应用 `transforms.ToTensor()` + `transforms.Normalize((0.1307,), (0.3081,))`
- 返回 train_loader 和 test_loader

### 步骤 6：实现 `utils.py` — 工具函数

- `get_device()`: 检测并返回可用设备（CUDA/CPU）
- `evaluate(model, dataloader, device)`: 评估模型准确率

### 步骤 7：实现 `train.py` — 训练逻辑

实现三个训练函数：

1. **`train_teacher(model, train_loader, test_loader, device, config)`**：
   - 使用 CrossEntropy 损失
   - Adam 优化器，lr=1e-3
   - 每 epoch 打印 loss 和 accuracy

2. **`distill_train(teacher, student, train_loader, test_loader, device, config)`**：
   - 教师模型设为 eval 模式，不更新梯度
   - 使用 DistillationLoss
   - Adam 优化器，lr=1e-3
   - 每 epoch 打印 loss 和 accuracy

3. **`train_student_baseline(model, train_loader, test_loader, device, config)`**：
   - 使用 CrossEntropy 损失
   - Adam 优化器，lr=1e-3
   - 每 epoch 打印 loss 和 accuracy

### 步骤 8：实现 `evaluate.py` — 评估与结果对比

- `print_final_results(teacher, student_kd, student_baseline, test_loader, device)`：
  - 输出三个模型的最终准确率
  - 计算蒸馏提升量

### 步骤 9：实现 `main.py` — 主入口

串联完整实验流程：

1. 加载配置
2. 准备数据
3. 训练教师模型
4. 蒸馏训练学生模型
5. 直接训练学生模型（基线）
6. 输出最终结果对比

### 步骤 10：运行实验验证

执行 `python main.py`，确认：
- 教师模型准确率 ~99.3%
- 蒸馏学生模型准确率 ~99.1%
- 基线学生模型准确率 ~98.5%
- 蒸馏提升 +0.6%

## 参考代码来源

代码实现基于 `stage1-knowledge-distillation-fundamentals.md` 第 5.1~5.2 节的参考代码，按照模块化结构进行拆分和优化。
