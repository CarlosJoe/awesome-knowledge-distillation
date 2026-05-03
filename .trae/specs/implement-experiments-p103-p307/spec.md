# 实现 P1-03 至 P3-07 实验案例 Spec

## Why

项目已完成 P1-01（MNIST 基础蒸馏）和 P1-02（温度参数对比）的实验实现，需要继续实现知识蒸馏实践清单中从 P1-03 到 P3-07 的全部 18 个实验案例，为学习者提供从入门到高级的完整实践代码。

## What Changes

- 新建 18 个实验文件夹，每个位于 `experiments/` 目录下
- 每个实验包含完整的代码文件：`config.py`, `models.py`, `losses.py`, `dataset.py`, `utils.py`, `train.py`, `evaluate.py`, `main.py`
- 遵循 P1-01 已建立的代码结构模式（dataclass 配置、模块化拆分、中文输出）
- 实验列表：
  - P1-03：蒸馏权重 α 对比实验（MNIST）
  - P2-01：CIFAR-10 Response KD
  - P2-02：FitNets 特征蒸馏
  - P2-03：Overhaul KD 特征蒸馏
  - P2-04：Attention Transfer
  - P2-05：Relational KD
  - P2-06：CRD 对比蒸馏
  - P2-07：Logit 标准化 KD
  - P2-08：Wasserstein KD
  - P2-09：助教蒸馏 (TA-KD)
  - P2-10：多方法组合实验
  - P3-01：BYOT 自蒸馏
  - P3-02：Born Again 迭代自蒸馏
  - P3-03：Noisy Student 自训练
  - P3-04：DeepInversion 无数据蒸馏
  - P3-05：TinyBERT NLP 蒸馏
  - P3-06：渐进式扩散蒸馏
  - P3-07：DeiT 蒸馏 Token

## Impact

- Affected specs: 无已有 spec 受影响
- Affected code: 仅新增文件，不修改已有代码

## ADDED Requirements

### Requirement: P1-03 蒸馏权重 α 对比实验

系统 SHALL 提供 `experiments/P1-03-alpha-comparison/` 目录下的完整实验代码，在 MNIST 数据集上使用 P1-01 相同的教师/学生模型，固定 T=4.0，扫描 α ∈ {0.3, 0.5, 0.7, 0.9, 1.0}，记录每个 α 值下的学生模型准确率，绘制 α-准确率曲线。

#### Scenario: α 扫描实验成功运行
- **WHEN** 用户运行 `python main.py`
- **THEN** 系统训练教师模型一次，对每个 α 值分别蒸馏训练学生模型，输出对比表格和 α-准确率曲线图

### Requirement: P2-01 CIFAR-10 Response KD

系统 SHALL 提供 `experiments/P2-01-cifar10-response-kd/` 目录下的完整实验代码，在 CIFAR-10 上实现 ResNet-56(教师) → ResNet-20(学生) 的 Response-based KD。

#### Scenario: CIFAR-10 蒸馏实验成功运行
- **WHEN** 用户运行 `python main.py`
- **THEN** 系统训练 ResNet-56 教师、ResNet-20 基线学生和蒸馏学生，输出三者准确率对比

### Requirement: P2-02 FitNets 特征蒸馏

系统 SHALL 提供 `experiments/P2-02-fitnets-feature-kd/` 目录下的完整实验代码，实现 FitNets 两阶段训练（Hint Training + KD Training），包含 HintRegressor 回归器。

#### Scenario: FitNets 蒸馏实验成功运行
- **WHEN** 用户运行 `python main.py`
- **THEN** 系统先进行 Hint Training 对齐中间层特征，再进行 KD Training，输出与纯 Response KD 的对比结果

### Requirement: P2-03 Overhaul KD 特征蒸馏

系统 SHALL 提供 `experiments/P2-03-overhaul-kd/` 目录下的完整实验代码，实现 Margin ReLU + L1 损失 + 通道注意力的 Overhaul KD。

#### Scenario: Overhaul KD 实验成功运行
- **WHEN** 用户运行 `python main.py`
- **THEN** 系统在多 stage 特征上应用 Margin ReLU 变换和通道注意力加权的 L1 损失，输出蒸馏结果

### Requirement: P2-04 Attention Transfer

系统 SHALL 提供 `experiments/P2-04-attention-transfer/` 目录下的完整实验代码，实现基于激活的注意力转移，包含注意力图计算和归一化 L2 损失。

#### Scenario: Attention Transfer 实验成功运行
- **WHEN** 用户运行 `python main.py`
- **THEN** 系统在多个层计算注意力图并转移，输出蒸馏结果和注意力图可视化

### Requirement: P2-05 Relational KD

系统 SHALL 提供 `experiments/P2-05-relational-kd/` 目录下的完整实验代码，在 CIFAR-100 上实现 RKD 的距离蒸馏 + 角度蒸馏。

#### Scenario: RKD 实验成功运行
- **WHEN** 用户运行 `python main.py`
- **THEN** 系统计算样本对距离和三元组角度的 Huber 损失，输出蒸馏结果

### Requirement: P2-06 CRD 对比蒸馏

系统 SHALL 提供 `experiments/P2-06-crd-contrastive-kd/` 目录下的完整实验代码，在 CIFAR-100 上实现 InfoNCE + 投影头的对比蒸馏。

#### Scenario: CRD 实验成功运行
- **WHEN** 用户运行 `python main.py`
- **THEN** 系统通过投影头对齐师生表示空间，使用 InfoNCE 损失进行对比蒸馏，输出蒸馏结果

### Requirement: P2-07 Logit 标准化 KD

系统 SHALL 提供 `experiments/P2-07-logit-standardization-kd/` 目录下的完整实验代码，实现先对 logits 做 z-score 标准化再计算 KL 散度的蒸馏方法。

#### Scenario: Logit 标准化 KD 实验成功运行
- **WHEN** 用户运行 `python main.py`
- **THEN** 系统对教师和学生 logits 标准化后计算蒸馏损失，输出与标准 KD 的对比结果

### Requirement: P2-08 Wasserstein KD

系统 SHALL 提供 `experiments/P2-08-wasserstein-kd/` 目录下的完整实验代码，实现用 Wasserstein 距离替代 KL 散度的蒸馏方法。

#### Scenario: Wasserstein KD 实验成功运行
- **WHEN** 用户运行 `python main.py`
- **THEN** 系统使用排序后的概率分布 L1 距离作为蒸馏损失，输出与 KL 散度 KD 的对比结果

### Requirement: P2-09 助教蒸馏 (TA-KD)

系统 SHALL 提供 `experiments/P2-09-teacher-assistant-kd/` 目录下的完整实验代码，实现两阶段渐进蒸馏：教师→助教→学生。

#### Scenario: TA-KD 实验成功运行
- **WHEN** 用户运行 `python main.py`
- **THEN** 系统先蒸馏训练助教模型，再用助教蒸馏训练学生模型，输出与直接蒸馏的对比结果

### Requirement: P2-10 多方法组合实验

系统 SHALL 提供 `experiments/P2-10-combined-methods/` 目录下的完整实验代码，在 CIFAR-100 上实现 Response + Feature + Attention 三种蒸馏方法的组合。

#### Scenario: 多方法组合实验成功运行
- **WHEN** 用户运行 `python main.py`
- **THEN** 系统同时使用多种蒸馏损失训练学生模型，输出与单一方法的对比结果

### Requirement: P3-01 BYOT 自蒸馏

系统 SHALL 提供 `experiments/P3-01-byot-self-distillation/` 目录下的完整实验代码，实现带辅助分类器的层间自蒸馏。

#### Scenario: BYOT 实验成功运行
- **WHEN** 用户运行 `python main.py`
- **THEN** 系统使用深层输出作为教师信号指导浅层辅助分类器，输出与直接训练基线的对比结果

### Requirement: P3-02 Born Again 迭代自蒸馏

系统 SHALL 提供 `experiments/P3-02-born-again/` 目录下的完整实验代码，实现同架构模型的迭代自蒸馏。

#### Scenario: Born Again 实验成功运行
- **WHEN** 用户运行 `python main.py`
- **THEN** 系统迭代训练多代模型，每代使用上一代作为教师，输出逐代准确率对比

### Requirement: P3-03 Noisy Student 自训练

系统 SHALL 提供 `experiments/P3-03-noisy-student/` 目录下的完整实验代码，实现伪标签生成 + 噪声注入的自训练方法。

#### Scenario: Noisy Student 实验成功运行
- **WHEN** 用户运行 `python main.py`
- **THEN** 系统用教师生成伪标签，用带噪声的学生训练，输出迭代轮次的结果

### Requirement: P3-04 DeepInversion 无数据蒸馏

系统 SHALL 提供 `experiments/P3-04-deepinversion/` 目录下的完整实验代码，实现从教师模型 BN 统计量反演生成数据并蒸馏。

#### Scenario: DeepInversion 实验成功运行
- **WHEN** 用户运行 `python main.py`
- **THEN** 系统从随机噪声反演生成图像，用生成图像训练学生模型，输出蒸馏结果

### Requirement: P3-05 TinyBERT NLP 蒸馏

系统 SHALL 提供 `experiments/P3-05-tinybert-nlp/` 目录下的完整实验代码，在 SST-2 上实现 BERT-base → TinyBERT 的三层蒸馏。

#### Scenario: TinyBERT 实验成功运行
- **WHEN** 用户运行 `python main.py`
- **THEN** 系统对齐嵌入层、Transformer 层和预测层，输出蒸馏结果

### Requirement: P3-06 渐进式扩散蒸馏

系统 SHALL 提供 `experiments/P3-06-progressive-diffusion/` 目录下的完整实验代码，在 CIFAR-10 上实现 DDPM 步数压缩的渐进式蒸馏。

#### Scenario: 渐进式扩散蒸馏实验成功运行
- **WHEN** 用户运行 `python main.py`
- **THEN** 系统逐步将采样步数从 128 压缩到 4，输出每步的 FID 指标

### Requirement: P3-07 DeiT 蒸馏 Token

系统 SHALL 提供 `experiments/P3-07-deit-distillation-token/` 目录下的完整实验代码，实现带蒸馏 Token 的 Vision Transformer。

#### Scenario: DeiT 实验成功运行
- **WHEN** 用户运行 `python main.py`
- **THEN** 系统使用分类 Token + 蒸馏 Token 并行训练，输出与无蒸馏 ViT 的对比结果

## MODIFIED Requirements

无修改需求。

## REMOVED Requirements

无移除需求。
