# Tasks

## 阶段一：入门级实验（P1-03）

- [x] Task 1: 实现 P1-03 蒸馏权重 α 对比实验
  - [x] 创建 `experiments/P1-03-alpha-comparison/` 目录
  - [x] 编写 `config.py`：α 扫描列表 {0.3, 0.5, 0.7, 0.9, 1.0}，固定 T=4.0
  - [x] 编写 `models.py`：复用 P1-01 的 TeacherNet / StudentNet
  - [x] 编写 `losses.py`：复用 P1-01 的 DistillationLoss
  - [x] 编写 `dataset.py`：复用 P1-01 的 MNIST 数据加载
  - [x] 编写 `utils.py`：复用 P1-01 工具函数 + set_seed
  - [x] 编写 `train.py`：α 扫描训练逻辑，每次蒸馏前重新初始化学生
  - [x] 编写 `evaluate.py`：α-准确率曲线绘制、对比表格输出
  - [x] 编写 `main.py`：入口文件

## 阶段二：进阶级实验（P2-01 ~ P2-10）

- [x] Task 2: 实现 P2-01 CIFAR-10 Response KD
  - [x] 创建 `experiments/P2-01-cifar10-response-kd/` 目录
  - [x] 编写 `config.py`：CIFAR-10 超参数（SGD, CosineAnnealing, epochs=100）
  - [x] 编写 `models.py`：ResNet BasicBlock + ResNet-56 / ResNet-20
  - [x] 编写 `losses.py`：DistillationLoss（同 P1-01）
  - [x] 编写 `dataset.py`：CIFAR-10 数据加载（含 RandomCrop/HorizontalFlip 增强）
  - [x] 编写 `utils.py`：evaluate + get_device + set_seed
  - [x] 编写 `train.py`：教师训练 + 基线学生训练 + 蒸馏训练
  - [x] 编写 `evaluate.py`：三者准确率对比
  - [x] 编写 `main.py`：入口文件

- [x] Task 3: 实现 P2-02 FitNets 特征蒸馏
  - [x] 创建 `experiments/P2-02-fitnets-feature-kd/` 目录
  - [x] 编写 `config.py`：FitNets 超参数（T=4.0, α=0.7, β=0.1）
  - [x] 编写 `models.py`：FitNetsTeacher / FitNetsStudent（含 hint_layer + return_hint）
  - [x] 编写 `losses.py`：FitNetsLoss（L_soft + L_hard + β·L_hint），HintRegressor
  - [x] 编写 `dataset.py`：CIFAR-10 数据加载
  - [x] 编写 `utils.py`：evaluate + get_device + set_seed
  - [x] 编写 `train.py`：两阶段训练（Hint Training → KD Training）
  - [x] 编写 `evaluate.py`：与纯 Response KD 对比
  - [x] 编写 `main.py`：入口文件

- [x] Task 4: 实现 P2-03 Overhaul KD 特征蒸馏
  - [x] 创建 `experiments/P2-03-overhaul-kd/` 目录
  - [x] 编写 `config.py`：Overhaul 超参数（margin=-0.2, β=0.1）
  - [x] 编写 `models.py`：ResNet-56 / ResNet-20（含多 stage 特征提取 return_features）
  - [x] 编写 `losses.py`：MarginReLU + OverhaulDistillationLoss（通道注意力 + L1）
  - [x] 编写 `dataset.py`：CIFAR-10 数据加载
  - [x] 编写 `utils.py`：evaluate + get_device + set_seed
  - [x] 编写 `train.py`：多 stage 特征蒸馏训练
  - [x] 编写 `evaluate.py`：与纯 Response KD 对比
  - [x] 编写 `main.py`：入口文件

- [x] Task 5: 实现 P2-04 Attention Transfer
  - [x] 创建 `experiments/P2-04-attention-transfer/` 目录
  - [x] 编写 `config.py`：AT 超参数（p=2, β=1e3）
  - [x] 编写 `models.py`：ResNet-56 / ResNet-20（含多层特征提取 return_features）
  - [x] 编写 `losses.py`：AttentionTransferLoss + ATDistillationLoss
  - [x] 编写 `dataset.py`：CIFAR-10 数据加载
  - [x] 编写 `utils.py`：evaluate + get_device + set_seed
  - [x] 编写 `train.py`：注意力蒸馏训练
  - [x] 编写 `evaluate.py`：蒸馏结果 + 注意力图可视化
  - [x] 编写 `main.py`：入口文件

- [x] Task 6: 实现 P2-05 Relational KD
  - [x] 创建 `experiments/P2-05-relational-kd/` 目录
  - [x] 编写 `config.py`：RKD 超参数（w_dist=25.0, w_angle=50.0, batch_size≥64）
  - [x] 编写 `models.py`：ResNet-56 / ResNet-20（CIFAR-100 版本，num_classes=100）
  - [x] 编写 `losses.py`：DistanceWiseRKD + AngleWiseRKD + RKDLoss
  - [x] 编写 `dataset.py`：CIFAR-100 数据加载
  - [x] 编写 `utils.py`：evaluate + get_device + set_seed
  - [x] 编写 `train.py`：RKD 蒸馏训练
  - [x] 编写 `evaluate.py`：与纯 Response KD 对比
  - [x] 编写 `main.py`：入口文件

- [x] Task 7: 实现 P2-06 CRD 对比蒸馏
  - [x] 创建 `experiments/P2-06-crd-contrastive-kd/` 目录
  - [x] 编写 `config.py`：CRD 超参数（crd_temp=0.07, n_negatives=16384）
  - [x] 编写 `models.py`：ResNet-32×4 / ResNet-8×4 + Projector 投影头
  - [x] 编写 `losses.py`：CRDLoss（InfoNCE 对比损失）
  - [x] 编写 `dataset.py`：CIFAR-100 数据加载
  - [x] 编写 `utils.py`：evaluate + get_device + set_seed
  - [x] 编写 `train.py`：CRD 蒸馏训练
  - [x] 编写 `evaluate.py`：与纯 Response KD 对比
  - [x] 编写 `main.py`：入口文件

- [x] Task 8: 实现 P2-07 Logit 标准化 KD
  - [x] 创建 `experiments/P2-07-logit-standardization-kd/` 目录
  - [x] 编写 `config.py`：Logit 标准化超参数
  - [x] 编写 `models.py`：ResNet-56 / ResNet-20
  - [x] 编写 `losses.py`：LogitStandardizedKD（z-score 标准化 + KL 散度）
  - [x] 编写 `dataset.py`：CIFAR-10 数据加载
  - [x] 编写 `utils.py`：evaluate + get_device + set_seed
  - [x] 编写 `train.py`：标准化 KD 训练
  - [x] 编写 `evaluate.py`：与标准 KD 对比
  - [x] 编写 `main.py`：入口文件

- [x] Task 9: 实现 P2-08 Wasserstein KD
  - [x] 创建 `experiments/P2-08-wasserstein-kd/` 目录
  - [x] 编写 `config.py`：Wasserstein KD 超参数
  - [x] 编写 `models.py`：ResNet-56 / ResNet-20
  - [x] 编写 `losses.py`：WassersteinKD（排序概率 L1 距离）
  - [x] 编写 `dataset.py`：CIFAR-10 数据加载
  - [x] 编写 `utils.py`：evaluate + get_device + set_seed
  - [x] 编写 `train.py`：Wasserstein KD 训练
  - [x] 编写 `evaluate.py`：与 KL 散度 KD 对比
  - [x] 编写 `main.py`：入口文件

- [x] Task 10: 实现 P2-09 助教蒸馏 (TA-KD)
  - [x] 创建 `experiments/P2-09-teacher-assistant-kd/` 目录
  - [x] 编写 `config.py`：TA-KD 超参数
  - [x] 编写 `models.py`：ResNet-56 / ResNet-44(助教) / ResNet-20
  - [x] 编写 `losses.py`：DistillationLoss（复用）
  - [x] 编写 `dataset.py`：CIFAR-10 数据加载
  - [x] 编写 `utils.py`：evaluate + get_device + set_seed
  - [x] 编写 `train.py`：两阶段蒸馏（教师→助教→学生）+ 直接蒸馏对比
  - [x] 编写 `evaluate.py`：助教蒸馏 vs 直接蒸馏对比
  - [x] 编写 `main.py`：入口文件

- [x] Task 11: 实现 P2-10 多方法组合实验
  - [x] 创建 `experiments/P2-10-combined-methods/` 目录
  - [x] 编写 `config.py`：组合超参数（α=0.5, β=0.1, γ=1e3）
  - [x] 编写 `models.py`：ResNet-32×4 / ResNet-8×4（含特征提取）
  - [x] 编写 `losses.py`：CombinedDistillationLoss（Response + Feature + Attention）
  - [x] 编写 `dataset.py`：CIFAR-100 数据加载
  - [x] 编写 `utils.py`：evaluate + get_device + set_seed
  - [x] 编写 `train.py`：多方法组合蒸馏训练
  - [x] 编写 `evaluate.py`：与单一方法对比
  - [x] 编写 `main.py`：入口文件

## 阶段三：高级实验（P3-01 ~ P3-07）

- [x] Task 12: 实现 P3-01 BYOT 自蒸馏
  - [x] 创建 `experiments/P3-01-byot-self-distillation/` 目录
  - [x] 编写 `config.py`：BYOT 超参数（α=0.5, β=0.3, γ=0.2, epochs=200）
  - [x] 编写 `models.py`：SelfDistillationResNet（含 aux_fc1, aux_fc2 辅助分类器）
  - [x] 编写 `losses.py`：SelfDistillationLoss（主 CE + 辅助 CE + 辅助 KD）
  - [x] 编写 `dataset.py`：CIFAR-10 数据加载
  - [x] 编写 `utils.py`：evaluate + get_device + set_seed
  - [x] 编写 `train.py`：自蒸馏训练 + 直接训练基线对比
  - [x] 编写 `evaluate.py`：自蒸馏 vs 直接训练对比
  - [x] 编写 `main.py`：入口文件

- [x] Task 13: 实现 P3-02 Born Again 迭代自蒸馏
  - [x] 创建 `experiments/P3-02-born-again/` 目录
  - [x] 编写 `config.py`：Born Again 超参数（λ=0.5, 迭代 3~5 代）
  - [x] 编写 `models.py`：ResNet-20（同架构师生）
  - [x] 编写 `losses.py`：BornAgainLoss（CE + λ·KL）
  - [x] 编写 `dataset.py`：CIFAR-10 数据加载
  - [x] 编写 `utils.py`：evaluate + get_device + set_seed
  - [x] 编写 `train.py`：迭代自蒸馏训练循环
  - [x] 编写 `evaluate.py`：逐代准确率对比
  - [x] 编写 `main.py`：入口文件

- [x] Task 14: 实现 P3-03 Noisy Student 自训练
  - [x] 创建 `experiments/P3-03-noisy-student/` 目录
  - [x] 编写 `config.py`：Noisy Student 超参数（Dropout=0.3, 随机深度=0.1, 迭代 3 轮）
  - [x] 编写 `models.py`：ResNet-56（教师）+ NoisyResNet-56（带噪声学生）
  - [x] 编写 `losses.py`：伪标签 CE 损失
  - [x] 编写 `dataset.py`：CIFAR-100 数据加载（含无标签子集模拟）
  - [x] 编写 `utils.py`：evaluate + get_device + set_seed + 伪标签生成
  - [x] 编写 `train.py`：教师训练 → 伪标签生成 → 噪声学生训练迭代
  - [x] 编写 `evaluate.py`：迭代轮次结果对比
  - [x] 编写 `main.py`：入口文件

- [x] Task 15: 实现 P3-04 DeepInversion 无数据蒸馏
  - [x] 创建 `experiments/P3-04-deepinversion/` 目录
  - [x] 编写 `config.py`：DeepInversion 超参数（iterations=2000, bn_weight=5.0, tv_weight=1e-4）
  - [x] 编写 `models.py`：ResNet-56 / ResNet-20
  - [x] 编写 `losses.py`：BN 统计量损失 + 总变差 + L2 正则
  - [x] 编写 `dataset.py`：无数据集加载（使用生成数据）
  - [x] 编写 `utils.py`：evaluate + get_device + set_seed + total_variation
  - [x] 编写 `train.py`：图像反演生成 + 无数据蒸馏训练
  - [x] 编写 `evaluate.py`：生成图像可视化 + 蒸馏结果
  - [x] 编写 `main.py`：入口文件

- [x] Task 16: 实现 P3-05 TinyBERT NLP 蒸馏
  - [x] 创建 `experiments/P3-05-tinybert-nlp/` 目录
  - [x] 编写 `config.py`：TinyBERT 超参数（α_pred=0.7, α_attn=0.1, α_hidn=0.1, α_embd=0.1）
  - [x] 编写 `models.py`：BERT-base 教师 + TinyBERT 学生配置
  - [x] 编写 `losses.py`：TinyBERTDistiller（嵌入层 + Transformer 层 + 预测层蒸馏）
  - [x] 编写 `dataset.py`：SST-2 数据加载（使用 HuggingFace datasets）
  - [x] 编写 `utils.py`：evaluate + get_device + set_seed
  - [x] 编写 `train.py`：教师微调 + TinyBERT 蒸馏训练
  - [x] 编写 `evaluate.py`：蒸馏 vs 直接微调对比
  - [x] 编写 `main.py`：入口文件

- [x] Task 17: 实现 P3-06 渐进式扩散蒸馏
  - [x] 创建 `experiments/P3-06-progressive-diffusion/` 目录
  - [x] 编写 `config.py`：渐进蒸馏超参数（步数计划 128→64→32→16→8→4）
  - [x] 编写 `models.py`：简化 DDPM UNet 模型
  - [x] 编写 `losses.py`：MSE 蒸馏损失
  - [x] 编写 `dataset.py`：CIFAR-10 数据加载
  - [x] 编写 `utils.py`：evaluate + get_device + set_seed + FID 计算
  - [x] 编写 `train.py`：教师 DDPM 训练 + 渐进式蒸馏
  - [x] 编写 `evaluate.py`：每步 FID 指标 + 生成样本可视化
  - [x] 编写 `main.py`：入口文件

- [x] Task 18: 实现 P3-07 DeiT 蒸馏 Token
  - [x] 创建 `experiments/P3-07-deit-distillation-token/` 目录
  - [x] 编写 `config.py`：DeiT 超参数（embed_dim=384, depth=12, num_heads=6）
  - [x] 编写 `models.py`：DistillationVisionTransformer（含 cls_token + dist_token）
  - [x] 编写 `losses.py`：DeiTDistillationLoss（0.5·CE + 0.5·KL）
  - [x] 编写 `dataset.py`：CIFAR-100 数据加载（作为 ImageNet 子集简化替代）
  - [x] 编写 `utils.py`：evaluate + get_device + set_seed
  - [x] 编写 `train.py`：DeiT 蒸馏训练 + 无蒸馏 ViT 基线对比
  - [x] 编写 `evaluate.py`：蒸馏 vs 无蒸馏对比
  - [x] 编写 `main.py`：入口文件

# Task Dependencies

- Task 1 (P1-03) 依赖 P1-01 已完成 ✅
- Task 2 (P2-01) 依赖 P1-01 已完成 ✅（ResNet 模型定义独立）
- Task 3 (P2-02) 依赖 Task 2（复用 ResNet 模型和 Response KD 基线）
- Task 4 (P2-03) 依赖 Task 2（复用 ResNet 模型，需多 stage 特征提取）
- Task 5 (P2-04) 依赖 Task 2（复用 ResNet 模型，需多层特征提取）
- Task 6 (P2-05) 依赖 Task 2（使用 CIFAR-100 版 ResNet）
- Task 7 (P2-06) 依赖 Task 2（使用 ResNet-32×4 / ResNet-8×4）
- Task 8 (P2-07) 依赖 Task 2（复用 ResNet 模型）
- Task 9 (P2-08) 依赖 Task 8（与 Logit 标准化 KD 对比）
- Task 10 (P2-09) 依赖 Task 2（复用 ResNet 模型）
- Task 11 (P2-10) 依赖 Task 3, Task 5（组合 Feature + Attention 方法）
- Task 12 (P3-01) 无直接依赖（自蒸馏不需要外部教师）
- Task 13 (P3-02) 依赖 Task 12（迭代自蒸馏基于自蒸馏概念）
- Task 14 (P3-03) 依赖 Task 2（伪标签生成基于 Response KD）
- Task 15 (P3-04) 依赖 Task 2（无数据蒸馏基于 Response KD 框架）
- Task 16 (P3-05) 无直接依赖（NLP 领域独立）
- Task 17 (P3-06) 无直接依赖（扩散模型独立）
- Task 18 (P3-07) 依赖 Task 2（蒸馏概念基础）

# 可并行化的任务

以下任务组可以并行实现：
- 组1: Task 1 (P1-03), Task 2 (P2-01), Task 12 (P3-01), Task 16 (P3-05), Task 17 (P3-06) — 无互相依赖
- 组2: Task 3~6, Task 8, Task 10 — 依赖 Task 2 完成后可并行
- 组3: Task 7, Task 11 — 依赖特定模型定义
- 组4: Task 9 — 依赖 Task 8
- 组5: Task 13 — 依赖 Task 12
- 组6: Task 14, Task 15, Task 18 — 依赖 Task 2
