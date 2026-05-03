# Checklist

## 通用检查项（适用于所有实验）

- [x] 每个实验目录存在于 `experiments/` 下
- [x] 每个实验包含 8 个标准文件：config.py, models.py, losses.py, dataset.py, utils.py, train.py, evaluate.py, main.py
- [x] config.py 使用 dataclass 定义配置
- [x] 代码风格与 P1-01 一致（无多余注释、中文输出、模块化）
- [x] main.py 可直接运行 `python main.py`

## P1-03 检查项

- [x] α 扫描范围包含 {0.3, 0.5, 0.7, 0.9, 1.0}
- [x] 固定 T=4.0
- [x] 每个 α 值独立训练学生模型（重新初始化）
- [x] 输出 α-准确率对比表格
- [x] 绘制 α-准确率曲线图

## P2-01 检查项

- [x] 使用 ResNet-56（教师）和 ResNet-20（学生）
- [x] 使用 CIFAR-10 数据集（含数据增强）
- [x] 使用 SGD + CosineAnnealing 优化器
- [x] 输出教师/蒸馏学生/基线学生三者准确率对比

## P2-02 检查项

- [x] 实现 HintRegressor（1×1 卷积对齐通道）
- [x] 实现两阶段训练：Hint Training → KD Training
- [x] 模型支持 return_hint 参数返回中间层特征
- [x] 输出与纯 Response KD 的对比结果

## P2-03 检查项

- [x] 实现 MarginReLU（margin=-0.2）
- [x] 实现通道注意力加权
- [x] 使用 L1 损失替代 L2 损失
- [x] 模型支持多 stage 特征提取（return_features）
- [x] 输出与纯 Response KD 的对比结果

## P2-04 检查项

- [x] 实现注意力图计算（|A|^p, p=2）
- [x] 实现归一化 L2 注意力转移损失
- [x] 模型支持多层特征提取
- [x] 输出蒸馏结果和注意力图可视化

## P2-05 检查项

- [x] 使用 CIFAR-100 数据集
- [x] 实现距离蒸馏（pairwise distance + Huber loss）
- [x] 实现角度蒸馏（pairwise angle + Huber loss）
- [x] batch_size ≥ 64
- [x] 输出与纯 Response KD 的对比结果

## P2-06 检查项

- [x] 使用 CIFAR-100 数据集
- [x] 实现 Projector 投影头（MLP）
- [x] 实现 InfoNCE 对比损失
- [x] 使用 ResNet-32×4 / ResNet-8×4
- [x] 输出与纯 Response KD 的对比结果

## P2-07 检查项

- [x] 实现 z-score logits 标准化
- [x] 标准化后计算 KL 散度
- [x] 输出与标准 KD 的对比结果

## P2-08 检查项

- [x] 实现 Wasserstein 距离（排序后概率 L1 距离）
- [x] 输出与 KL 散度 KD 的对比结果

## P2-09 检查项

- [x] 实现两阶段渐进蒸馏：教师→助教→学生
- [x] 使用 ResNet-44 作为助教模型
- [x] 输出助教蒸馏 vs 直接蒸馏对比

## P2-10 检查项

- [x] 使用 CIFAR-100 数据集
- [x] 实现 Response + Feature + Attention 三重组合损失
- [x] 输出与单一方法的对比结果

## P3-01 检查项

- [x] 实现 SelfDistillationResNet（含 aux_fc1, aux_fc2）
- [x] 实现 SelfDistillationLoss（主 CE + 辅助 CE + 辅助 KD）
- [x] 深层输出 detach 作为教师信号
- [x] 输出自蒸馏 vs 直接训练基线对比

## P3-02 检查项

- [x] 实现迭代自蒸馏循环（3~5 代）
- [x] 同架构教师-学生（ResNet-20）
- [x] 输出逐代准确率对比

## P3-03 检查项

- [x] 实现伪标签生成
- [x] 实现噪声注入（Dropout + 随机深度 + 数据增强）
- [x] 实现迭代自训练循环
- [x] 输出迭代轮次结果

## P3-04 检查项

- [x] 实现 DeepInversion 图像反演生成
- [x] 实现 BN 统计量损失
- [x] 实现总变差和 L2 正则
- [x] 全程不使用原始训练数据
- [x] 输出生成图像可视化和蒸馏结果

## P3-05 检查项

- [x] 使用 SST-2 数据集
- [x] 实现 BERT-base → TinyBERT 蒸馏
- [x] 实现三层蒸馏损失（嵌入层 + Transformer 层 + 预测层）
- [x] 使用 HuggingFace Transformers 库
- [x] 输出蒸馏 vs 直接微调对比

## P3-06 检查项

- [x] 实现 DDPM UNet 模型
- [x] 实现渐进式蒸馏（128→64→32→16→8→4 步）
- [x] 学生一步模拟教师两步去噪
- [x] 输出每步 FID 指标

## P3-07 检查项

- [x] 实现 DistillationVisionTransformer（含 cls_token + dist_token）
- [x] 实现 DeiTDistillationLoss（0.5·CE + 0.5·KL）
- [x] 使用 CIFAR-100 作为简化替代数据集
- [x] 输出蒸馏 vs 无蒸馏 ViT 对比
