# 知识蒸馏（Knowledge Distillation）学习路线图

---

## 目录

1. [概述](#概述)
2. [第一阶段：理解核心概念（入门）](#第一阶段理解核心概念入门)
3. [第二阶段：掌握经典方法（进阶）](#第二阶段掌握经典方法进阶)
4. [第三阶段：深入专题方向（高级）](#第三阶段深入专题方向高级)
5. [第四阶段：动手实践](#第四阶段动手实践)
6. [第五阶段：阅读综述，建立全局视野](#第五阶段阅读综述建立全局视野)
7. [学习建议](#学习建议)

---

## 概述

这是一个非常丰富的知识蒸馏（Knowledge Distillation）资源仓库，包含了从1990年到2026年的160+篇论文、视频教程和多框架代码实现。

知识蒸馏是模型压缩领域的核心技术，旨在将大型模型（教师）的知识迁移到小型模型（学生）中，使小模型在保持高效推理速度的同时接近大模型的性能。

---

## 第一阶段：理解核心概念（入门）

### 必读奠基论文（按时间顺序）

| 年份 | 论文 | 说明 | 链接 |
|------|------|------|------|
| 1990 | Neural Network Ensembles | 理解"集成模型"的思想，这是知识蒸馏的根源 | [PDF](https://www.researchgate.net/publication/3191841_Neural_Network_Ensembles) |
| 2006 | Model Compression | Caruana 的模型压缩工作，将大模型/集成模型压缩为小模型的先驱 | [PDF](http://www.cs.cornell.edu/~caruana/compression.kdd06.pdf) |
| 2014 | Dark Knowledge | Hinton 首次提出"暗知识"概念 | [PDF](http://www.ttic.edu/dl/dark14.pdf) |
| 2015 | **Distilling the Knowledge in a Neural Network** | ⭐ **最核心的论文！** Hinton 等人正式提出知识蒸馏框架（软标签 + 温度参数），必读中的必读 | [PDF](https://arxiv.org/pdf/1503.02531.pdf) |

### 必看视频

| 视频 | 说明 | 链接 |
|------|------|------|
| Dark Knowledge | Geoffrey Hinton 亲自讲解 | [YouTube](https://www.youtube.com/watch?v=EK61htlw8hY) |
| Model Compression | Rich Caruana 讲解模型压缩 | [YouTube](https://www.youtube.com/watch?v=0WZmuryQdgg) |

---

## 第二阶段：掌握经典方法（进阶）

按**蒸馏对象**分类阅读：

| 类别 | 核心论文 | 关键思想 | 链接 |
|------|---------|---------|------|
| **基于响应（Response-based）** | Hinton 2015 | 蒸馏教师模型的输出 logits/软标签 | [PDF](https://arxiv.org/pdf/1503.02531.pdf) |
| **基于特征（Feature-based）** | FitNets | 蒸馏中间层特征（hint），学生模仿教师的隐藏层表示 | [PDF](https://arxiv.org/pdf/1412.6550) |
| **基于注意力（Attention-based）** | Attention Transfer | 转移注意力图，让学生关注教师关注的区域 | [PDF](https://arxiv.org/pdf/1612.03928) |
| **基于关系（Relation-based）** | Relational KD | 蒸馏样本间的关系结构，而非单个样本的特征 | [arXiv](https://arxiv.org/abs/1904.05068) |
| **对比蒸馏** | CRD | 用对比学习框架进行知识蒸馏 | [PDF](https://arxiv.org/pdf/1910.10699.pdf) |
| **特征蒸馏全面改进** | A Comprehensive Overhaul of Feature Distillation | 对特征蒸馏的系统性改进 | [arXiv](https://arxiv.org/abs/1904.01866) |

### 经典方法分类详解

#### 基于响应的蒸馏

最传统的方法，直接让学生学习教师的输出概率分布。

**代表论文**：Hinton 2015
- **核心思想**：用教师的软标签（概率分布）代替硬标签（one-hot）来训练学生
- **关键技巧**：温度参数 T，用于平滑概率分布，让暗知识变得可见
- **损失函数**：KL 散度 + 交叉熵的加权组合

#### 基于特征的蒸馏

不仅让学生模仿教师的输出，还让学生模仿教师的中间层特征。

**代表论文**：FitNets (2015)
- **核心思想**：让学生模仿教师隐藏层的输出
- **方法**：引入"提示层"（hint layer），让学生中间层回归教师中间层
- **优点**：能传递更丰富的知识，不仅仅是最终预测

#### 基于注意力的蒸馏

让学生学习教师的注意力图。

**代表论文**：Attention Transfer (2016)
- **核心思想**：神经网络的注意力图包含了"关注哪里"的信息
- **方法**：让学生和教师的注意力图尽可能相似
- **应用场景**：特别适合图像分类和目标检测

#### 基于关系的蒸馏

关注样本之间的关系，而非单个样本的输出。

**代表论文**：Relational KD (2019)
- **核心思想**：教师模型对样本之间的关系建模，学生也应该学到
- **方法**：比较样本对之间的距离结构
- **优点**：不依赖具体的输出形式，更通用

---

## 第三阶段：深入专题方向（高级）

根据你的兴趣选择专题：

### 方向A：自蒸馏（Self-Distillation）

不需要外部教师，模型自己教自己。

| 论文 | 年份 | 说明 | 链接 |
|------|------|------|------|
| Be Your Own Teacher | 2019 | 自己当自己的教师 | [arXiv](https://arxiv.org/abs/1905.08094) |
| Born Again Neural Networks | 2018 | 迭代式自蒸馏 | [arXiv](https://arxiv.org/abs/1805.04770) |
| Self-training with Noisy Student | 2019 | Google 的经典工作 | [arXiv](https://arxiv.org/abs/1911.04252) |

**核心思想**：用当前模型或之前迭代的模型作为教师，训练学生模型。

**优势**：
- 不需要预训练的大教师模型
- 可以提升模型性能而不增加推理成本
- 适合资源受限的场景

### 方向B：无数据蒸馏（Data-Free KD）

不需要原始训练数据。

| 论文 | 年份 | 说明 | 链接 |
|------|------|------|------|
| Data-Free Knowledge Distillation | 2017 | 无数据蒸馏先驱 | [链接](http://raphagl.com/research/replayed-distillation/) |
| Data-Free Learning of Student Networks | 2019 | 利用生成数据 | [PDF](https://arxiv.org/pdf/1904.01186.pdf) |
| Dreaming to Distill: DeepInversion | 2020 | 通过深度反演生成数据 | [arXiv](https://arxiv.org/abs/1912.08795) |

**核心思想**：当无法访问原始训练数据时，通过生成样本或使用其他数据来训练学生模型。

**方法**：
- 使用生成对抗网络生成样本
- 使用深度反演（DeepInversion）从教师模型反演生成训练数据
- 使用知识图谱或其他外部数据源

### 方向C：NLP/LLM 蒸馏

| 论文 | 年份 | 说明 | 链接 |
|------|------|------|------|
| TinyBERT | 2019 | BERT 蒸馏经典 | [PDF](https://arxiv.org/pdf/1909.10351.pdf) |
| Distilling BERT into Simple Neural Networks | 2019 | BERT 压缩 | [arXiv](https://arxiv.org/abs/1903.12136) |
| Causal Distillation for Language Models | 2021 | 因果蒸馏 | [arXiv](https://arxiv.org/abs/2112.02505) |
| UniversalNER | 2023 | 从 LLM 蒸馏做 NER | [arXiv](https://arxiv.org/abs/2308.03279) |
| Precision Shaking and DORPO | 2024 | LLM 蒸馏新方法 | [GitHub](https://github.com/OpZest/Papers/blob/main/White_papers/Precision_Shaking_and_DORPO.md) |
| Scaling Reasoning Efficiently via Relaxed On-Policy Distillation | 2026 | 最新推理蒸馏 | [arXiv](https://arxiv.org/abs/2603.11137) |

**核心应用**：
- 将大型预训练语言模型（如 BERT、GPT）的知识迁移到小模型
- 用于模型压缩、边缘部署、推理加速

### 方向D：扩散模型蒸馏

| 论文 | 年份 | 说明 | 链接 |
|------|------|------|------|
| On Distillation of Guided Diffusion Models | 2022 | 引导扩散模型蒸馏 | [arXiv](https://arxiv.org/abs/2210.03142) |
| Progressive Distillation for Fast Sampling | 2022 | 渐进式蒸馏加速采样 | [arXiv](https://arxiv.org/abs/2202.00512) |
| Adversarial Diffusion Distillation | 2023 | Stability AI 的工作 | [arXiv](https://arxiv.org/abs/2311.17042) |
| One-step Diffusion with Distribution Matching Distillation | 2023 | 一步扩散蒸馏 | [arXiv](https://arxiv.org/abs/2311.18828) |

**核心目标**：减少扩散模型的采样步数，从几百步减少到几步，加速推理。

### 方向E：视觉模型蒸馏

| 论文 | 年份 | 说明 | 链接 |
|------|------|------|------|
| DeiT: Training data-efficient image transformers | 2020 | Transformer 蒸馏 | [arXiv](https://arxiv.org/abs/2012.12877) |
| ScaleKD: Strong Vision Transformers Could Be Excellent Teachers | 2024 | ViT 作为教师 | [arXiv](https://arxiv.org/abs/2411.06786) |
| Vision-Language-Vision Auto-Encoder | 2025 | 跨模态蒸馏 | [arXiv](https://arxiv.org/abs/2507.07104) |

**核心应用**：
- 将大型 Vision Transformer (ViT) 的知识迁移到 CNN 或轻量级 ViT
- 用于图像分类、目标检测、语义分割

---

## 第四阶段：动手实践

推荐从 PyTorch 实现入手（最活跃的生态）：

### 入门实践

| 项目 | 说明 | 链接 |
|------|------|------|
| knowledge-distillation-pytorch | 简洁的 KD 实验框架 | [GitHub](https://github.com/peterliht/knowledge-distillation-pytorch) |
| Knowledge-Distillation-Zoo | 多种 KD 方法集合 | [GitHub](https://github.com/AberHu/Knowledge-Distillation-Zoo) |

### 系统框架

| 项目 | 说明 | 链接 |
|------|------|------|
| torchdistill | 配置驱动的 KD 框架，非常灵活 | [GitHub](https://github.com/yoshitomo-matsubara/torchdistill) |
| Intel Neural Network Distiller | Intel 的压缩研究工具 | [GitHub](https://github.com/IntelLabs/distiller) |
| KD_Lib | 知识蒸馏 benchmark 库 | [GitHub](https://github.com/SforAiDl/KD_Lib) |

### 专题实践

| 领域 | 项目 | 说明 | 链接 |
|------|------|------|------|
| 对比蒸馏 | RepDistiller | Contrastive Representation Distillation | [GitHub](https://github.com/HobbitLong/RepDistiller) |
| 知识回顾 | ReviewKD | 知识回顾蒸馏 | [GitHub](https://github.com/dvlab-research/ReviewKD) |
| NLP | TinyBERT | BERT 蒸馏 | [GitHub](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT) |
| 扩散模型 | DeepInversion | 无数据蒸馏 | [GitHub](https://github.com/NVlabs/DeepInversion) |
| 扩散模型 | ARD | 自回归蒸馏 | [GitHub](https://github.com/alsdudrla10/ARD) |

### 其他框架实现

| 框架 | 项目 | 说明 | 链接 |
|------|------|------|------|
| TensorFlow | Knowledge Distillation Methods | 多种 KD 方法 TF 实现 | [GitHub](https://github.com/sseung0703/KD_methods_with_TF) |
| TensorFlow | Zero-Shot Knowledge Distillation | 零样本蒸馏 | [GitHub](https://github.com/sseung0703/Zero-shot_Knowledge_Distillation) |
| Caffe | Face Model Compression | 人脸模型压缩 | [GitHub](https://github.com/liuziwei7/mobile-id) |
| Keras | Knowledge distillation with Keras | Keras 实现 | [GitHub](https://github.com/TropComplique/knowledge-distillation-keras) |

---

## 第五阶段：阅读综述，建立全局视野

| 论文 | 年份 | 说明 | 链接 |
|------|------|------|------|
| Knowledge Distillation and Student-Teacher Learning: A Review | 2020 | 早期但全面的综述 | [arXiv](https://arxiv.org/abs/2004.05937) |
| Knowledge Distillation: A Survey | 2021 | 经典综述 | [arXiv](https://arxiv.org/abs/2006.05525) |
| A Comprehensive Survey on Knowledge Distillation | 2025 | 最新综合综述 | [arXiv](https://arxiv.org/abs/1912.10850) |
| Knowledge Distillation and Dataset Distillation of LLMs | 2025 | LLM 蒸馏综述 | [arXiv](https://arxiv.org/abs/2504.14772) |

### 综述阅读建议

1. **入门**：先读 2021 年的 "Knowledge Distillation: A Survey"，建立全局框架
2. **进阶**：读 2020 年的学生-教师学习综述，了解不同蒸馏策略
3. **前沿**：读 2025 年的两篇最新综述，了解 LLM 蒸馏和扩散模型蒸馏的最新进展

---

## 学习建议

### 1. 先读 Hinton 2015 那篇

理解软标签、温度参数、KL 散度损失的核心思想。这是知识蒸馏的根基。

### 2. 动手实现最基础的 KD

用 PyTorch 在 CIFAR-10/MNIST 上跑一个 teacher-student 实验。推荐代码框架：
- [knowledge-distillation-pytorch](https://github.com/peterliht/knowledge-distillation-pytorch)

### 3. 按需深入

根据你的应用场景选择对应方向：

| 应用场景 | 推荐方向 | 推荐论文 |
|---------|---------|---------|
| LLM 压缩/部署 | NLP/LLM 蒸馏 | TinyBERT, UniversalNER |
| 视觉模型加速 | 视觉模型蒸馏 | DeiT, ScaleKD |
| 扩散模型加速 | 扩散模型蒸馏 | Progressive Distillation, DMD |
| 数据隐私/无数据 | 无数据蒸馏 | DeepInversion |
| 资源极度受限 | 自蒸馏 | Be Your Own Teacher |

### 4. 关注最新趋势

2023-2026 年的论文集中在：
- **LLM 蒸馏**：如何将大语言模型的知识迁移到小模型
- **扩散模型蒸馏**：如何加速扩散模型的采样过程
- **安全蒸馏**：蒸馏对模型安全性的影响

### 5. 注意安全

蒸馏可能削弱模型安全性。参考论文：

- [To Distill or Not to Distill: When Knowledge Transfer Undermines Safety of LLMs (2026)](https://openreview.net/pdf/fed763a30898a94daf0c79a480b698875f2cf105.pdf)

---

## 附录：资源导航

### 论文分类速查

| 类别 | 代表论文 |
|------|---------|
| 奠基之作 | Hinton 2015, Caruana 2006 |
| 中间层蒸馏 | FitNets, Overhaul KD |
| 注意力蒸馏 | Attention Transfer |
| 关系蒸馏 | Relational KD, CRD |
| 自蒸馏 | Born Again, BYOT |
| 无数据蒸馏 | DeepInversion, DAFL |
| NLP 蒸馏 | TinyBERT, UniversalNER |
| 扩散蒸馏 | Progressive Distillation, ADD |
| 视觉蒸馏 | DeiT, ScaleKD |

### 按年份分类

| 年份范围 | 主要贡献 |
|---------|---------|
| 1990-2014 | 集成学习、模型压缩、暗知识概念 |
| 2015-2017 | FitNets、Attention Transfer、Mean Teacher |
| 2018-2019 | 自蒸馏、关系蒸馏、特征蒸馏全面改进 |
| 2020-2021 | DeepInversion、DeiT、NLP 蒸馏快速发展 |
| 2022-2023 | 扩散模型蒸馏、LLM 蒸馏兴起 |
| 2024-2026 | LLM 安全蒸馏、最新推理蒸馏 |

---

*本文档由 AI 助手生成，基于 awesome-knowledge-distillation 仓库内容整理。*
