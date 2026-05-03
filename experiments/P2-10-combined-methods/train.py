import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import ResNet32x4, ResNet8x4
from losses import CombinedDistillationLoss
from utils import evaluate


def train_teacher(config, train_loader, test_loader, device):
    print("\n" + "=" * 60)
    print("训练教师模型 (ResNet-32x4)")
    print("=" * 60)

    model = ResNet32x4(num_classes=config.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_acc = 0.0
    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()
        train_acc = 100.0 * correct / total
        avg_loss = running_loss / total

        if epoch % 10 == 0 or epoch == config.epochs:
            test_acc = evaluate(model, test_loader, device)
            print(f"Epoch [{epoch}/{config.epochs}] 损失: {avg_loss:.4f} 训练准确率: {train_acc:.2f}% 测试准确率: {test_acc:.2f}%")
            if test_acc > best_acc:
                best_acc = test_acc

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(config.checkpoint_dir, "teacher_resnet32x4.pth"))
    print(f"\n教师模型最佳准确率: {best_acc:.2f}%")
    config.teacher_acc = best_acc
    return model


def distill_combined(config, teacher_model, train_loader, test_loader, device):
    print("\n" + "=" * 60)
    print("组合蒸馏训练 (Response + Feature + Attention)")
    print("=" * 60)
    print(f"温度 T={config.temperature}, α={config.alpha}, β={config.beta}, γ={config.gamma}")

    teacher_model.eval()
    student_model = ResNet8x4(num_classes=config.num_classes).to(device)
    criterion = CombinedDistillationLoss(
        temperature=config.temperature,
        alpha=config.alpha,
        beta=config.beta,
        gamma=config.gamma,
        at_p=config.at_p,
    )
    optimizer = torch.optim.SGD(
        student_model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_acc = 0.0
    for epoch in range(1, config.epochs + 1):
        student_model.train()
        running_loss = 0.0
        running_soft = 0.0
        running_hard = 0.0
        running_feat = 0.0
        running_attn = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                teacher_logits, teacher_features = teacher_model(images, return_features=True)

            student_logits, student_features = student_model(images, return_features=True)
            loss, l_soft, l_hard, l_feat, l_attn = criterion(
                student_logits, teacher_logits, student_features, teacher_features, labels
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_soft += l_soft.item() * images.size(0)
            running_hard += l_hard.item() * images.size(0)
            running_feat += l_feat.item() * images.size(0)
            running_attn += l_attn.item() * images.size(0)
            _, predicted = student_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()
        train_acc = 100.0 * correct / total
        avg_loss = running_loss / total
        avg_soft = running_soft / total
        avg_hard = running_hard / total
        avg_feat = running_feat / total
        avg_attn = running_attn / total

        if epoch % 10 == 0 or epoch == config.epochs:
            test_acc = evaluate(student_model, test_loader, device)
            print(f"Epoch [{epoch}/{config.epochs}] 损失: {avg_loss:.4f} (软: {avg_soft:.4f}, 硬: {avg_hard:.4f}, 特征: {avg_feat:.4f}, 注意力: {avg_attn:.4f}) 训练准确率: {train_acc:.2f}% 测试准确率: {test_acc:.2f}%")
            if test_acc > best_acc:
                best_acc = test_acc

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    torch.save(student_model.state_dict(), os.path.join(config.checkpoint_dir, "student_combined_resnet8x4.pth"))
    print(f"\n组合蒸馏学生模型最佳准确率: {best_acc:.2f}%")
    config.combined_acc = best_acc
    return student_model


def distill_response_only(config, teacher_model, train_loader, test_loader, device):
    print("\n" + "=" * 60)
    print("Response-only 蒸馏训练")
    print("=" * 60)
    print(f"温度 T={config.temperature}, α={config.alpha}")

    teacher_model.eval()
    student_model = ResNet8x4(num_classes=config.num_classes).to(device)
    optimizer = torch.optim.SGD(
        student_model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_acc = 0.0
    for epoch in range(1, config.epochs + 1):
        student_model.train()
        running_loss = 0.0
        running_kd = 0.0
        running_ce = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                teacher_logits = teacher_model(images)

            student_logits = student_model(images)

            soft_student = F.log_softmax(student_logits / config.temperature, dim=1)
            soft_teacher = F.softmax(teacher_logits / config.temperature, dim=1)
            kd_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (config.temperature ** 2)
            ce_loss = F.cross_entropy(student_logits, labels)
            loss = config.alpha * kd_loss + (1 - config.alpha) * ce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_kd += kd_loss.item() * images.size(0)
            running_ce += ce_loss.item() * images.size(0)
            _, predicted = student_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()
        train_acc = 100.0 * correct / total
        avg_loss = running_loss / total
        avg_kd = running_kd / total
        avg_ce = running_ce / total

        if epoch % 10 == 0 or epoch == config.epochs:
            test_acc = evaluate(student_model, test_loader, device)
            print(f"Epoch [{epoch}/{config.epochs}] 损失: {avg_loss:.4f} (KD: {avg_kd:.4f}, CE: {avg_ce:.4f}) 训练准确率: {train_acc:.2f}% 测试准确率: {test_acc:.2f}%")
            if test_acc > best_acc:
                best_acc = test_acc

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    torch.save(student_model.state_dict(), os.path.join(config.checkpoint_dir, "student_response_resnet8x4.pth"))
    print(f"\nResponse-only 蒸馏学生模型最佳准确率: {best_acc:.2f}%")
    config.response_only_acc = best_acc
    return student_model


def distill_feature_only(config, teacher_model, train_loader, test_loader, device):
    print("\n" + "=" * 60)
    print("Feature-only 蒸馏训练")
    print("=" * 60)
    print(f"温度 T={config.temperature}, α={config.alpha}, β={config.beta}")

    teacher_model.eval()
    student_model = ResNet8x4(num_classes=config.num_classes).to(device)
    optimizer = torch.optim.SGD(
        student_model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_acc = 0.0
    for epoch in range(1, config.epochs + 1):
        student_model.train()
        running_loss = 0.0
        running_kd = 0.0
        running_ce = 0.0
        running_feat = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                teacher_logits, teacher_features = teacher_model(images, return_features=True)

            student_logits, student_features = student_model(images, return_features=True)

            soft_student = F.log_softmax(student_logits / config.temperature, dim=1)
            soft_teacher = F.softmax(teacher_logits / config.temperature, dim=1)
            kd_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (config.temperature ** 2)
            ce_loss = F.cross_entropy(student_logits, labels)

            feat_loss = 0.0
            n_stages = 0
            for s_feat, t_feat in zip(student_features, teacher_features):
                if s_feat.shape[2:] != t_feat.shape[2:]:
                    t_feat = F.adaptive_avg_pool2d(t_feat, s_feat.shape[2:])
                feat_loss += F.l1_loss(s_feat, t_feat)
                n_stages += 1
            if n_stages > 0:
                feat_loss = feat_loss / n_stages

            loss = config.alpha * kd_loss + (1 - config.alpha) * ce_loss + config.beta * feat_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_kd += kd_loss.item() * images.size(0)
            running_ce += ce_loss.item() * images.size(0)
            running_feat += feat_loss.item() * images.size(0)
            _, predicted = student_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()
        train_acc = 100.0 * correct / total
        avg_loss = running_loss / total
        avg_kd = running_kd / total
        avg_ce = running_ce / total
        avg_feat = running_feat / total

        if epoch % 10 == 0 or epoch == config.epochs:
            test_acc = evaluate(student_model, test_loader, device)
            print(f"Epoch [{epoch}/{config.epochs}] 损失: {avg_loss:.4f} (KD: {avg_kd:.4f}, CE: {avg_ce:.4f}, 特征: {avg_feat:.4f}) 训练准确率: {train_acc:.2f}% 测试准确率: {test_acc:.2f}%")
            if test_acc > best_acc:
                best_acc = test_acc

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    torch.save(student_model.state_dict(), os.path.join(config.checkpoint_dir, "student_feature_resnet8x4.pth"))
    print(f"\nFeature-only 蒸馏学生模型最佳准确率: {best_acc:.2f}%")
    config.feature_only_acc = best_acc
    return student_model


def distill_attention_only(config, teacher_model, train_loader, test_loader, device):
    print("\n" + "=" * 60)
    print("Attention-only 蒸馏训练")
    print("=" * 60)
    print(f"温度 T={config.temperature}, α={config.alpha}, γ={config.gamma}, p={config.at_p}")

    teacher_model.eval()
    student_model = ResNet8x4(num_classes=config.num_classes).to(device)
    optimizer = torch.optim.SGD(
        student_model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_acc = 0.0
    for epoch in range(1, config.epochs + 1):
        student_model.train()
        running_loss = 0.0
        running_kd = 0.0
        running_ce = 0.0
        running_attn = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                teacher_logits, teacher_features = teacher_model(images, return_features=True)

            student_logits, student_features = student_model(images, return_features=True)

            soft_student = F.log_softmax(student_logits / config.temperature, dim=1)
            soft_teacher = F.softmax(teacher_logits / config.temperature, dim=1)
            kd_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (config.temperature ** 2)
            ce_loss = F.cross_entropy(student_logits, labels)

            attn_loss = 0.0
            n_stages = 0
            for s_feat, t_feat in zip(student_features, teacher_features):
                s_attn = (s_feat.abs() ** config.at_p).sum(dim=1, keepdim=True)
                t_attn = (t_feat.abs() ** config.at_p).sum(dim=1, keepdim=True)
                if s_attn.shape[2:] != t_attn.shape[2:]:
                    t_attn = F.adaptive_avg_pool2d(t_attn, s_attn.shape[2:])
                s_norm = s_attn / s_attn.norm(p=2, dim=(2, 3), keepdim=True).clamp(min=1e-8)
                t_norm = t_attn / t_attn.norm(p=2, dim=(2, 3), keepdim=True).clamp(min=1e-8)
                attn_loss += F.mse_loss(s_norm, t_norm)
                n_stages += 1
            if n_stages > 0:
                attn_loss = attn_loss / n_stages

            loss = config.alpha * kd_loss + (1 - config.alpha) * ce_loss + config.gamma * attn_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_kd += kd_loss.item() * images.size(0)
            running_ce += ce_loss.item() * images.size(0)
            running_attn += attn_loss.item() * images.size(0)
            _, predicted = student_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()
        train_acc = 100.0 * correct / total
        avg_loss = running_loss / total
        avg_kd = running_kd / total
        avg_ce = running_ce / total
        avg_attn = running_attn / total

        if epoch % 10 == 0 or epoch == config.epochs:
            test_acc = evaluate(student_model, test_loader, device)
            print(f"Epoch [{epoch}/{config.epochs}] 损失: {avg_loss:.4f} (KD: {avg_kd:.4f}, CE: {avg_ce:.4f}, 注意力: {avg_attn:.4f}) 训练准确率: {train_acc:.2f}% 测试准确率: {test_acc:.2f}%")
            if test_acc > best_acc:
                best_acc = test_acc

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    torch.save(student_model.state_dict(), os.path.join(config.checkpoint_dir, "student_attention_resnet8x4.pth"))
    print(f"\nAttention-only 蒸馏学生模型最佳准确率: {best_acc:.2f}%")
    config.attention_only_acc = best_acc
    return student_model


def train_student_baseline(config, train_loader, test_loader, device):
    print("\n" + "=" * 60)
    print("训练基线学生模型 (ResNet-8x4, 无蒸馏)")
    print("=" * 60)

    model = ResNet8x4(num_classes=config.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_acc = 0.0
    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()
        train_acc = 100.0 * correct / total
        avg_loss = running_loss / total

        if epoch % 10 == 0 or epoch == config.epochs:
            test_acc = evaluate(model, test_loader, device)
            print(f"Epoch [{epoch}/{config.epochs}] 损失: {avg_loss:.4f} 训练准确率: {train_acc:.2f}% 测试准确率: {test_acc:.2f}%")
            if test_acc > best_acc:
                best_acc = test_acc

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(config.checkpoint_dir, "student_baseline_resnet8x4.pth"))
    print(f"\n基线学生模型最佳准确率: {best_acc:.2f}%")
    config.baseline_acc = best_acc
    return model
