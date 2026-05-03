import os
import torch
import torch.nn as nn
from models import ResNet56, ResNet20
from losses import WassersteinKDLoss, KLDistillationLoss
from utils import evaluate


def train_teacher(config, train_loader, test_loader, device):
    print("\n" + "=" * 60)
    print("训练教师模型 (ResNet-56)")
    print("=" * 60)

    model = ResNet56(num_classes=config.num_classes).to(device)
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
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
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
    torch.save(model.state_dict(), os.path.join(config.checkpoint_dir, "teacher_resnet56.pth"))
    print(f"\n教师模型最佳准确率: {best_acc:.2f}%")
    config.teacher_acc = best_acc
    return model


def distill_train_wasserstein(config, teacher_model, train_loader, test_loader, device):
    print("\n" + "=" * 60)
    print("Wasserstein 蒸馏训练学生模型 (ResNet-20 ← ResNet-56)")
    print("=" * 60)
    print(f"α={config.alpha}, 距离度量: W_1 (排序概率 L1)")

    teacher_model.eval()
    student_model = ResNet20(num_classes=config.num_classes).to(device)
    distill_loss = WassersteinKDLoss(alpha=config.alpha)
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
        running_wd = 0.0
        running_ce = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                teacher_logits = teacher_model(images)

            student_logits = student_model(images)
            loss, w_dist, ce_loss = distill_loss(student_logits, teacher_logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_wd += w_dist.item() * images.size(0)
            running_ce += ce_loss.item() * images.size(0)
            _, predicted = student_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()
        train_acc = 100.0 * correct / total
        avg_loss = running_loss / total
        avg_wd = running_wd / total
        avg_ce = running_ce / total

        if epoch % 10 == 0 or epoch == config.epochs:
            test_acc = evaluate(student_model, test_loader, device)
            print(f"Epoch [{epoch}/{config.epochs}] 损失: {avg_loss:.4f} (W1: {avg_wd:.4f}, CE: {avg_ce:.4f}) 训练准确率: {train_acc:.2f}% 测试准确率: {test_acc:.2f}%")
            if test_acc > best_acc:
                best_acc = test_acc

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    torch.save(student_model.state_dict(), os.path.join(config.checkpoint_dir, "student_wasserstein_resnet20.pth"))
    print(f"\nWasserstein 蒸馏学生模型最佳准确率: {best_acc:.2f}%")
    config.wasserstein_acc = best_acc
    return student_model


def distill_train_kl(config, teacher_model, train_loader, test_loader, device):
    print("\n" + "=" * 60)
    print("KL 散度蒸馏训练学生模型 (ResNet-20 ← ResNet-56)")
    print("=" * 60)
    print(f"温度 T={config.temperature}, α={config.alpha}")

    teacher_model.eval()
    student_model = ResNet20(num_classes=config.num_classes).to(device)
    distill_loss = KLDistillationLoss(temperature=config.temperature, alpha=config.alpha)
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
            loss, kd_loss, ce_loss = distill_loss(student_logits, teacher_logits, labels)

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
            print(f"Epoch [{epoch}/{config.epochs}] 损失: {avg_loss:.4f} (KL: {avg_kd:.4f}, CE: {avg_ce:.4f}) 训练准确率: {train_acc:.2f}% 测试准确率: {test_acc:.2f}%")
            if test_acc > best_acc:
                best_acc = test_acc

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    torch.save(student_model.state_dict(), os.path.join(config.checkpoint_dir, "student_kl_resnet20.pth"))
    print(f"\nKL 散度蒸馏学生模型最佳准确率: {best_acc:.2f}%")
    config.kl_acc = best_acc
    return student_model


def train_student_baseline(config, train_loader, test_loader, device):
    print("\n" + "=" * 60)
    print("训练基线学生模型 (ResNet-20, 无蒸馏)")
    print("=" * 60)

    model = ResNet20(num_classes=config.num_classes).to(device)
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
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
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
    torch.save(model.state_dict(), os.path.join(config.checkpoint_dir, "student_baseline_resnet20.pth"))
    print(f"\n基线学生模型最佳准确率: {best_acc:.2f}%")
    config.baseline_acc = best_acc
    return model
