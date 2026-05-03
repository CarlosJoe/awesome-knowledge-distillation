import os
import torch
import torch.nn as nn
from models import FitNetsTeacher, FitNetsStudent, HintRegressor
from losses import HintLoss, FitNetsLoss
from utils import evaluate


def train_teacher(config, train_loader, test_loader, device):
    print("\n" + "=" * 60)
    print("训练教师模型 (FitNetsTeacher)")
    print("=" * 60)

    model = FitNetsTeacher(num_classes=config.num_classes).to(device)
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
    torch.save(model.state_dict(), os.path.join(config.checkpoint_dir, "teacher_fitnets.pth"))
    print(f"\n教师模型最佳准确率: {best_acc:.2f}%")
    config.teacher_acc = best_acc
    return model


def hint_training(config, teacher_model, student_model, regressor, train_loader, test_loader, device):
    print("\n" + "=" * 60)
    print("阶段一: Hint Training (学习教师中间层特征)")
    print("=" * 60)
    print(f"Hint Epochs: {config.hint_epochs}, LR: {config.hint_learning_rate}")

    teacher_model.eval()
    hint_loss_fn = HintLoss()

    params = list(student_model.hint_layer.parameters()) + list(regressor.parameters())
    optimizer = torch.optim.SGD(
        params,
        lr=config.hint_learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.hint_epochs)

    best_acc = 0.0
    for epoch in range(1, config.hint_epochs + 1):
        student_model.train()
        regressor.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                _, teacher_hint = teacher_model(images, return_hint=True)

            _, student_hint = student_model(images, return_hint=True)
            regressor_hint = regressor(student_hint)

            loss = hint_loss_fn(regressor_hint, teacher_hint)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            outputs = student_model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()
        train_acc = 100.0 * correct / total
        avg_loss = running_loss / total

        if epoch % 10 == 0 or epoch == config.hint_epochs:
            test_acc = evaluate(student_model, test_loader, device)
            print(f"Epoch [{epoch}/{config.hint_epochs}] Hint损失: {avg_loss:.4f} 训练准确率: {train_acc:.2f}% 测试准确率: {test_acc:.2f}%")
            if test_acc > best_acc:
                best_acc = test_acc

    config.hint_acc = best_acc
    print(f"\nHint Training 后学生模型准确率: {best_acc:.2f}%")
    return student_model, regressor


def kd_training(config, teacher_model, student_model, regressor, train_loader, test_loader, device):
    print("\n" + "=" * 60)
    print("阶段二: KD Training (软标签 + 硬标签 + Hint蒸馏)")
    print("=" * 60)
    print(f"KD Epochs: {config.kd_epochs}, T={config.temperature}, α={config.alpha}, β={config.beta}")

    teacher_model.eval()
    fitnets_loss = FitNetsLoss(
        temperature=config.temperature,
        alpha=config.alpha,
        beta=config.beta,
    )

    optimizer = torch.optim.SGD(
        student_model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.kd_epochs)

    best_acc = 0.0
    for epoch in range(1, config.kd_epochs + 1):
        student_model.train()
        regressor.train()
        running_loss = 0.0
        running_kd = 0.0
        running_ce = 0.0
        running_hint = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                teacher_logits, teacher_hint = teacher_model(images, return_hint=True)

            student_logits, student_hint = student_model(images, return_hint=True)
            regressor_hint = regressor(student_hint)

            loss, kd_loss, ce_loss, hint_loss = fitnets_loss(
                student_logits, teacher_logits, labels,
                student_hint=student_hint,
                teacher_hint=teacher_hint,
                regressor_hint=regressor_hint,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_kd += kd_loss.item() * images.size(0)
            running_ce += ce_loss.item() * images.size(0)
            running_hint += hint_loss.item() * images.size(0)
            _, predicted = student_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()
        train_acc = 100.0 * correct / total
        avg_loss = running_loss / total
        avg_kd = running_kd / total
        avg_ce = running_ce / total
        avg_hint = running_hint / total

        if epoch % 10 == 0 or epoch == config.kd_epochs:
            test_acc = evaluate(student_model, test_loader, device)
            print(f"Epoch [{epoch}/{config.kd_epochs}] 损失: {avg_loss:.4f} (KD: {avg_kd:.4f}, CE: {avg_ce:.4f}, Hint: {avg_hint:.4f}) 训练准确率: {train_acc:.2f}% 测试准确率: {test_acc:.2f}%")
            if test_acc > best_acc:
                best_acc = test_acc

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    torch.save(student_model.state_dict(), os.path.join(config.checkpoint_dir, "student_fitnets_kd.pth"))
    print(f"\nFitNets蒸馏学生模型最佳准确率: {best_acc:.2f}%")
    config.distilled_acc = best_acc
    return student_model


def train_student_baseline(config, train_loader, test_loader, device):
    print("\n" + "=" * 60)
    print("训练基线学生模型 (FitNetsStudent, 无蒸馏)")
    print("=" * 60)

    model = FitNetsStudent(num_classes=config.num_classes).to(device)
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
    torch.save(model.state_dict(), os.path.join(config.checkpoint_dir, "student_baseline_fitnets.pth"))
    print(f"\n基线学生模型最佳准确率: {best_acc:.2f}%")
    config.baseline_acc = best_acc
    return model
