import os
import torch
import torch.nn as nn
from models import ResNet20
from losses import BornAgainLoss
from utils import evaluate


def train_generation_0(config, train_loader, test_loader, device):
    print("\n" + "=" * 60)
    print("第 0 代训练 (ResNet-20, 正常训练)")
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
    torch.save(model.state_dict(), os.path.join(config.checkpoint_dir, "generation_0.pth"))
    print(f"\n第 0 代最佳准确率: {best_acc:.2f}%")
    return model, best_acc


def train_generation_k(config, teacher_model, generation, train_loader, test_loader, device):
    print("\n" + "=" * 60)
    print(f"第 {generation} 代训练 (ResNet-20 ← 第 {generation - 1} 代教师)")
    print("=" * 60)
    print(f"温度 T={config.temperature}, λ={config.lambda_kl}")

    teacher_model.eval()
    student_model = ResNet20(num_classes=config.num_classes).to(device)
    distill_loss = BornAgainLoss(temperature=config.temperature, lambda_kl=config.lambda_kl)
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
        running_kl = 0.0
        running_ce = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                teacher_logits = teacher_model(images)

            student_logits = student_model(images)
            loss, kl_loss, ce_loss = distill_loss(student_logits, teacher_logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_kl += kl_loss.item() * images.size(0)
            running_ce += ce_loss.item() * images.size(0)
            _, predicted = student_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()
        train_acc = 100.0 * correct / total
        avg_loss = running_loss / total
        avg_kl = running_kl / total
        avg_ce = running_ce / total

        if epoch % 10 == 0 or epoch == config.epochs:
            test_acc = evaluate(student_model, test_loader, device)
            print(f"Epoch [{epoch}/{config.epochs}] 损失: {avg_loss:.4f} (KL: {avg_kl:.4f}, CE: {avg_ce:.4f}) 训练准确率: {train_acc:.2f}% 测试准确率: {test_acc:.2f}%")
            if test_acc > best_acc:
                best_acc = test_acc

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    torch.save(student_model.state_dict(), os.path.join(config.checkpoint_dir, f"generation_{generation}.pth"))
    print(f"\n第 {generation} 代最佳准确率: {best_acc:.2f}%")
    return student_model, best_acc
