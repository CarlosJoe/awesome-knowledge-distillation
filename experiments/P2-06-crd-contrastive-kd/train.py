import os
import torch
import torch.nn as nn
from models import ResNet32x4, ResNet8x4, Projector
from losses import CRDLoss
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
            logits, _ = model(images)
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


def distill_train_crd(config, teacher_model, train_loader, test_loader, device):
    print("\n" + "=" * 60)
    print("CRD 对比蒸馏训练 (ResNet-8x4 <- ResNet-32x4)")
    print("=" * 60)
    print(f"温度 T={config.temperature}, α={config.alpha}, CRD温度={config.crd_temp}")

    teacher_model.eval()

    student_model = ResNet8x4(num_classes=config.num_classes).to(device)

    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    with torch.no_grad():
        _, teacher_feat = teacher_model(dummy_input)
        _, student_feat = student_model(dummy_input)
    teacher_dim = teacher_feat.shape[1]
    student_dim = student_feat.shape[1]

    projector = Projector(student_dim, config.projector_hidden, teacher_dim).to(device)

    crd_loss_fn = CRDLoss(
        temperature=config.temperature,
        alpha=config.alpha,
        crd_temp=config.crd_temp,
    )

    params = list(student_model.parameters()) + list(projector.parameters())
    optimizer = torch.optim.SGD(
        params,
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_acc = 0.0
    for epoch in range(1, config.epochs + 1):
        student_model.train()
        projector.train()
        running_loss = 0.0
        running_kd = 0.0
        running_ce = 0.0
        running_crd = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                teacher_logits, teacher_feat = teacher_model(images)

            student_logits, student_feat = student_model(images)
            student_proj = projector(student_feat)

            loss, kd_loss, ce_loss, crd_loss = crd_loss_fn(
                student_logits, teacher_logits, student_proj, teacher_feat, labels
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_kd += kd_loss.item() * images.size(0)
            running_ce += ce_loss.item() * images.size(0)
            running_crd += crd_loss.item() * images.size(0)
            _, predicted = student_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()
        train_acc = 100.0 * correct / total
        avg_loss = running_loss / total
        avg_kd = running_kd / total
        avg_ce = running_ce / total
        avg_crd = running_crd / total

        if epoch % 10 == 0 or epoch == config.epochs:
            test_acc = evaluate(student_model, test_loader, device)
            print(f"Epoch [{epoch}/{config.epochs}] 损失: {avg_loss:.4f} (KD: {avg_kd:.4f}, CE: {avg_ce:.4f}, CRD: {avg_crd:.4f}) 训练准确率: {train_acc:.2f}% 测试准确率: {test_acc:.2f}%")
            if test_acc > best_acc:
                best_acc = test_acc

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    torch.save(student_model.state_dict(), os.path.join(config.checkpoint_dir, "student_crd_resnet8x4.pth"))
    torch.save(projector.state_dict(), os.path.join(config.checkpoint_dir, "projector_crd.pth"))
    print(f"\nCRD 蒸馏学生模型最佳准确率: {best_acc:.2f}%")
    config.crd_acc = best_acc
    return student_model, projector


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
            logits, _ = model(images)
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
