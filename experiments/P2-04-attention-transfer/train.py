import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from config import ATConfig
from losses import ATDistillationLoss
from utils import evaluate


def train_teacher(model, train_loader, test_loader, device, config=ATConfig()):
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config.lr_milestones, gamma=config.lr_gamma
    )

    best_acc = 0.0
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()
        train_acc = 100.0 * correct / total
        avg_loss = total_loss / total

        if (epoch + 1) % 20 == 0 or epoch == 0:
            test_acc = evaluate(model, test_loader, device)
            print(f"[Teacher] Epoch {epoch+1}/{config.epochs}  Loss: {avg_loss:.4f}  Train Acc: {train_acc:.2f}%  Test Acc: {test_acc:.2f}%")
            if test_acc > best_acc:
                best_acc = test_acc
                save_path = Path(config.save_dir)
                save_path.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_path / "teacher_resnet56.pth")

    print(f"[Teacher] 最佳准确率: {best_acc:.2f}%")
    return best_acc


def distill_train_at(teacher, student, train_loader, test_loader, device, config=ATConfig()):
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    teacher._register_hooks()
    student._register_hooks()

    criterion = ATDistillationLoss(
        temperature=config.temperature,
        alpha=config.alpha,
        at_p=config.at_p,
        at_beta=config.at_beta,
    )

    optimizer = torch.optim.SGD(
        student.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config.lr_milestones, gamma=config.lr_gamma
    )

    best_acc = 0.0
    for epoch in range(config.epochs):
        student.train()
        teacher.eval()
        total_loss = 0.0
        total_soft = 0.0
        total_hard = 0.0
        total_at = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                teacher_logits, teacher_features = teacher(images, return_features=True)

            student_logits, student_features = student(images, return_features=True)

            loss, soft_l, hard_l, at_l = criterion(
                student_logits, teacher_logits, student_features, teacher_features, labels
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_soft += soft_l.item() * images.size(0)
            total_hard += hard_l.item() * images.size(0)
            total_at += at_l.item() * images.size(0)
            _, predicted = student_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()
        train_acc = 100.0 * correct / total
        avg_loss = total_loss / total

        if (epoch + 1) % 20 == 0 or epoch == 0:
            test_acc = evaluate(student, test_loader, device)
            print(
                f"[AT Distill] Epoch {epoch+1}/{config.epochs}  "
                f"Loss: {avg_loss:.4f} (soft: {total_soft/total:.4f}  hard: {total_hard/total:.4f}  at: {total_at/total:.4f})  "
                f"Train Acc: {train_acc:.2f}%  Test Acc: {test_acc:.2f}%"
            )
            if test_acc > best_acc:
                best_acc = test_acc
                save_path = Path(config.save_dir)
                save_path.mkdir(parents=True, exist_ok=True)
                torch.save(student.state_dict(), save_path / "student_resnet20_at.pth")

    teacher.remove_hooks()
    student.remove_hooks()
    print(f"[AT Distill] 最佳准确率: {best_acc:.2f}%")
    return best_acc


def train_student_baseline(model, train_loader, test_loader, device, config=ATConfig()):
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config.lr_milestones, gamma=config.lr_gamma
    )

    best_acc = 0.0
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()
        train_acc = 100.0 * correct / total
        avg_loss = total_loss / total

        if (epoch + 1) % 20 == 0 or epoch == 0:
            test_acc = evaluate(model, test_loader, device)
            print(f"[Baseline] Epoch {epoch+1}/{config.epochs}  Loss: {avg_loss:.4f}  Train Acc: {train_acc:.2f}%  Test Acc: {test_acc:.2f}%")
            if test_acc > best_acc:
                best_acc = test_acc
                save_path = Path(config.save_dir)
                save_path.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_path / "student_resnet20_baseline.pth")

    print(f"[Baseline] 最佳准确率: {best_acc:.2f}%")
    return best_acc
