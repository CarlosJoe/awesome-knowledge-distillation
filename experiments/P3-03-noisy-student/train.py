import torch
import torch.nn.functional as F

from config import Config
from models import ResNet56, NoisyResNet56
from losses import PseudoLabelLoss
from dataset import (
    get_labeled_loader, get_unlabeled_loader, get_combined_loader,
    get_standard_transform, get_plain_transform, get_strong_transform
)
from utils import evaluate, generate_pseudo_labels


def train_teacher(model, labeled_loader, test_loader, device, config=Config()):
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config.learning_rate,
        momentum=config.momentum, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.teacher_epochs)

    for epoch in range(config.teacher_epochs):
        model.train()
        total_loss = 0
        for images, labels in labeled_loader:
            images, labels = images.to(device), labels.to(device)
            loss = F.cross_entropy(model(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        acc = evaluate(model, test_loader, device)
        print(f'[Teacher] Epoch {epoch+1}/{config.teacher_epochs}, Loss: {total_loss/len(labeled_loader):.4f}, Acc: {acc:.2f}%')


def train_noisy_student(student, combined_loader, test_loader, device, config=Config(), iteration=1):
    criterion = PseudoLabelLoss(alpha=config.alpha)
    optimizer = torch.optim.SGD(
        student.parameters(), lr=config.learning_rate,
        momentum=config.momentum, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.student_epochs)

    for epoch in range(config.student_epochs):
        student.train()
        total_loss = 0
        for images, labels, is_labeled in combined_loader:
            images = images.to(device)
            labels = labels.to(device)
            is_labeled = is_labeled.to(device)
            logits = student(images)
            loss = criterion(logits, labels, is_labeled)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        acc = evaluate(student, test_loader, device)
        print(f'[Noisy Student 迭代 {iteration}] Epoch {epoch+1}/{config.student_epochs}, Loss: {total_loss/len(combined_loader):.4f}, Acc: {acc:.2f}%')


def iterative_training(config, labeled_data, labeled_targets, unlabeled_data, unlabeled_targets, test_loader, device):
    results = []

    print("=== 训练教师模型 ===")
    teacher = ResNet56(num_classes=100, base_channels=config.base_channels).to(device)
    labeled_loader = get_labeled_loader(labeled_data, labeled_targets, get_standard_transform(), config.batch_size)
    train_teacher(teacher, labeled_loader, test_loader, device, config)
    teacher_acc = evaluate(teacher, test_loader, device)
    results.append(teacher_acc)
    print(f"教师模型准确率: {teacher_acc:.2f}%")

    for iteration in range(1, config.num_iterations + 1):
        print(f"\n=== 迭代 {iteration}/{config.num_iterations} ===")

        print("生成伪标签...")
        unlabeled_loader = get_unlabeled_loader(unlabeled_data, get_plain_transform(), config.batch_size)
        pseudo_labels = generate_pseudo_labels(teacher, unlabeled_loader, device)
        print(f"伪标签生成完成 ({len(pseudo_labels)} 样本)")

        print("训练 Noisy Student...")
        student = NoisyResNet56(
            num_classes=100, base_channels=config.base_channels,
            dropout=config.dropout, stochastic_depth_p=config.stochastic_depth_p
        ).to(device)
        combined_loader = get_combined_loader(
            labeled_data, labeled_targets, unlabeled_data, pseudo_labels,
            get_strong_transform(), config.batch_size
        )
        train_noisy_student(student, combined_loader, test_loader, device, config, iteration)

        student_acc = evaluate(student, test_loader, device)
        results.append(student_acc)
        print(f"迭代 {iteration} 学生模型准确率: {student_acc:.2f}%")

        teacher = student

    return results
