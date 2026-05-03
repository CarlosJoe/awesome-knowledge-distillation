import torch
import torch.nn.functional as F

from config import Config
from losses import DeiTDistillationLoss
from utils import evaluate


def train_teacher(model, train_loader, test_loader, device, config=Config()):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.teacher_epochs)
    for epoch in range(config.teacher_epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            loss = F.cross_entropy(model(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        acc = evaluate(model, test_loader, device)
        print(f'[Teacher] Epoch {epoch+1}/{config.teacher_epochs}, Loss: {total_loss/len(train_loader):.4f}, Acc: {acc:.2f}%')


def train_deit_distill(teacher, student, train_loader, test_loader, device, config=Config()):
    criterion = DeiTDistillationLoss(T=config.T, alpha=config.alpha)
    optimizer = torch.optim.AdamW(student.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.student_epochs)
    teacher.eval()

    for epoch in range(config.student_epochs):
        student.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                teacher_logits = teacher(images)
            cls_out, dist_out = student(images)
            loss = criterion(cls_out, dist_out, teacher_logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        acc = evaluate(student, test_loader, device, is_deit_distill=True)
        print(f'[DeiT Distill] Epoch {epoch+1}/{config.student_epochs}, Loss: {total_loss/len(train_loader):.4f}, Acc: {acc:.2f}%')


def train_deit_baseline(model, train_loader, test_loader, device, config=Config()):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.student_epochs)

    for epoch in range(config.student_epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            loss = F.cross_entropy(model(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        acc = evaluate(model, test_loader, device)
        print(f'[DeiT Baseline] Epoch {epoch+1}/{config.student_epochs}, Loss: {total_loss/len(train_loader):.4f}, Acc: {acc:.2f}%')
