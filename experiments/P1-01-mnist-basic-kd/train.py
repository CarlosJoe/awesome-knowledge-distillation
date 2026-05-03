import torch
import torch.nn.functional as F

from config import Config
from losses import DistillationLoss
from utils import evaluate


def train_teacher(model, train_loader, test_loader, device, config=Config()):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma)
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


def distill_train(teacher, student, train_loader, test_loader, device, config=Config()):
    criterion = DistillationLoss(T=config.T, alpha=config.alpha)
    optimizer = torch.optim.Adam(student.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma)
    teacher.eval()

    for epoch in range(config.student_epochs):
        student.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                teacher_logits = teacher(images)
            student_logits = student(images)
            loss = criterion(student_logits, teacher_logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        acc = evaluate(student, test_loader, device)
        print(f'[Student KD] Epoch {epoch+1}/{config.student_epochs}, Loss: {total_loss/len(train_loader):.4f}, Acc: {acc:.2f}%')


def train_student_baseline(model, train_loader, test_loader, device, config=Config()):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma)
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
        print(f'[Student Baseline] Epoch {epoch+1}/{config.student_epochs}, Loss: {total_loss/len(train_loader):.4f}, Acc: {acc:.2f}%')
