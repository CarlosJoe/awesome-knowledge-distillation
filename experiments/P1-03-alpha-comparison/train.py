import torch
import torch.nn.functional as F

from config import Config
from losses import DistillationLoss
from models import StudentNet
from utils import evaluate, set_seed


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
        print(f'[教师] Epoch {epoch+1}/{config.teacher_epochs}, Loss: {total_loss/len(train_loader):.4f}, Acc: {acc:.2f}%')


def distill_with_alpha(teacher, train_loader, test_loader, device, alpha, config=Config()):
    set_seed(config.seed)
    student = StudentNet().to(device)
    criterion = DistillationLoss(T=config.T, alpha=alpha)
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
        print(f'  [α={alpha}] Epoch {epoch+1}/{config.student_epochs}, Loss: {total_loss/len(train_loader):.4f}, Acc: {acc:.2f}%')

    final_acc = evaluate(student, test_loader, device)
    return student, final_acc


def train_student_baseline(train_loader, test_loader, device, config=Config()):
    set_seed(config.seed)
    student = StudentNet().to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma)
    for epoch in range(config.student_epochs):
        student.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            loss = F.cross_entropy(student(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        acc = evaluate(student, test_loader, device)
        print(f'[学生基线] Epoch {epoch+1}/{config.student_epochs}, Loss: {total_loss/len(train_loader):.4f}, Acc: {acc:.2f}%')
    return student, evaluate(student, test_loader, device)


def run_alpha_sweep(teacher, train_loader, test_loader, device, config=Config()):
    results = {}
    for alpha in config.alpha_values:
        print(f"\n--- 蒸馏训练: α = {alpha} (T={config.T}) ---")
        _, acc = distill_with_alpha(teacher, train_loader, test_loader, device, alpha, config)
        results[alpha] = acc
        print(f"  α={alpha} 最终准确率: {acc:.2f}%")
    return results
