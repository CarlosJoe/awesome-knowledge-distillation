import torch
import torch.nn.functional as F

from config import Config
from losses import SelfDistillationLoss
from utils import evaluate


def train_self_distillation(model, train_loader, test_loader, device, config=Config()):
    criterion = SelfDistillationLoss(T=config.T, alpha=config.alpha, beta=config.beta, gamma=config.gamma)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            main_out, aux1_out, aux2_out = model(images)
            loss = criterion(main_out, aux1_out, aux2_out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        acc = evaluate(model, test_loader, device)
        print(f'[自蒸馏] Epoch {epoch+1}/{config.epochs}, Loss: {total_loss/len(train_loader):.4f}, Acc: {acc:.2f}%')


def train_baseline(model, train_loader, test_loader, device, config=Config()):
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = F.cross_entropy(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        acc = evaluate(model, test_loader, device)
        print(f'[直接训练] Epoch {epoch+1}/{config.epochs}, Loss: {total_loss/len(train_loader):.4f}, Acc: {acc:.2f}%')
