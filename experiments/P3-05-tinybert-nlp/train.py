import torch
from transformers import get_linear_schedule_with_warmup, AdamW

from config import Config
from losses import TinyBERTDistiller
from utils import evaluate, get_layer_map


def finetune_teacher(model, train_loader, val_loader, device, config: Config = Config()):
    optimizer = AdamW(model.parameters(), lr=config.teacher_lr, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.teacher_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    for epoch in range(config.teacher_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        acc = evaluate(model, val_loader, device)
        print(f'[Teacher] Epoch {epoch+1}/{config.teacher_epochs}, Loss: {total_loss/len(train_loader):.4f}, Acc: {acc:.2f}%')


def distill_train(teacher, student, W_e, W_h, train_loader, val_loader, device, config: Config = Config()):
    distiller = TinyBERTDistiller(config)
    layer_map = get_layer_map(config.teacher_num_layers, config.num_student_layers)
    optimizer = AdamW(
        list(student.parameters()) + list(W_e.parameters()) + list(W_h.parameters()),
        lr=config.student_lr,
        weight_decay=config.weight_decay,
    )
    total_steps = len(train_loader) * config.student_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    teacher.eval()

    for epoch in range(config.student_epochs):
        student.train()
        W_e.train()
        W_h.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.no_grad():
                teacher_outputs = teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    output_attentions=True,
                )

            student_outputs = student(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True,
            )

            loss = distiller.compute_loss(student_outputs, teacher_outputs, labels, W_e, W_h, layer_map)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        acc = evaluate(student, val_loader, device)
        print(f'[Student KD] Epoch {epoch+1}/{config.student_epochs}, Loss: {total_loss/len(train_loader):.4f}, Acc: {acc:.2f}%')


def train_student_baseline(model, train_loader, val_loader, device, config: Config = Config()):
    optimizer = AdamW(model.parameters(), lr=config.student_lr, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.student_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    for epoch in range(config.student_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        acc = evaluate(model, val_loader, device)
        print(f'[Student Baseline] Epoch {epoch+1}/{config.student_epochs}, Loss: {total_loss/len(train_loader):.4f}, Acc: {acc:.2f}%')
