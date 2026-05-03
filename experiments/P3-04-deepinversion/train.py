import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from config import DeepInversionConfig
from models import ResNet56, ResNet20
from losses import DeepInversionLoss, KDLoss
from dataset import get_cifar10_trainloader, get_cifar10_testloader, get_generated_dataloader
from utils import evaluate, get_device, set_seed, ensure_dir


def train_teacher(config):
    device = get_device(config.device)
    model = ResNet56(num_classes=config.num_classes).to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.teacher_lr,
        momentum=config.teacher_momentum,
        weight_decay=config.teacher_weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.teacher_epochs)
    criterion = nn.CrossEntropyLoss()

    trainloader = get_cifar10_trainloader(config.student_batch_size, config.num_workers)
    testloader = get_cifar10_testloader(config.student_batch_size, config.num_workers)

    best_acc = 0.0
    for epoch in range(config.teacher_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()
        train_acc = 100.0 * correct / total

        if (epoch + 1) % 10 == 0 or epoch == config.teacher_epochs - 1:
            test_acc = evaluate(model, testloader, device)
            print(f"[教师训练] Epoch {epoch+1}/{config.teacher_epochs} | "
                  f"训练损失: {train_loss/len(trainloader):.4f} | "
                  f"训练准确率: {train_acc:.2f}% | 测试准确率: {test_acc:.2f}%")

            if test_acc > best_acc:
                best_acc = test_acc
                ensure_dir(os.path.dirname(config.teacher_path))
                torch.save(model.state_dict(), config.teacher_path)
                print(f"  -> 保存最佳教师模型, 准确率: {best_acc:.2f}%")

    print(f"\n教师模型训练完成, 最佳准确率: {best_acc:.2f}%")
    return model


def generate_images(config, teacher):
    device = get_device(config.device)
    teacher.eval()

    inv_loss_fn = DeepInversionLoss(config.bn_weight, config.tv_weight, config.l2_weight)
    inv_loss_fn.register_hooks(teacher)

    all_images = []
    all_labels = []

    for cls_idx in range(config.num_classes):
        print(f"\n[图像生成] 正在生成类别 {cls_idx} 的图像...")
        num_batches = (config.images_per_class + config.inv_batch_size - 1) // config.inv_batch_size
        remaining = config.images_per_class

        for batch_idx in range(num_batches):
            batch_size = min(remaining, config.inv_batch_size)
            remaining -= batch_size

            targets = torch.full((batch_size,), cls_idx, dtype=torch.long, device=device)
            images = torch.randn(batch_size, *config.img_size, device=device, requires_grad=True)

            optimizer = torch.optim.Adam([images], lr=config.inv_lr)

            for it in tqdm(range(config.inv_iterations), desc=f"类别{cls_idx} 批次{batch_idx+1}"):
                teacher.zero_grad()
                if images.grad is not None:
                    images.grad.zero_()

                logits = teacher(images)
                total_loss, ce_loss, bn_loss, tv_loss, l2_loss = inv_loss_fn(
                    logits, targets, images, teacher
                )
                total_loss.backward()
                optimizer.step()

                with torch.no_grad():
                    images.clamp_(0, 1)

                if (it + 1) % 500 == 0:
                    print(f"  迭代 {it+1}/{config.inv_iterations} | "
                          f"总损失: {total_loss.item():.4f} | "
                          f"CE: {ce_loss.item():.4f} | "
                          f"BN: {bn_loss.item():.4f} | "
                          f"TV: {tv_loss.item():.4f} | "
                          f"L2: {l2_loss.item():.4f}")

            with torch.no_grad():
                gen_images = images.detach().clamp(0, 1)
                all_images.append(gen_images.cpu())
                all_labels.append(targets.cpu())

    inv_loss_fn.remove_hooks()

    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    print(f"\n图像生成完成! 总计 {all_images.shape[0]} 张图像")

    ensure_dir(config.generated_dir)
    torch.save({"images": all_images, "labels": all_labels},
               os.path.join(config.generated_dir, "generated_data.pth"))
    print(f"生成图像已保存至 {config.generated_dir}/generated_data.pth")

    return all_images, all_labels


def distill_datafree(config, teacher, generated_images, generated_labels):
    device = get_device(config.device)
    student = ResNet20(num_classes=config.num_classes).to(device)
    optimizer = optim.SGD(
        student.parameters(),
        lr=config.student_lr,
        momentum=config.student_momentum,
        weight_decay=config.student_weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.student_epochs)
    kd_loss_fn = KDLoss(config.temperature, config.alpha)

    gen_loader = get_generated_dataloader(
        generated_images, generated_labels, config.student_batch_size, num_workers=0
    )
    testloader = get_cifar10_testloader(config.student_batch_size, config.num_workers)

    teacher.eval()
    best_acc = 0.0

    for epoch in range(config.student_epochs):
        student.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in gen_loader:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                teacher_logits = teacher(images)

            student_logits = student(images)
            loss = kd_loss_fn(student_logits, teacher_logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = student_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()
        train_acc = 100.0 * correct / total

        if (epoch + 1) % 10 == 0 or epoch == config.student_epochs - 1:
            test_acc = evaluate(student, testloader, device)
            print(f"[无数据蒸馏] Epoch {epoch+1}/{config.student_epochs} | "
                  f"训练损失: {train_loss/max(len(gen_loader),1):.4f} | "
                  f"训练准确率: {train_acc:.2f}% | 测试准确率: {test_acc:.2f}%")

            if test_acc > best_acc:
                best_acc = test_acc

    print(f"\n无数据蒸馏完成! 学生最佳准确率: {best_acc:.2f}%")
    return student, best_acc
