import random

import numpy as np
import torch


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            images = batch[0].to(device)
            labels = batch[1].to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total


def generate_pseudo_labels(model, dataloader, device):
    model.eval()
    pseudo_labels = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch[0].to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            pseudo_labels.extend(predicted.cpu().tolist())
    return pseudo_labels
