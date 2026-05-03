import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class StochasticBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, survival_prob=1.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.survival_prob = survival_prob

    def forward(self, x):
        if self.training and torch.rand(1).item() > self.survival_prob:
            return self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet56(nn.Module):
    def __init__(self, num_classes=100, base_channels=32):
        super().__init__()
        self.in_channels = base_channels
        self.conv1 = nn.Conv2d(3, base_channels, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.layer1 = self._make_layer(BasicBlock, base_channels, 9, stride=1)
        self.layer2 = self._make_layer(BasicBlock, base_channels * 2, 9, stride=2)
        self.layer3 = self._make_layer(BasicBlock, base_channels * 4, 9, stride=2)
        self.linear = nn.Linear(base_channels * 4, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = the.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class NoisyResNet56(nn.Module):
    def __init__(self, num_classes=100, base_channels=32, dropout=0.3, stochastic_depth_p=0.1):
        super().__init__()
        self.in_channels = base_channels
        self.stochastic_depth_p = stochastic_depth_p
        self.total_blocks = 27
        self.block_idx = 0
        self.conv1 = nn.Conv2d(3, base_channels, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.layer1 = self._make_layer(StochasticBasicBlock, base_channels, 9, stride=1)
        self.layer2 = self._make_layer(StochasticBasicBlock, base_channels * 2, 9, stride=2)
        self.layer3 = self._make_layer(StochasticBasicBlock, base_channels * 4, 9, stride=2)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(base_channels * 4, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            survival_prob = 1 - (self.block_idx / self.total_blocks) * self.stochastic_depth_p
            layers.append(block(self.in_channels, out_channels, s, survival_prob))
            self.in_channels = out_channels
            self.block_idx += 1
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.linear(out)
        return out
