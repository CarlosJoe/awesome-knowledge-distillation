import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, blocks, width_multiplier, num_classes=100):
        super().__init__()
        base_channels = 16 * width_multiplier
        self.in_channels = base_channels

        self.conv1 = nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)

        self.layer1 = self._make_layer(base_channels, blocks[0], stride=1)
        self.layer2 = self._make_layer(base_channels * 2, blocks[1], stride=2)
        self.layer3 = self._make_layer(base_channels * 4, blocks[2], stride=2)

        self.linear = nn.Linear(base_channels * 4, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        out = F.relu(self.bn1(self.conv1(x)))
        feat1 = self.layer1(out)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        out = F.adaptive_avg_pool2d(feat3, 1)
        out = out.view(out.size(0), -1)
        logits = self.linear(out)

        if return_features:
            return logits, [feat1, feat2, feat3]
        return logits


def ResNet32x4(num_classes=100):
    return ResNet((5, 5, 5), width_multiplier=4, num_classes=num_classes)


def ResNet8x4(num_classes=100):
    return ResNet((1, 1, 1), width_multiplier=4, num_classes=num_classes)
