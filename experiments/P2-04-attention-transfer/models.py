import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self._features = {}
        self._hooks = []

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _register_hooks(self):
        self._hooks.clear()

        def hook_fn(name):
            def hook(module, input, output):
                self._features[name] = output
            return hook

        self._hooks.append(self.layer1.register_forward_hook(hook_fn("layer1")))
        self._hooks.append(self.layer2.register_forward_hook(hook_fn("layer2")))
        self._hooks.append(self.layer3.register_forward_hook(hook_fn("layer3")))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._features.clear()

    def forward(self, x, return_features=False):
        if return_features:
            self._features.clear()

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        if return_features:
            features = dict(self._features)
            return out, features
        return out

    def get_attention_maps(self, x, p=2):
        _, features = self.forward(x, return_features=True)
        attention_maps = {}
        for name, feat in features.items():
            attention_maps[name] = self._compute_attention(feat, p)
        return attention_maps

    @staticmethod
    def _compute_attention(feature, p=2):
        return (feature.abs() ** p).sum(dim=1, keepdim=True)


def resnet20(num_classes=10):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)


def resnet56(num_classes=10):
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes)


def get_model(depth, num_classes=10):
    if depth == 20:
        return resnet20(num_classes)
    elif depth == 56:
        return resnet56(num_classes)
    else:
        raise ValueError(f"Unsupported depth: {depth}")
