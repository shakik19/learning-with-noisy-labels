"""Networks: ResNet-18 (CIFAR variant) and a small CNN fallback for tight compute / smoke tests."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet18(nn.Module):
    """ResNet-18 for 32x32 inputs: 3x3 stem, no max-pool, [2, 2, 2, 2] basic blocks."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        layers, in_planes = [], 64
        for planes, stride in [(64, 1), (128, 2), (256, 2), (512, 2)]:
            layers += [BasicBlock(in_planes, planes, stride), BasicBlock(planes, planes, 1)]
            in_planes = planes
        self.layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.pool(self.layers(out)).flatten(1)
        return self.fc(out)


class SmallCNN(nn.Module):
    """Six conv layers (64-64-128-128-256-256) with batch norm; much cheaper than ResNet-18."""

    def __init__(self, num_classes: int = 10):
        super().__init__()

        def block(cin, cout):
            return [nn.Conv2d(cin, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]

        self.features = nn.Sequential(
            *block(3, 64), *block(64, 64), nn.MaxPool2d(2),
            *block(64, 128), *block(128, 128), nn.MaxPool2d(2),
            *block(128, 256), *block(256, 256), nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))


def build_model(arch: str, num_classes: int = 10) -> nn.Module:
    if arch == "resnet18":
        return ResNet18(num_classes)
    if arch == "cnn":
        return SmallCNN(num_classes)
    raise ValueError(f"unknown arch: {arch}")
