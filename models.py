import math
from abc import ABC, abstractmethod

import torch


def conv3x3(in_channels, out_channels, stride=1):
    return torch.nn.Conv2d(
        in_channels, out_channels, kernel_size=3,
        stride=stride, padding=1, bias=False)

class ResidualBlock(torch.nn.Module):
    def __init__(self, num_channels, stride=1):
        super().__init__()
        self.conv1 = conv3x3(num_channels, num_channels, stride)
        self.bn1 = torch.nn.BatchNorm2d(num_channels)
        self.conv2 = conv3x3(num_channels, num_channels)
        self.bn2 = torch.nn.BatchNorm2d(num_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.nn.functional.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += x
        out = torch.nn.fiunctional.relu(out)
        return out

# appendix F Network architechture (for Atari?)
class DownSample(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels//2, kernel_size=3,
            stride=2, padding=1, bias=False
        )
        self.resnet1 = torch.nn.ModuleList([ResidualBlock(out_channels//2) for _ in range(2)])
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=2, padding=1, bias=False
        )
        self.resnet2 = torch.nn.ModuleList([ResidualBlock(out_channels) for _ in range(3)])
        self.avgpool1 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.resnet3 = torch.nn.ModuleList([ResidualBlock(out_channels) for _ in range(3)])
        self.avgpool2 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        for block in self.resnet1:
            x = block(x)
        x = self.conv2(x)
        for block in self.resnet2:
            x = block(x)
        x = self.avgpool1(x)
        for block in self.resnet3:
            x = block(x)
        x = self.avgpool2(x)
        return x

