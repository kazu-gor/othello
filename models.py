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


class DownsampleCNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, h_w):
        super().__init__()
        mid_channels = (in_channels + out_channels) // 2
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels, mid_channels, kernel_size=h_w[0] * 2, stride=4, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(mid_channels, out_channels, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d(h_w)
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x


class RepresentationNetwork(torch.nn.Module):
    def __init__(self, observation_shape, stacked_observations, num_blocks,
                 num_channels, dowmsample):
        super().__init__()
        self.downsample = downsample
        if self.downsample:
            if self.downsample == "resnet":
                self.downsample_net = DownSample(
                    observation_shape[0] * (stacked_observations + 1)
                    + stacked_observations,
                    num_channels
                )
            elif self.downsample == "CNN":
                self.downsample_net = DownsampleCNN(
                    observation_shape[0] * (stacked_observations + 1)
                    + stacked_observations,
                    num_channels,
                    (
                        math.ceil(observation_shape[1] / 16),
                        math.ceil(observation_shape[2] / 16),n
                    )
                )
            else:
                raise NotImplementedError('downsample should be "resnet" or "CNN"')
        self.conv = conv3x3(
            observation_shape[0] * (stacked_observations + 1) + stacked_observations,
            num_channels,
        )
        self.bn = torch.nn.BatchNorm2d(num_channels)
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

    def forward(self, x):
        if self.downsample:
            x = self.downsample_net(x)
        else:
            x = self.conv(x)
            x = self.bn(x)
            x = torch.nn.functional.relu(x)
        for block in self.resblocks:
            x = block(x)
        return x


def mlp(input_size, layer_sizes, output_size,
        output_activation=torch.nn.Identity, activation=torch.nn.ELU):
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i+1]), act()]
    return torch.nn.Sequential(*layers)


class AbstractNetwork(ABC, torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass
    @abstractmethod
    def initial_inference(self, observation):
        pass

    @abstractmethod
    def recurrent_inference(self, encoded_state, action):
        pass

    def get_weights(self):
        return dict_to_cpu(self.state_dict())

    def set_weights(self, weights):
        self.load_state_dict(weights)

def dict_to_cpu(dictionary: dict) -> dict:
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict

