"""
Ported to PyTorch from the Keras Xception implementation thanks to [tstandley](https://github.com/tstandley/Xception-PyTorch)

Model Info:
- Achieves ~78.89% top-1 and ~94.29% top-5 accuracy on ImageNet val set.
- Input size must be (3 x 299 x 299)
- Normalization: mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]
- Validation pipeline: Resize to 333 → CenterCrop 299

This implementation includes:
- Xception model (with depthwise separable convolutions)
- Optional pretrained weights (ImageNet)
- A modified variant for 15-channel input (Xception_concat)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init

# Pretrained model settings for loading ImageNet weights
pretrained_settings = {
    'xception': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000,
            'scale': 0.8975
        }
    }
}


class SeparableConv2d(nn.Module):
    """
    Depthwise separable convolution:
    - First, depthwise convolution (groups=in_channels)
    - Then, pointwise (1x1) convolution to combine channels
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding,
                               dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    """
    Xception residual block consisting of SeparableConv2d layers and optional skip connection.

    Args:
        in_filters (int): Number of input channels.
        out_filters (int): Number of output channels.
        reps (int): Number of repeated layers.
        strides (int): Stride for the block.
        start_with_relu (bool): Whether to start the block with ReLU.
        grow_first (bool): Whether to increase channel size at the beginning.
    """
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super().__init__()
        self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False) if (out_filters != in_filters or strides != 1) else None
        self.skipbn = nn.BatchNorm2d(out_filters) if self.skip else None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep += [self.relu,
                    SeparableConv2d(in_filters, out_filters, 3, 1, 1),
                    nn.BatchNorm2d(out_filters)]
            filters = out_filters

        for _ in range(reps - 1):
            rep += [self.relu,
                    SeparableConv2d(filters, filters, 3, 1, 1),
                    nn.BatchNorm2d(filters)]

        if not grow_first:
            rep += [self.relu,
                    SeparableConv2d(in_filters, out_filters, 3, 1, 1),
                    nn.BatchNorm2d(out_filters)]

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))

        self.rep = nn.Sequential(*rep)

    def forward(self, x):
        out = self.rep(x)
        skip = self.skipbn(self.skip(x)) if self.skip else x
        return out + skip


class Xception(nn.Module):
    """
    Standard Xception model with ImageNet-style architecture.

    Args:
        num_classes (int): Number of output classes (default 1000 for ImageNet).
    """
    def __init__(self, num_classes=1000):
        super().__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = Block(64, 128, 2, 2, False, True)
        self.block2 = Block(128, 256, 2, 2)
        self.block3 = Block(256, 728, 2, 2)

        self.block4 = Block(728, 728, 3)
        self.block5 = Block(728, 728, 3)
        self.block6 = Block(728, 728, 3)
        self.block7 = Block(728, 728, 3)
        self.block8 = Block(728, 728, 3)
        self.block9 = Block(728, 728, 3)
        self.block10 = Block(728, 728, 3)
        self.block11 = Block(728, 728, 3)

        self.block12 = Block(728, 1024, 2, 2, True, False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)

    def features(self, input):
        x = self.relu(self.bn1(self.conv1(input)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        return x

    def logits(self, features):
        x = self.relu(features)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        return self.logits(self.features(input))


class Xception_concat(nn.Module):
    """
    Xception model variant that accepts 15-channel input (for multiple frame or stream fusion).

    Args:
        num_classes (int): Number of output classes (default 1000).
    """
    def __init__(self, num_classes=1000):
        super().__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(15, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = Block(64, 128, 2, 2, False, True)
        self.block2 = Block(128, 256, 2, 2)
        self.block3 = Block(256, 728, 2, 2)

        self.block4 = Block(728, 728, 3)
        self.block5 = Block(728, 728, 3)
        self.block6 = Block(728, 728, 3)
        self.block7 = Block(728, 728, 3)
        self.block8 = Block(728, 728, 3)
        self.block9 = Block(728, 728, 3)
        self.block10 = Block(728, 728, 3)
        self.block11 = Block(728, 728, 3)

        self.block12 = Block(728, 1024, 2, 2, True, False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)

    def features(self, input):
        x = self.relu(self.bn1(self.conv1(input)))
        x = self.relu(self.bn2(self.conv2(x)))
        for block in [self.block1, self.block2, self.block3, self.block4, self.block5,
                      self.block6, self.block7, self.block8, self.block9, self.block10,
                      self.block11, self.block12]:
            x = block(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        return x

    def logits(self, features):
        x = self.relu(features)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        return self.logits(self.features(input))


def xception(num_classes=1000, pretrained='imagenet'):
    """
    Creates Xception model. Optionally loads ImageNet pretrained weights.

    Args:
        num_classes (int): Number of output classes.
        pretrained (str or bool): If 'imagenet', loads weights trained on ImageNet.

    Returns:
        Xception: Xception model instance.
    """
    model = Xception(num_classes=num_classes)
    if pretrained:
        settings = pretrained_settings['xception'][pretrained]
        assert num_classes == settings['num_classes'], \
            f"num_classes should be {settings['num_classes']}, but is {num_classes}"
        model.load_state_dict(model_zoo.load_url(settings['url']))
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']

    model.last_linear = model.fc
    del model.fc
    return model


def xception_concat(num_classes=1000):
    """
    Creates Xception_concat model (15-channel input).

    Args:
        num_classes (int): Number of output classes.

    Returns:
        Xception_concat: model instance.
    """
    model = Xception_concat(num_classes=num_classes)
    model.last_linear = model.fc
    del model.fc
    return model
