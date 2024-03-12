import torch.nn as nn
from torch.nn import functional as F
import torch
import math

# Modified from https://github.com/tonylins/pytorch-mobilenet-v2/blob/master/MobileNetV2.py.
# In this version, Relu6 is replaced with Relu to make it ONNX compatible.
# BatchNorm Layer is optional to make it easy do batch norm confusion.


def conv_bn(inp, oup, stride, use_batch_norm=True, onnx_compatible=False):
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6

    if use_batch_norm:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            ReLU(inplace=True)
        )


def conv_1x1_bn(inp, oup, use_batch_norm=True, onnx_compatible=False):
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6
    if use_batch_norm:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            ReLU(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio=6, use_res_connect=True, use_batch_norm=True, onnx_compatible=False):
        super(InvertedResidual, self).__init__()
        ReLU = nn.ReLU if onnx_compatible else nn.ReLU6

        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup and use_res_connect

        if expand_ratio == 1:
            if use_batch_norm:
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            else:
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                )
        else:
            if use_batch_norm:
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    ReLU(inplace=True),
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            else:
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    ReLU(inplace=True),
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class BiFPNBlock(nn.Module):
    """
    Bi-directional Feature Pyramid Network
    """

    def __init__(self, feature_size=64, expand_ratio=1, epsilon=0.0001):
        super(BiFPNBlock, self).__init__()
        self.epsilon = epsilon

        # InvertedResidual回来之前已经norm过了
        self.conv4_up = InvertedResidual(feature_size, feature_size, 1)
        self.conv5_up = InvertedResidual(feature_size, feature_size, 1)
        self.conv4_down = InvertedResidual(feature_size, feature_size, 1)
        self.conv3_down = InvertedResidual(feature_size, feature_size, 1)

        self.p5_upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.p4_downsample = nn.MaxPool2d(3, stride=2, padding=1)
        self.p3_downsample = nn.MaxPool2d(3, stride=2, padding=1)

        self.w1 = nn.Parameter(torch.Tensor(2, 3).fill_(0.5))
        self.w2 = nn.Parameter(torch.Tensor(3, 1).fill_(0.5))

    def forward(self, inputs):
        p3_x, p4_x, p5_x = inputs

        w1 = F.relu(self.w1.clone(),inplace=False)
        w1 = w1/(torch.sum(w1, dim=0) + self.epsilon)
        w2 = F.relu(self.w2.clone(),inplace=False)
        w2 = w2/(torch.sum(w2, dim=0) + self.epsilon)

        p5_td = p5_x
        p4_td = self.conv4_down(F.relu(w1[0, 0] * p4_x + w1[1, 0] * self.p5_upsample(p5_td)))
        p3_td = self.conv3_down(F.relu(w1[0, 1] * p3_x + w1[1, 1] * self.p4_upsample(p4_td)))

        # Calculate Bottom-Up Pathway
        p3_bu = p3_td
        p4_bu = self.conv4_up(F.relu(w2[0, 0] * p4_x + w2[1, 0] * p4_td + w2[2, 0] * self.p3_downsample(p3_bu)))
        p5_bu = self.conv5_up(F.relu(w1[0, 2] * p5_td + w1[1, 2] * self.p4_downsample(p4_bu)))

        return p3_bu, p4_bu, p5_bu


class Wrapped_InvertedResidual(nn.Module):
    def __init__(self, inp, out, t, n, s, use_res_connect=True, width_mult=1.0, use_batch_norm=True, onnx_compatible=False):
        super(Wrapped_InvertedResidual, self).__init__()
        block = InvertedResidual
        output_channel = int(out * width_mult)

        self.features = []

        for i in range(n):
            if i == 0:
                self.features.append(block(inp, output_channel, s, use_res_connect=use_res_connect,
                                           expand_ratio=t, use_batch_norm=use_batch_norm,
                                           onnx_compatible=onnx_compatible))
            else:
                self.features.append(block(output_channel, output_channel, 1,  use_res_connect=use_res_connect,
                                           expand_ratio=t, use_batch_norm=use_batch_norm,
                                           onnx_compatible=onnx_compatible))

        self.features = nn.Sequential(*self.features)

    def forward(self, x):
        x = self.features(x)
        return x


class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1., dropout_ratio=0.2,
                 use_batch_norm=True, onnx_compatible=False):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2, onnx_compatible=onnx_compatible)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s,
                                               expand_ratio=t, use_batch_norm=use_batch_norm,
                                               onnx_compatible=onnx_compatible))
                else:
                    self.features.append(block(input_channel, output_channel, 1,
                                               expand_ratio=t, use_batch_norm=use_batch_norm,
                                               onnx_compatible=onnx_compatible))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel,
                                         use_batch_norm=use_batch_norm, onnx_compatible=onnx_compatible))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
