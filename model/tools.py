import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math

class Conv2dStaticSamePadding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, dilation=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=bias, groups=groups)
        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]

        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.conv(x)
        return x

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

        self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.relu=nn.ReLU()


    def forward(self, x):
        x = self.pointwise_conv(x)

        x = self.bn(x)

        x=self.relu(x)



        return x


class MaxPool2dStaticSamePadding(nn.Module):
    def __init__(self,kernal_size):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size=kernal_size)


    def forward(self, x):
        n,c,t,v=x.size()#h=150,w=25
        x=x.permute(0,1,3,2).contiguous().view(n,c*v,t)
        x=self.pool(x)
        x=x.view(n,c,v,-1).permute(0,1,3,2)
        return x

class TimeUpsample(nn.Module):
    def __init__(self,scale_factor=2):
        super(TimeUpsample, self).__init__()
        self.UpSample=nn.Upsample(scale_factor=scale_factor, mode='nearest')
    def forward(self,x):
        #n,c,t,v->n,c*v,t
        n,c,t,v=x.size()
        x=x.permute(0,1,3,2).contiguous().view(n,c*v,t)
        x=self.UpSample(x)
        x=x.view(n,c,v,-1).permute(0,1,3,2)
        return x