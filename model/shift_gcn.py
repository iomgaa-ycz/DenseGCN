import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from .tools import *

import sys

sys.path.append("./model/Temporal_shift/")

from cuda.shift import Shift


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_init(conv):
    nn.init.kaiming_normal(conv.weight, mode='fan_out')
    nn.init.constant(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant(bn.weight, scale)
    nn.init.constant(bn.bias, 0)


class tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class YczAtt(nn.Module):
    def __init__(self, in_channels, out_chancels, in_time):
        super(YczAtt, self).__init__()
        self.in_time = in_time
        self.in_channels = in_channels
        self.out_channels = out_chancels
        self.ViewConv = nn.Sequential(
            nn.Conv1d(in_channels=25, out_channels=1, kernel_size=1),
            nn.BatchNorm1d(1),
            nn.Hardswish());
        self.TimeConv = nn.Sequential(
            nn.Conv1d(in_channels=self.in_time, out_channels=1, kernel_size=1),
            nn.BatchNorm1d(1),
            nn.Hardswish());
        self.SumConv = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1),
            nn.BatchNorm1d(self.out_channels),
            nn.Hardswish())
        self.ViewConv2 = nn.Conv1d(in_channels=self.out_channels, out_channels=self.in_channels, kernel_size=1)
        self.TimeConv2 = nn.Conv1d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        n, c, t, v = x.size()
        x_view = x.permute(0, 3, 1, 2)  # n,v,c,t
        x_view = x_view.contiguous().view(n, v, -1)  # n,v,c*t
        x_time = x.permute(0, 2, 1, 3)  # n,t,c,v
        x_time = x_time.contiguous().view(n, t, -1)  # n,t,c*v
        x_view = self.ViewConv(x_view)  # n,1,c*v
        x_time = self.TimeConv(x_time)  # n,1,c*t
        x_sum = torch.cat([x_time, x_view], 2).view(-1, self.in_channels, (self.in_time + 25));  # 矩阵加分;#n,c,v+t
        # x_sum=self.SumLinear(x_sum)
        x_sum = self.SumConv(x_sum)  # n,c_out,325
        # x_sum=x_sum.permute(0,2,1)#n,c,32
        # x_time_t=self.TimeLinear_t(x_sum.contiguous().view(n,-1))
        # x_time_c = self.TimeLinear_c(x_sum.contiguous().view(n, -1))
        # x_view_v=self.ViewLinear_v(x_sum.contiguous().view(n,-1))
        # x_view_c = self.ViewLinear_c(x_sum.contiguous().view(n, -1))
        # x_view=torch.mm(x_view_v.permute(1,0),x_view_c)
        # x_time=torch.mm(x_time_c.permute(1,0),x_time_t)
        x_time, x_view = torch.split(x_sum, [self.in_time, 25], dim=2)
        x_time = self.TimeConv2(x_time).mean(0, keepdims=True)
        x_view = self.ViewConv2(x_view).mean(0, keepdims=True)
        x_time = self.sigmoid(x_time)  # 1,c_out,t
        x_view = self.sigmoid(x_view)  # 1,c_int,v
        # print("time")
        # print(x_time.size())
        # print("view")
        # print(x_view.size())
        return x_time, x_view.permute(0, 2, 1)


class YczFPN(nn.Module):
    def __init__(self, epsilon=1e-4, onnx_export=False):
        super(YczFPN, self).__init__()
        self.epsilon = epsilon
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()
        # 通道扩张
        #self.x75_down_channel = nn.Sequential(
        #    Conv2dStaticSamePadding(256, 64, 1),
        #    nn.BatchNorm2d(64, momentum=0.01, eps=1e-3)
        #)
        #self.x150_down_channel = nn.Sequential(
        #    Conv2dStaticSamePadding(128, 64, 1),
        #    nn.BatchNorm2d(64, momentum=0.01, eps=1e-3)
        #)
        # self.x300_down_channel=nn.Sequential(
        #    Conv2dStaticSamePadding(64,256,1),
        #    nn.BatchNorm2d(256, momentum=0.01, eps=1e-3)
        # )

        # 融合
        self.conv_x150_down = SeparableConvBlock(32, onnx_export=onnx_export)
        self.conv_x300_down = SeparableConvBlock(32, onnx_export=onnx_export)

        self.conv_x150_down_2 = SeparableConvBlock(32, onnx_export=onnx_export)
        self.conv_x300_down_2 = SeparableConvBlock(32, onnx_export=onnx_export)

        self.conv_x75_up = SeparableConvBlock(32, onnx_export=onnx_export)

        self.conv_x75_up_2 = SeparableConvBlock(32, onnx_export=onnx_export)

        self.conv_x75_x75 = SeparableConvBlock(32, onnx_export=onnx_export)
        self.conv_x150_x150 = SeparableConvBlock(32, onnx_export=onnx_export)
        self.conv_x300_x300 = SeparableConvBlock(32, onnx_export=onnx_export)

        self.conv_x75_x75_75 = SeparableConvBlock(256, onnx_export=onnx_export)

        # 简易注意力机制的weights
        self.x75_x150 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.x150_x300 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.x300_x75 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)

        self.x75_x150_2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.x150_x300_2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.x300_x75_2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)

        self.x75_x75 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.x150_x150 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.x300_x300 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)

        self.x75_x75_x75 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)

        # 上采样
        self.x150_upsample = TimeUpsample()
        self.x300_upsample = TimeUpsample()

        self.x150_upsample_2 = TimeUpsample()
        self.x300_upsample_2 = TimeUpsample()

        self.x75_up_channel_1 = nn.Sequential(
            Conv2dStaticSamePadding(32, 72, 1),
            nn.BatchNorm2d(72, momentum=0.01, eps=1e-3)
        )
        self.x75_up_channel_2 = nn.Sequential(
            Conv2dStaticSamePadding(72, 256, 1),
            nn.BatchNorm2d(256, momentum=0.01, eps=1e-3)
        )
        #self.x75_up_channel_3 = nn.Sequential(
        #    Conv2dStaticSamePadding(32, 256, 1),
        #    nn.BatchNorm2d(256, momentum=0.01, eps=1e-3)
        #)

        # 下采样
        self.x150_dowmsample = MaxPool2dStaticSamePadding(2);
        self.x300_dowmsample = MaxPool2dStaticSamePadding(4);

        self.x150_dowmsample_2 = MaxPool2dStaticSamePadding(2);
        self.x300_dowmsample_2 = MaxPool2dStaticSamePadding(4);

        self.relu = nn.ReLU()
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x75, x150, x300):
        x75_1 = x75
        # block1
        #x75_in = self.x75_down_channel(x75)
        #x150_in = self.x150_down_channel(x150)
        # x150_in_2=self.x150_down_channel2(x150)
        # x300_in_1=self.x300_down_channel(x300)
        # x300_in_2=self.x300_down_channel2(x300)
        x75_in=x75
        x150_in=x150

        # x75与x150
        x75_x150 = self.relu(self.x75_x150)
        weight = x75_x150 / (torch.sum(x75_x150, dim=0) + self.epsilon)
        x150_td = self.conv_x150_down(self.swish(weight[0] * x150_in + weight[1] * self.x150_upsample(x75_in)))

        # x150与x300
        x150_x300 = self.relu(self.x150_x300)
        weight = x150_x300 / (torch.sum(x150_x300, dim=0) + self.epsilon)
        x300_td = self.conv_x300_down(self.swish(weight[0] * x300 + weight[1] * self.x300_upsample(x150_td)))

        # x75,x150与x300
        x300_x75 = self.relu(self.x300_x75)
        weight = x300_x75 / (torch.sum(x300_x75, dim=0) + self.epsilon)
        a_75 = weight[0] * x75_in
        b_150 = weight[1] * self.x150_dowmsample(x150_td)
        c_300 = weight[2] * self.x300_dowmsample(x300_td)
        x75_td = self.conv_x75_up(self.swish(a_75 + b_150 + c_300))

        x75_2 = x75_td

        # block2
        x75_x75 = self.relu(self.x75_x75)
        weight = x75_x75 / (torch.sum(x75_x75, dim=0) + self.epsilon)
        x75_in = self.swish(weight[0] * x75_in + weight[1] * x75_td)

        x150_x150 = self.relu(self.x150_x150)
        weight = x150_x150 / (torch.sum(x150_x150, dim=0) + self.epsilon)
        x150_in = self.swish(weight[0] * x150_in + weight[1] * x150_td)

        x300_x300 = self.relu(self.x300_x300)
        weight = x300_x300 / (torch.sum(x300_x300, dim=0) + self.epsilon)
        x300_in = self.swish(weight[0] * x300 + weight[1] * x300_td)

        # x75与x150
        x75_x150 = self.relu(self.x75_x150_2)
        weight = x75_x150 / (torch.sum(x75_x150, dim=0) + self.epsilon)
        x150_td = self.conv_x150_down_2(self.swish(weight[0] * x150_in + weight[1] * self.x150_upsample_2(x75_in)))

        # x150与x300
        x150_x300 = self.relu(self.x150_x300_2)
        weight = x150_x300 / (torch.sum(x150_x300, dim=0) + self.epsilon)
        x300_td = self.conv_x300_down_2(self.swish(weight[0] * x300_in + weight[1] * self.x300_upsample_2(x150_td)))

        # x75,x150与x300
        x300_x75 = self.relu(self.x300_x75_2)
        weight = x300_x75 / (torch.sum(x300_x75, dim=0) + self.epsilon)
        a_75 = weight[0] * x75_in
        b_150 = weight[1] * self.x150_dowmsample_2(x150_td)
        c_300 = weight[2] * self.x300_dowmsample_2(x300_td)
        x75_td = self.conv_x75_up_2(self.swish(a_75 + b_150 + c_300))

        # x75_3=x75_td

        x75_x75_75 = self.relu(self.x75_x75_x75)
        weight = x75_x75_75 / (torch.sum(x75_x75_75, dim=0) + self.epsilon)
        a = weight[0] * x75_1
        b = weight[1] * x75_2
        c = weight[2] * x75_td
        a= self.x75_up_channel_1(a+b+c)
        a= self.x75_up_channel_2(a)
        # print(a.size())
        # print(b.size())
        # print(c.size())
        x75_td = self.conv_x75_x75_75(self.swish(a))

        return x75_td


class Shift_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(Shift_tcn, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.bn = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        bn_init(self.bn2, 1)
        self.relu = nn.ReLU(inplace=True)
        self.shift_in = Shift(channel=in_channels, stride=1, init_scale=1)
        self.shift_out = Shift(channel=out_channels, stride=stride, init_scale=1)

        self.temporal_linear = nn.Conv2d(in_channels, out_channels, 1)
        nn.init.kaiming_normal(self.temporal_linear.weight, mode='fan_out')

    def forward(self, x, x_time):
        x = self.bn(x)
        # shift1
        x = self.shift_in(x.contiguous())
        n, c, t, v = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(n * v, c, t)  # n,c,t,v->n.v,c,t
        x = x * (torch.tanh(x_time) + 1);
        x = x.view(n, v, c, t).permute(0, 2, 3, 1)
        x = self.temporal_linear(x)
        x = self.relu(x)
        # shift2
        x = self.shift_out(x)
        x = self.bn2(x)
        return x


class Shift_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3, in_time=300):
        super(Shift_gcn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_time = in_time
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.Linear_weight = nn.Parameter(torch.zeros(in_channels, out_channels, requires_grad=True, device='cuda'),
                                          requires_grad=True)
        nn.init.normal_(self.Linear_weight, 0, math.sqrt(1.0 / out_channels))

        self.Linear_bias = nn.Parameter(torch.zeros(1, 1, out_channels, requires_grad=True, device='cuda'),
                                        requires_grad=True)
        nn.init.constant(self.Linear_bias, 0)

        # self.Feature_Mask = nn.Parameter(torch.ones(1,25,in_channels, requires_grad=True,device='cuda'),requires_grad=True)
        # nn.init.constant(self.Feature_Mask, 0)
        self.Feature_Att = YczAtt(in_channels=self.in_channels, out_chancels=out_channels, in_time=self.in_time)

        self.bn = nn.BatchNorm1d(25 * out_channels)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

        index_array = np.empty(25 * in_channels).astype(np.int)
        for i in range(25):
            for j in range(in_channels):
                index_array[i * in_channels + j] = (i * in_channels + j + j * in_channels) % (in_channels * 25)
        self.shift_in = nn.Parameter(torch.from_numpy(index_array), requires_grad=False)

        index_array = np.empty(25 * out_channels).astype(np.int)
        for i in range(25):
            for j in range(out_channels):
                index_array[i * out_channels + j] = (i * out_channels + j - j * out_channels) % (out_channels * 25)
        self.shift_out = nn.Parameter(torch.from_numpy(index_array), requires_grad=False)

    def forward(self, x0):
        n, c, t, v = x0.size()
        x = x0.permute(0, 2, 3, 1).contiguous()

        # shift1
        x = x.view(n * t, v * c)
        x = torch.index_select(x, 1, self.shift_in)
        x = x.contiguous().view(n, t, v, c).permute(0, 3, 1, 2)
        x_time, x_view = self.Feature_Att(x);
        x = x.permute(0, 2, 3, 1);
        x = x.contiguous().view(n * t, v, c)
        x = x * (torch.tanh(x_view) + 1)

        x = torch.einsum('nwc,cd->nwd', (x, self.Linear_weight)).contiguous()  # nt,v,c
        x = x + self.Linear_bias

        # shift2
        x = x.view(n * t, -1)
        x = torch.index_select(x, 1, self.shift_out)
        x = self.bn(x)
        x = x.view(n, t, v, self.out_channels).permute(0, 3, 1, 2)  # n,c,t,v

        x = x + self.down(x0)
        x = self.relu(x)
        return x, x_time


class MoveNet(nn.Module):
    def __init__(self, in_channels):
        # 每一个骨架视频的1/4通道发生变动
        # 变动分为两对，所以通道数目应该可以被16整除
        # 每一对位移的幅度由一个可训练的参数确定 所以其大小为2
        super(MoveNet, self).__init__()
        if in_channels % 16 == 0:
            self.shift_in = Shift(channel=int(in_channels / 4), stride=1, init_scale=1)
        # self.Movenum=nn.Parameter(torch.ones(2,requires_grad=True),
        #                              requires_grad=True)#偏移数目

    def forward(self, x):
        # n,c,t,v
        n, t, c, v = x.size()  # 获得矩阵维度

        # change = nn.Parameter(self.Movenum,
        #                      requires_grad=False)
        # index_array = np.empty(t).astype(np.int)  # 索引
        if c % 16 != 0:
            return x  # 如果不可以被16整除，那么直接返回x
        fold = c / 16  # 确定变化的通道数目
        x_change = x[:, :, 0:int(4 * fold), :]  # 取出要改变那部分

        x_change = self.shift_in(x_change)

        x[:, :, 0:int(4 * fold), :] = x_change
        return x


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, in_time=300):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = Shift_gcn(in_channels, out_channels, A, in_time=in_time)
        self.tcn1 = Shift_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        # self.MoveNet = MoveNet(int(in_channels / 4))

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        n, t, c, v = x.size()
        # print(x.size())
        # if c%4==0:
        #    x = self.MoveNet(x)
        x1, x_time = self.gcn1(x)
        x = self.tcn1(x1, x_time) + self.residual(x)
        return self.relu(x)


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 epsilon=1e-4, onnx_export=False):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.epsilon = epsilon
        self.relu = nn.ReLU()
        self.onnx_export = onnx_export
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        self.l1 = TCN_GCN_unit(3, 32, A, residual=False, in_time=300)
        self.l2 = TCN_GCN_unit(32, 32, A, in_time=300)
        self.l3 = TCN_GCN_unit(32, 32, A, in_time=300)
        self.l4 = TCN_GCN_unit(32, 32, A, stride=2, in_time=300)
        self.l5 = TCN_GCN_unit(32, 32, A, in_time=150)
        self.l6 = TCN_GCN_unit(32, 32, A, in_time=150)
        self.l7 = TCN_GCN_unit(32, 32, A, stride=2, in_time=150)
        self.l8 = TCN_GCN_unit(32, 32, A, in_time=75)
        self.l9 = TCN_GCN_unit(32, 32, A, stride=5, in_time=75)
        self.l10 = TCN_GCN_unit(32, 32, A, in_time=15)
        # self.l9 = TCN_GCN_unit(64, 64, A, stride=3, in_time=15)
        # self.l10 = TCN_GCN_unit(64, 64, A, in_time=5)
        # self.l9 = TCN_GCN_unit(64, 128, A, stride=2, in_time=300)
        # self.l10 = TCN_GCN_unit(128, 256, A, stride=2, in_time=150)



        self.x150_x300 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.x75_x150 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.x15_x75 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.x5_x15 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)


        self.conv_x15 = SeparableConvBlock(32, onnx_export=onnx_export)
        self.conv_x75 = SeparableConvBlock(32, onnx_export=onnx_export)
        self.conv_x150 = SeparableConvBlock(32, onnx_export=onnx_export)
        self.conv_x300 = SeparableConvBlock(32, onnx_export=onnx_export)


        # self.x75_down_channel = nn.Sequential(
        #    Conv2dStaticSamePadding(256, 128, 1),
        #    nn.BatchNorm2d(128, momentum=0.01, eps=1e-3)
        # )

        # self.x150_down_channel = nn.Sequential(
        #    Conv2dStaticSamePadding(128, 64, 1),
        #    nn.BatchNorm2d(64, momentum=0.01, eps=1e-3)
        # )

        #self.x5_upsample = TimeUpsample(scale_factor=3)
        self.x15_upsample = TimeUpsample(scale_factor=5)
        self.x75_upsample = TimeUpsample()
        self.x150_upsample = TimeUpsample()

        self.l2_down_channel=nn.Sequential(
            Conv2dStaticSamePadding(35,32, 1),
            nn.BatchNorm2d(32, momentum=0.01, eps=1e-3)
         )
        self.l3_down_channel = nn.Sequential(
            Conv2dStaticSamePadding(67,32, 1),
            nn.BatchNorm2d(32, momentum=0.01, eps=1e-3)
        )
        self.l4_down_channel = nn.Sequential(
            Conv2dStaticSamePadding(99,32, 1),
            nn.BatchNorm2d(32, momentum=0.01, eps=1e-3)
        )
        self.l5_down_channel = nn.Sequential(
            Conv2dStaticSamePadding(131, 32, 1),
            nn.BatchNorm2d(32, momentum=0.01, eps=1e-3)
        )
        self.l6_down_channel = nn.Sequential(
            Conv2dStaticSamePadding(163, 32, 1),
            nn.BatchNorm2d(32, momentum=0.01, eps=1e-3)
        )
        self.l7_down_channel = nn.Sequential(
            Conv2dStaticSamePadding(195, 32, 1),
            nn.BatchNorm2d(32, momentum=0.01, eps=1e-3)
        )
        self.l8_down_channel = nn.Sequential(
            Conv2dStaticSamePadding(227, 32, 1),
            nn.BatchNorm2d(32, momentum=0.01, eps=1e-3)
        )
        self.l9_down_channel = nn.Sequential(
            Conv2dStaticSamePadding(259, 32, 1),
            nn.BatchNorm2d(32, momentum=0.01, eps=1e-3)
        )
        self.l10_down_channel = nn.Sequential(
            Conv2dStaticSamePadding(291,32, 1),
            nn.BatchNorm2d(32, momentum=0.01, eps=1e-3)
        )

        self.l4_l5 = MaxPool2dStaticSamePadding(2)
        self.l7_l8 = MaxPool2dStaticSamePadding(2)
        self.l9_l10 = MaxPool2dStaticSamePadding(5)

        self.FPN = YczFPN()

        self.fc = nn.Linear(256, num_class)
        nn.init.normal(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        # x = self.l1(x)
        # x = self.l2(x)
        # x = self.l3(x)
        # x = self.l4(x)
        # x300=x
        # x = self.l5(x)
        # x = self.l6(x)
        # x = self.l7(x)
        # x150=x
        # x = self.l8(x)
        # x = self.l9(x)
        # x = self.l10(x)
        #x_in=x
        x300 = self.l1(x)
        x_in=torch.cat((x,x300),1)#64+3=67
        x300 = self.l2(self.l2_down_channel(x_in))
        x_in=torch.cat((x_in,x300),1)#67+64=131
        x300 = self.l3(self.l3_down_channel(x_in))
        # x300 = self.l3(x300)

        x_in = torch.cat((x_in, x300), 1)  # 131+64=195
        x150 = self.l4(self.l4_down_channel(x_in))
        x_in = torch.cat((self.l4_l5(x_in), x150), 1)  # 195+64=259
        x150 = self.l5(self.l5_down_channel(x_in))
        x_in = torch.cat((x_in, x150), 1)  # 259+64=323
        x150 = self.l6(self.l6_down_channel(x_in))
        # x150 = self.l6(x150)

        x_in = torch.cat((x_in, x150), 1)  # 323+64=387
        x75 = self.l7(self.l7_down_channel(x_in))
        x_in = torch.cat((self.l7_l8(x_in), x75), 1)  # 387+64=451
        x75 = self.l8(self.l8_down_channel(x_in))

        x_in = torch.cat((x_in, x75), 1)  # 461+64=515
        x15 = self.l9(self.l9_down_channel(x_in))
        x_in = torch.cat((self.l9_l10(x_in), x15), 1)  # 515+64=579
        x15 = self.l10(self.l10_down_channel(x_in))

        # x5 = self.l9(x15)
        # x5 = self.l10(x5)

        # refine
        # x5_x15 = self.relu(self.x5_x15)
        # weight = x5_x15 / (torch.sum(x5_x15, dim=0) + self.epsilon)
        # x15 = self.conv_x15(self.swish(weight[0] * x15 + weight[1] * self.x5_upsample(x5)))

        x15_x75 = self.relu(self.x15_x75)
        weight = x15_x75 / (torch.sum(x15_x75, dim=0) + self.epsilon)
        # x75 = self.conv_x75(self.swish(weight[0] * x75 + weight[1] * self.x15_upsample(x15)))
        a = weight[0] * x75 + weight[1] * self.x15_upsample(x15)
        b = self.swish(a)
        c = self.conv_x75(b)

        x75_x150 = self.relu(self.x75_x150)
        weight = x75_x150 / (torch.sum(x75_x150, dim=0) + self.epsilon)
        # print(self.x75_upsample(x75).size())
        # print(self.x75_down_channel(self.x75_upsample(x75)).size())
        x150 = self.conv_x150(self.swish(weight[0] * x150 + weight[1] * self.x75_upsample(x75)))
        # 128

        x150_x300 = self.relu(self.x150_x300)
        weight = x150_x300 / (torch.sum(x150_x300, dim=0) + self.epsilon)
        x300 = self.conv_x300(
            self.swish(weight[0] * x300 + weight[1] * self.x150_upsample(x150)))  # 64



        #x75 = self.conv_x75_2(x75)

        #x150 = self.conv_x150_2(x150)
        # x150=self.l9(x300)
        # x75=self.l10(x150)

        # x75=x
        x = self.FPN(x75=x75, x150=x150, x300=x300)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        return self.fc(x)
