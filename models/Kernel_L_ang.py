import torch
import torch.nn as nn
import torch.nn.functional as F
import models

# TSC of ori layer num for imagenet
class SDCconv2d1x1s_3_Lt1(nn.Module):
    def __init__(self, input, channel, bias=True):
        super(SDCconv2d1x1s_3_Lt1, self).__init__()
        output = int(channel)
        self.conv1_1 = nn.Conv2d(input, output, 1, bias=bias)
        self.conv1_2 = nn.Conv2d(input, output, 1, bias=bias)
        self.conv1_3 = nn.Conv2d(input, output, 1, bias=bias)
        self.conv1_4 = nn.Conv2d(input, output, 1, bias=bias)
        self.conv1_5 = nn.Conv2d(input, output, 1, bias=bias)


    def forward(self, x):
        y1_1 = self.conv1_1(x)
        y1_2 = self.conv1_2(x)
        y1_3 = self.conv1_3(x)
        y1_4 = self.conv1_4(x)
        y1_5 = self.conv1_5(x)

        y1_1 = torch.nn.functional.pad(y1_1, [0, 2, 0, 2], 'constant', 0)
        y1_2 = torch.nn.functional.pad(y1_2, [0, 2, 1, 1], 'constant', 0)
        y1_3 = torch.nn.functional.pad(y1_3, [0, 2, 2, 0], 'constant', 0)
        y1_4 = torch.nn.functional.pad(y1_4, [1, 1, 2, 0], 'constant', 0)
        y1_5 = torch.nn.functional.pad(y1_5, [2, 0, 2, 0], 'constant', 0)

        y = y1_1 + y1_2 + y1_3 + y1_4 + y1_5

        return y[:, :, 1:-1, 1:-1]

class SDCconv2d1x1s_3_Lt2(nn.Module):
    def __init__(self, input, channel, bias=True):
        super(SDCconv2d1x1s_3_Lt2, self).__init__()
        output = int(channel)
        self.conv1_1 = nn.Conv2d(input, output, 1, bias=bias)
        self.conv1_2 = nn.Conv2d(input, output, 1, bias=bias)
        self.conv1_3 = nn.Conv2d(input, output, 1, bias=bias)
        self.conv1_4 = nn.Conv2d(input, output, 1, bias=bias)
        self.conv1_5 = nn.Conv2d(input, output, 1, bias=bias)

    def forward(self, x):
        y1_1 = self.conv1_1(x)
        y1_2 = self.conv1_2(x)
        y1_3 = self.conv1_3(x)
        y1_4 = self.conv1_4(x)
        y1_5 = self.conv1_5(x)
        y1_1 = torch.nn.functional.pad(y1_1, [0, 2, 0, 2], 'constant', 0)
        y1_2 = torch.nn.functional.pad(y1_2, [1, 1, 0, 2], 'constant', 0)
        y1_3 = torch.nn.functional.pad(y1_3, [2, 0, 0, 2], 'constant', 0)
        y1_4 = torch.nn.functional.pad(y1_4, [2, 0, 1, 1], 'constant', 0)
        y1_5 = torch.nn.functional.pad(y1_5, [2, 0, 2, 0], 'constant', 0)

        y = y1_1 + y1_2 + y1_3 + y1_4 + y1_5

        return y[:, :, 1:-1, 1:-1]

class SDCconv2d1x1s_3_ang1(nn.Module):
    def __init__(self, input, channel, bias=True):
        super(SDCconv2d1x1s_3_ang1, self).__init__()
        output = int(channel)
        self.conv1_1 = nn.Conv2d(input, output, 1, bias=bias)
        self.conv1_2 = nn.Conv2d(input, output, 1, bias=bias)
        self.conv1_3 = nn.Conv2d(input, output, 1, bias=bias)
        self.conv1_4 = nn.Conv2d(input, output, 1, bias=bias)
        self.conv1_5 = nn.Conv2d(input, output, 1, bias=bias)


    def forward(self, x):
        y1_1 = self.conv1_1(x)
        y1_2 = self.conv1_2(x)
        y1_3 = self.conv1_3(x)
        y1_4 = self.conv1_4(x)
        y1_5 = self.conv1_5(x)

        y1_1 = torch.nn.functional.pad(y1_1, [1, 1, 1, 1], 'constant', 0)
        y1_2 = torch.nn.functional.pad(y1_2, [2, 0, 2, 0], 'constant', 0)
        y1_3 = torch.nn.functional.pad(y1_3, [2, 0, 0, 2], 'constant', 0)
        y1_4 = torch.nn.functional.pad(y1_4, [0, 2, 2, 0], 'constant', 0)
        y1_5 = torch.nn.functional.pad(y1_5, [1, 1, 2, 0], 'constant', 0)

        y = y1_1 + y1_2 + y1_3 + y1_4 + y1_5
        return y[:, :, 1:-1, 1:-1]
class SDCconv2d1x1s_3_ang2(nn.Module):
    def __init__(self, input, channel, bias=True):
        super(SDCconv2d1x1s_3_ang2, self).__init__()
        output = int(channel)
        self.conv1_1 = nn.Conv2d(input, output, 1, bias=bias)
        self.conv1_2 = nn.Conv2d(input, output, 1, bias=bias)
        self.conv1_3 = nn.Conv2d(input, output, 1, bias=bias)
        self.conv1_4 = nn.Conv2d(input, output, 1, bias=bias)
        self.conv1_5 = nn.Conv2d(input, output, 1, bias=bias)


    def forward(self, x):
        y1_1 = self.conv1_1(x)
        y1_2 = self.conv1_2(x)
        y1_3 = self.conv1_3(x)
        y1_4 = self.conv1_4(x)
        y1_5 = self.conv1_5(x)

        y1_1 = torch.nn.functional.pad(y1_1, [1, 1, 1, 1], 'constant', 0)
        y1_2 = torch.nn.functional.pad(y1_2, [2, 0, 0, 2], 'constant', 0)
        y1_3 = torch.nn.functional.pad(y1_3, [0, 2, 0, 2], 'constant', 0)
        y1_4 = torch.nn.functional.pad(y1_4, [0, 2, 2, 0], 'constant', 0)
        y1_5 = torch.nn.functional.pad(y1_5, [1, 1, 0, 2], 'constant', 0)

        y = y1_1 + y1_2 + y1_3 + y1_4 + y1_5
        return y[:, :, 1:-1, 1:-1]

class SDCconv2d1x1s_3_ang(nn.Module):
    def __init__(self, input, channel, bias=True):
        super(SDCconv2d1x1s_3_ang, self).__init__()
        self.conv1_1 = nn.Conv2d(input, channel, 1, bias=bias)
        self.conv1_2 = nn.Conv2d(input, channel, 1, bias=bias)
        self.conv1_3 = nn.Conv2d(input, channel, 1, bias=bias)
        self.conv1_4 = nn.Conv2d(input, channel, 1, bias=bias)
        self.conv1_5 = nn.Conv2d(input, channel, 1, bias=bias)

    def forward(self, x):
        y1 = self.conv1_1(x)
        y2 = self.conv1_2(x)
        y3 = self.conv1_3(x)
        y4 = self.conv1_4(x)
        y5 = self.conv1_5(x)
        y1 = torch.nn.functional.pad(y1, [1, 1, 1, 1], 'constant', 0)
        y2 = torch.nn.functional.pad(y2, [2, 0, 2, 0], 'constant', 0)
        y3 = torch.nn.functional.pad(y3, [2, 0, 0, 2], 'constant', 0)
        y4 = torch.nn.functional.pad(y4, [0, 2, 2, 0], 'constant', 0)
        y5 = torch.nn.functional.pad(y5, [1, 1, 2, 0], 'constant', 0)
        y = (y1+y2+y3+y4+y5)[:, :, 1:y1.shape[2]-1, 1:y1.shape[3]-1]
        return y

class SDCconv2d1x1s_3_Lt(nn.Module):
    def __init__(self, input, channel, bias=True):
        super(SDCconv2d1x1s_3_Lt, self).__init__()
        self.conv1_1 = nn.Conv2d(input, channel, 1, bias=bias)
        self.conv1_2 = nn.Conv2d(input, channel, 1, bias=bias)
        self.conv1_3 = nn.Conv2d(input, channel, 1, bias=bias)
        self.conv1_4 = nn.Conv2d(input, channel, 1, bias=bias)
        self.conv1_5 = nn.Conv2d(input, channel, 1, bias=bias)

    def forward(self, x):
        y1 = self.conv1_1(x)
        y2 = self.conv1_2(x)
        y3 = self.conv1_3(x)
        y4 = self.conv1_4(x)
        y5 = self.conv1_5(x)
        y1 = torch.nn.functional.pad(y1, [0, 2, 0, 2], 'constant', 0)
        y2 = torch.nn.functional.pad(y2, [0, 2, 1, 1], 'constant', 0)
        y3 = torch.nn.functional.pad(y3, [0, 2, 2, 0], 'constant', 0)
        y4 = torch.nn.functional.pad(y4, [1, 1, 2, 0], 'constant', 0)
        y5 = torch.nn.functional.pad(y5, [2, 0, 2, 0], 'constant', 0)
        y = (y1 + y2 + y3 + y4 + y5)[:, :, 1:y1.shape[2] - 1, 1:y1.shape[3] - 1]
        return y