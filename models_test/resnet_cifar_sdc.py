import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import models.Kernel_L_ang as SDC2

from torch.autograd import Variable

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)
class BasicBlock3x3_2stepsym(nn.Module):
    expansion = 1

    def __init__(self, kname, in_planes, planes, stride=1, option='A'):
        super(BasicBlock3x3_2stepsym, self).__init__()
        if stride != 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        else:
            self.conv1 = getattr(SDC2, kname[0])(in_planes, planes, bias=False)
            self.conv2 = getattr(SDC2, kname[1])(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlock3x3_TSC_A(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock3x3_TSC_A, self).__init__()
        if stride != 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=2, bias=False)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
class BasicBlock5x5_TSC_A(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock5x5_TSC_A, self).__init__()
        if stride != 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=5, stride=stride, padding=2, bias=False)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=5, padding=2, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=5, padding=4, bias=False)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=5, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out1 = self.layer3(out)
        out = F.avg_pool2d(out1, out1.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out1, out
class ResNet_DK(nn.Module):
    def __init__(self, kname, block, num_blocks, num_classes=10):
        super(ResNet_DK, self).__init__()
        self.in_planes = 16
        self.kname = kname
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.kname, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out1 = self.layer3(out)
        out = F.avg_pool2d(out1, out1.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out1, out

def resnet20_3x3_TSC_A():
    return ResNet(BasicBlock3x3_TSC_A, [3, 3, 3])
def resnet20_Lt1_Lt2():
    return ResNet_DK(['SDCconv2d1x1s_3_Lt1', 'SDCconv2d1x1s_3_Lt2'], BasicBlock3x3_2stepsym, [3, 3, 3])
def resnet20_ang1_ang2():
    return ResNet_DK(['SDCconv2d1x1s_3_ang1', 'SDCconv2d1x1s_3_ang2'], BasicBlock3x3_2stepsym, [3, 3, 3])

def resnet32_3x3_TSC_A():
    return ResNet(BasicBlock3x3_TSC_A, [5, 5, 5])
def resnet32_Lt1_Lt2():
    return ResNet_DK(['SDCconv2d1x1s_3_Lt1', 'SDCconv2d1x1s_3_Lt2'], BasicBlock3x3_2stepsym, [5, 5, 5])
def resnet32_ang1_ang2():
    return ResNet_DK(['SDCconv2d1x1s_3_ang1', 'SDCconv2d1x1s_3_ang2'], BasicBlock3x3_2stepsym, [5, 5, 5])

def resnet56_3x3_TSC_A():
    return ResNet(BasicBlock3x3_TSC_A, [9, 9, 9])
def resnet56_Lt1_Lt2():
    return ResNet_DK(['SDCconv2d1x1s_3_Lt1', 'SDCconv2d1x1s_3_Lt2'], BasicBlock3x3_2stepsym, [9, 9, 9])
def resnet56_ang1_ang2():
    return ResNet_DK(['SDCconv2d1x1s_3_ang1', 'SDCconv2d1x1s_3_ang2'], BasicBlock3x3_2stepsym, [9, 9, 9])


def resnet20_5x5_TSC_A():
    return ResNet(BasicBlock5x5_TSC_A, [3, 3, 3])
def resnet56_5x5_TSC_A():
    return ResNet(BasicBlock5x5_TSC_A, [9, 9, 9])


if __name__ == "__main__":
    pass