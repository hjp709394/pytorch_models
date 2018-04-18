import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, hole=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, dilation=hole, padding=hole, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, hole=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, dilation=hole, padding=hole, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ExpNet_V1(nn.Module):
    def __init__(self, block, num_classes=10):
        super(ExpNet_V1, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.extract_feature = self._make_feature_extractor_18(block, 64)
        self.classify = self._make_classifier(64, num_classes)
        self.attend = self._make_attention(64, num_classes)

    def _make_feature_extractor_18(self, block, planes):
        holes = [1, 1, 2, 2, 4, 4, 8, 8]
        layers = []
        for hole in holes:
            layers.append(block(self.in_planes, planes, hole))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _make_classifier(self, channels, classes):
        layers = []
        layers.append(nn.BatchNorm2d(channels))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(channels))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False))
        return nn.Sequential(*layers)


    def _make_attention(self, channels, classes):
        layers = []
        layers.append(nn.BatchNorm2d(channels))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(channels))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(channels, classes * channels, kernel_size=3, padding=1, bias=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.extract_feature(out)
        out_1 = self.classify(out)
        out_2 = self.attend(out)
        out_1 = out_1.view(out_1.size(0), -1)
        out_2 = out_2.view(out_1.size(0), -1, out_1.size(1))
        out = Variable(torch.zeros(out_2.size(0), out_2.size(1)), requires_grad=False)
        out = []
        for i in range(out_1.size(0)):
            out.append(torch.mm(out_1[i:i+1,:], torch.t(out_2[i,:,:])))

        return torch.cat(out, dim=0)


def ExpNetV1(num_classes=10):
    return ExpNet_V1(BasicBlock, num_classes=num_classes)


def test():
    net = ExpNetV1()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

# test()

