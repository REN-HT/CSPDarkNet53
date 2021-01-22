import torch as t
from torch import nn
from torch.nn import init
from torch.nn import functional as F

'''
写在开头：
卷积之后，如果要接BN操作，最好是不设置偏置，因为不起作用，而且占显卡内存。
'''

#Mish激活函数，自定义
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * (t.tanh(F.softplus(x)))

class FirstResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(FirstResidualBlock, self).__init__()
        self.left = nn.Sequential(nn.Conv2d(inchannel, outchannel//2, 1, 1, 0, bias=False),
                                  nn.BatchNorm2d(outchannel//2),
                                  Mish(),
                                  nn.Conv2d(outchannel//2, outchannel, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(outchannel),
                                  Mish())

    def forward(self, x):
        return x + self.left(x)

#the first Block is different from the rest of blocks
class FirstCSPNetBlock(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(FirstCSPNetBlock, self).__init__()
        self.front = nn.Sequential(nn.Conv2d(inchannel, outchannel, 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(outchannel),
                                   Mish())
        self.right = nn.Sequential(nn.Conv2d(outchannel, outchannel, 1, 1, 0, bias=False),
                                   nn.BatchNorm2d(outchannel),
                                   Mish())

        self.left = nn.Sequential(nn.Conv2d(outchannel, outchannel, 1, 1, 0, bias=False),
                                  nn.BatchNorm2d(outchannel),
                                  Mish(),
                                  FirstResidualBlock(outchannel, outchannel),
                                  nn.Conv2d(outchannel, outchannel, 1, 1, 0, bias=False),
                                  nn.BatchNorm2d(outchannel),
                                  Mish())
        self.cat = nn.Sequential(nn.Conv2d(outchannel * 2, outchannel, 1, 1, 0, bias=False),
                                 nn.BatchNorm2d(outchannel),
                                 Mish())

    def forward(self, x):
        x = self.front(x)
        left = self.left(x)
        right = self.right(x)
        out = t.cat([left, right], dim=1)
        out = self.cat(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(nn.Conv2d(inchannel, outchannel, 1, 1, 0, bias=False),
                                  nn.BatchNorm2d(outchannel),
                                  Mish(),
                                  nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(outchannel),
                                  Mish())

    def forward(self, x):
        return x + self.left(x)

class CSPNetBlock(nn.Module):
    def __init__(self, inchannel, outchannel, nums_block):
        super(CSPNetBlock, self).__init__()
        self.front = nn.Sequential(nn.Conv2d(inchannel, outchannel, 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(outchannel),
                                   Mish())
        self.right = nn.Sequential(nn.Conv2d(outchannel, outchannel // 2, 1, 1, 0, bias=False),
                                   nn.BatchNorm2d(outchannel // 2),
                                   Mish())
        layers = []
        for i in range(nums_block):
            layers.append(ResidualBlock(outchannel // 2, outchannel // 2))

        self.left = nn.Sequential(nn.Conv2d(outchannel, outchannel // 2, 1, 1, 0, bias=False),
                                  nn.BatchNorm2d(outchannel // 2),
                                  Mish(),
                                  nn.Sequential(*layers),
                                  nn.Conv2d(outchannel // 2, outchannel // 2, 1, 1, 0, bias=False),
                                  nn.BatchNorm2d(outchannel // 2),
                                  Mish())

        self.cat = nn.Sequential(nn.Conv2d(outchannel, outchannel, 1, 1, 0, bias=False),
                                 nn.BatchNorm2d(outchannel),
                                 Mish())

    def forward(self, x):
        x = self.front(x)
        left = self.left(x)
        right = self.right(x)
        out = t.cat([left, right], dim=1)
        out = self.cat(out)
        return out

class CSPDarkNet53(nn.Module):
    def __init__(self):
        super(CSPDarkNet53, self).__init__()
        self.prelayer = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(32),
                                      Mish())
        self.layer1 = FirstCSPNetBlock(32, 64)
        self.layer2 = CSPNetBlock(64, 128, 2)
        self.layer3 = CSPNetBlock(128, 256, 8)
        self.layer4 = CSPNetBlock(256, 512, 8)
        self.layer5 = CSPNetBlock(512, 1024, 4)

        #参数初始化
        import math
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.prelayer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x3 = self.layer3(x)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        return x3, x4, x5
