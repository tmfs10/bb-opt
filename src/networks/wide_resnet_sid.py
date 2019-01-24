import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Linear, ReLU, Softplus,Dropout

import sys
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1, do_batch_norm=False):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        self.conv3 = None
        if stride != 1 or in_planes != planes:
            self.conv3 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True)
            self.shortcut = nn.Sequential(
                self.conv3,
            )

        self.do_batch_norm = do_batch_norm

    def reset_parameters(self):
        nets = [self.conv1,
                self.conv2,
                self.conv3
                ]
        for k in nets:
            if k is not None:
                k.reset_parameters()

    def forward(self, x):
        if self.do_batch_norm:
            out = self.conv1(F.relu(self.bn1(x)))
            #out = self.conv1(F.relu(self.bn1(x)))
            out = self.conv2(F.relu(self.bn2(out)))
        else:
            out = self.conv1(F.relu(x))
            out = self.conv2(F.relu(out))
        out += self.shortcut(x)

        return out

class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate=0.3, fc_sampling=False, do_batch_norm=False):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16
        print(depth)

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = int((depth-4)/6)
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]
        self.non_linearity = ReLU()
        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1, do_batch_norm=do_batch_norm)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2, do_batch_norm=do_batch_norm)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2, do_batch_norm=do_batch_norm)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.nStages = nStages
        self.fc_sampling = fc_sampling
        if not fc_sampling:
            self.linear = nn.Linear(nStages[3],2)

        self.do_batch_norm = do_batch_norm

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride, do_batch_norm):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride, do_batch_norm=do_batch_norm))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def reset_parameters(self):
        nets = [self.conv1,
                self.layer1,
                self.layer2,
                self.layer3,
                self.linear,
                ]
        for k in nets:
            k.reset_parameters()

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(out)
        if self.do_batch_norm:
            out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)

        if not self.fc_sampling:
            out = self.linear(out)
            mean = torch.sigmoid(out[:,0])
            variance = torch.sigmoid(out[:,1])*0.1+1e-5
            return mean, variance
        else:
            return out

#if __name__ == '__main__':
#    net=Wide_ResNet(28, 10, 0.3, 10)
#    y = net(Variable(torch.randn(1,3,32,32)))

#    print(y.size())
