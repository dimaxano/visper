import math
import numpy as np

import gin
import torch
from torch import nn
from .depthwise import *
from ..modules.MultiHeadAttention import MultiHeadAttention

def LipRes(alpha=2, reduction=1, num_classes=256, se_block=False):
    block = lambda in_planes, planes, stride: LipResBlock(in_planes, planes, stride, reduction=reduction, se_block=se_block)    
    return ResNet(block, [alpha, alpha, alpha, alpha], num_classes=num_classes)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, reduction=1, num_classes=256):
        super(ResNet, self).__init__()
        self.reduction = float(reduction) ** 0.5
        self.num_classes = num_classes
        self.in_planes = 64 #int(16 / self.reduction)

        self.layer1 = self._make_layer(block, self.in_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512, num_classes)
        self.bnfc = nn.BatchNorm1d(num_classes)
        self.avgpool = nn.AvgPool2d(2)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        planes = int(planes)
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # 464 512 1 1
        x = x.view(x.size(0), -1)
        # 464 512
        x = self.fc(x)
        x = self.bnfc(x)

        return x


@gin.configurable
class LipNext(nn.Module):
    def __init__(self, input_dim=256, nclasses=500, frame_len=29, alpha=2, class_weight=None, se_block=False, temporal_attention=False):
        super(LipNext, self).__init__()
        self.inputDim = input_dim
        self.nClasses = nclasses
        self.frameLen = frame_len
        self.nLayers = 2
        self.alpha = alpha
        self.temporal_attention = temporal_attention

        # frontend3D
        self.frontend3D = nn.Sequential( # potential optimizable
                nn.Conv3d(1, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), groups=1, bias=False),
                nn.BatchNorm3d(64),
                nn.ReLU(True),
                # group convolution
                nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), groups=64, bias=False),
                nn.Conv3d(64, 64, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), groups=1, bias=False),
            )
        # resnet
        self.resnet34 = LipRes(self.alpha, se_block=se_block)


        if temporal_attention:
            # temporal attention module
            self.attention = MultiHeadAttention(attention_size=self.inputDim, in_features=self.inputDim, residual=True)

        # backend_conv
        self.backend_conv1 = nn.Sequential(
                nn.Conv1d(self.inputDim, 2*self.inputDim, 5, 2, 0, bias=False),
                nn.BatchNorm1d(2*self.inputDim),
                nn.ReLU(True),
                nn.MaxPool1d(2, 2),
                nn.Conv1d(2*self.inputDim, 4*self.inputDim, 5, 2, 0, bias=False),
                nn.BatchNorm1d(4*self.inputDim),
                nn.ReLU(True),
            )

        self.backend_conv2 = nn.Sequential(
                nn.Linear(4*self.inputDim, self.inputDim),
                nn.BatchNorm1d(self.inputDim),
                nn.ReLU(True),
                nn.Linear(self.inputDim, self.nClasses)
            )
        # initialize
        self._initialize_weights()

        if weight is not None:
            weight = torch.tensor(weight, dtype=torch.float32)
        self.loss = nn.CrossEntropyLoss(weight=weight)

    def forward(self, x):
        x = self.frontend3D(x)
        
        # 16, 64, 29, 22,22
        x = x.transpose(1, 2)
        # 16, 29, 64 , 22, 22
        x = x.contiguous()
        
        x = x.view(-1, 64, x.size(3), x.size(4))
        # 464, 64, 22, 22
        x = self.resnet34(x)
        # 464 256
        x = x.view(-1, self.frameLen, self.inputDim)

        # 16 29 256
        if self.temporal_attention:
            x, alignment = self.attention(x)

        x = x.transpose(1, 2)
        # 16 256 29
        x = self.backend_conv1(x)
        x = torch.mean(x,2)
        # x = x.view(-1, 4, 16, 16)
        x = self.backend_conv2(x)
        # x = x.view(-1, self.nClasses)

        return x, alignment

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                torch.nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def validator_function(self, output, labels):
        count = 0
        for i in range(labels.squeeze(1).size(0)):
            max_idx = output[i].argmax()

            if max_idx == labels[i]:
                count += 1
        
        return count


def lipnext(inputDim=256, hiddenDim=512, nClasses=500, frameLen=29, alpha=2):
    model = LipNext(inputDim=inputDim, nClasses=nClasses, frameLen=frameLen, alpha=alpha)
    return model


if __name__ == "__main__":
    from time import time
    from torch.autograd import Variable

    torch.autograd.set_detect_anomaly(True)
    model = LipNext(inputDim=256, nClasses=5, frameLen=29, alpha=2, se_block=True, temporal_attention=True)

    inp = torch.zeros(2,1,29,88,88)

    t1 = time()
    out, _ = model(inp)

    #target = torch.tensor([[1,0,0,0,0], [0,0,0,0,1]], dtype=torch.long)
    target = Variable(torch.FloatTensor(2).uniform_(0, 5).long())

    loss = model.loss(out, target)
    loss.backward()

    print(time()-t1)
    print(out.size())