import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torch import device
from tqdm.notebook import tqdm
torch.manual_seed(47)
np.random.seed(47)

class ResUnit(nn.Module):
    def __init__(self, p=64, stride=2,exp=1):
        super(ResUnit, self).__init__()
        self.c1 = nn.Conv2d(p, p*exp, kernel_size=3, stride=stride, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(p*exp)
        self.relu = nn.ReLU()
        self.c2 = nn.Conv2d( p*exp,  p*exp , kernel_size=3, stride=1, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d( p*exp)
        self.drp = nn.Dropout(0.2)
        self.relu2 = nn.ReLU()

        ## to ensure same dimension
        self.residual = nn.Sequential()
        if stride != 1 or exp!=1:
            self.residual = nn.Sequential(
                nn.Conv2d( p, p*exp, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d( p*exp )
            )

    def forward(self, x):
        out = self.relu(self.b1(self.c1(x)))
        out = self.b2(self.c2(out))
        out = self.drp(out)
        res = self.residual(x)
        out = self.relu2(out+res)
        return out


class ResNet18(nn.Module):
    def __init__(self,p,expansion,num_classes=100):
        super(ResNet18, self).__init__()
        #input size is 32x32x3
        #first layer
        # to go from 32*32 to
        p=64
        l=[
            nn.Conv2d(3, p, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(p),
            nn.ReLU(inplace=True),
          ]
        l.append(ResUnit(p,  1))
        l.append(ResUnit(p,  1))

        l.append(ResUnit(p,  2,expansion))
        p*=expansion
        l.append(ResUnit(p,  1))

        l.append(ResUnit(p,  2,expansion))
        p*=expansion
        l.append(ResUnit(p,  1))

        l.append(ResUnit(p,  2,expansion))
        p*=expansion
        l.append(ResUnit(p,  1))

        l.extend([
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
        ])

        self.model = nn.Sequential(*l)
        self.cl=nn.Linear(p,num_classes)

    def forward(self, x):
        
        x = self.model(x)
        x = self.cl(x)
        return x