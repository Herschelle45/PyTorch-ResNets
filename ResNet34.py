import torch
import torch.nn as nn 
import numpy as np 

class resblock(nn.Module):
    def __init__(self, inchannels, midchannels, outchannels, downsample):
        super(resblock,self).__init__()
        if downsample:
            self.conv1 = nn.Conv2d(inchannels,midchannels,kernel_size=3,stride=2, padding=1,bias=False)
            self.down = nn.Sequential(
                nn.Conv2d(inchannels, midchannels, kernel_size=3,stride=2, padding=1,bias=False), 
                nn.BatchNorm2d(outchannels)
            )
        else:
            self.conv1 = nn.Conv2d(inchannels,midchannels,kernel_size=3,stride=1, padding=1,bias=False)
            self.down = nn.Identity()
        self.seq = nn.Sequential(
            nn.BatchNorm2d(midchannels), 
            nn.ReLU(), 
            nn.Conv2d(midchannels,outchannels, kernel_size=3, stride=1,padding=1,bias=False), 
            nn.BatchNorm2d(outchannels)
        )
        self.relu = nn.ReLU()
    def forward(self,x):
        identity = x 
        identity = self.down(identity)
        x = self.seq(self.conv1(x))
        x+=identity
        return self.relu(x)
class ResNet34(nn.Module):
    def __init__(self,inchannels,numclasses):
        super(ResNet34,self).__init__()
        self.l0 = nn.Sequential(
                nn.Conv2d(inchannels, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
            )
        self.l1 = nn.Sequential(
            resblock(64,64,64,False),
            resblock(64,64,64,False),
            resblock(64,64,64,False),
        )
        self.l2 = nn.Sequential(
            resblock(64,128,128,True),
            resblock(128,128,128,False),
            resblock(128,128,128,False), 
            resblock(128,128,128,False),
        )
        self.l3 = nn.Sequential(
            resblock(128,256,256,True),
            resblock(256,256,256,False),
            resblock(256,256,256,False),
            resblock(256,256,256,False),
            resblock(256,256,256,False),
            resblock(256,256,256,False),
        )
        self.l4 = nn.Sequential(
            resblock(256,512,512,True),
            resblock(512,512,512,False),
            resblock(512,512,512,False),
        )
        self.finalseq = nn.Sequential(
             nn.AdaptiveAvgPool2d(1),
             nn.Flatten(), 
             nn.Linear(512,numclasses)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512,1000)
    def forward(self,x):
        x = self.l0(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        return self.finalseq(x) 
resnet34 = ResNet34(3,1000)
print(resnet34)
