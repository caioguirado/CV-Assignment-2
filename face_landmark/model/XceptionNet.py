import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms.functional as torchFunc
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms
from torchsummary import summary

class DepthwiseSepConv2D(nn.Module):
  def __init__(self, input_channels, output_channels, k_size, **kwargs):
    super(DepthwiseSepConv2D, self).__init__()

    self.depthwise = nn.Conv2d(input_channels, input_channels, k_size, groups=input_channels, bias=False, **kwargs)
    self.pointwise = nn.Conv2d(input_channels, output_channels, 1, bias=False)
  
  def forward(self, x):
    x = self.depthwise(x)
    x = self.pointwise(x)

    return x

class EntryBlock(nn.Module):
    def __init__(self):
        super(EntryBlock, self).__init__()

        self.conv3_residual = nn.Sequential(
            DepthewiseSeperableConv2d(64, 64, 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            DepthewiseSeperableConv2d(64, 128, 3, padding = 1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(3, stride = 2, padding = 1),
        )

        self.conv3_direct = nn.Sequential(
            nn.Conv2d(64, 128, 1, stride = 2),
            nn.BatchNorm2d(128),
        )

class EntryModule(nn.Module):
  def __init__(self):
    super(EntryModule, self).__init__()

    self.conv1 = nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(0.2)
    )

    self.conv2 = nn.Sequential(
        nn.Conv2d(32, 64, 3, padding=1, bias=False), 
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.2)
    )

    self.conv3_residual = nn.Sequential(
        DepthwiseSepConv2D(64, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.2),
        DepthwiseSepConv2D(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(3, stride=2, padding=1),
    )
    
    self.conv3_direct = nn.Sequential(
        nn.Conv2d(64, 128, 1, stride=2),
        nn.BatchNorm2d(128)
    )
    self.conv4_residual = nn.Sequential(
        nn.LeakyReLU(0.2),
        DepthwiseSepConv2D(128, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2),
        DepthwiseSepConv2D(128, 256, 3, padding=1),
        nn.BatchNorm2d(256),
        nn.MaxPool2d(3, stride=2, padding=1)
    )

    self.conv4_direct = nn.Sequential(
        nn.Conv2d(128, 256, 1, stride=2),
        nn.BatchNorm2d(256),
    )

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)

    residual = self.conv3_residual(x)
    direct = self.conv3_direct(x)
    x = residual + direct

    residual = self.conv4_residual(x)
    direct = self.conv4_direct(x)
    x = residual + direct

    return x

class MiddleBlockBasicModule(nn.Module):
    def __init__(self):
        super(MiddleBlockBasicModule, self).__init__()

        self.conv1 = nn.Sequential(
            nn.LeakyReLU(0.2),
            DepthwiseSepConv2D(256, 256, 3, padding = 1),
            nn.BatchNorm2d(256)
        )
        self.conv2 = nn.Sequential(
            nn.LeakyReLU(0.2),
            DepthwiseSepConv2D(256, 256, 3, padding = 1),
            nn.BatchNorm2d(256)
        )
        self.conv3 = nn.Sequential(
            nn.LeakyReLU(0.2),
            DepthwiseSepConv2D(256, 256, 3, padding = 1),
            nn.BatchNorm2d(256)
        )

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.conv2(residual)
        residual = self.conv3(residual)

        return x + residual


class MiddleBlockModule(nn.Module):
    def __init__(self, num_blocks):
        super().__init__()
        self.block = nn.Sequential(*[MiddleBlockBasicModule() for _ in range(num_blocks)])

    def forward(self, x):
        x = self.block(x)
        return x

class ExitBlockModule(nn.Module):
    def __init__(self):
        super(ExitBlockModule, self).__init__()

        self.residual = nn.Sequential(
            nn.LeakyReLU(0.2),
            DepthwiseSepConv2D(256, 256, 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            DepthwiseSepConv2D(256, 512, 3, padding = 1),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(3, stride = 2, padding = 1)
        )

        self.direct = nn.Sequential(
            nn.Conv2d(256, 512, 1, stride = 2),
            nn.BatchNorm2d(512)
        )

        self.conv = nn.Sequential(
            DepthwiseSepConv2D(512, 512, 3, padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            DepthwiseSepConv2D(512, 1024, 3, padding = 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2)
        )

        self.dropout = nn.Dropout(0.3)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        direct = self.direct(x)
        residual = self.residual(x)
        x = direct + residual
        
        x = self.conv(x)
        x = self.avgpool(x)
        x = self.dropout(x)

        return x

class XceptionNetModule(nn.Module):
  def __init__(self, num_middle_block=6):
    super(XceptionNetModule, self).__init__()

    self.entry_block = EntryModule()
    self.middel_block = MiddleBlockModule(num_middle_block)
    self.exit_block = ExitBlockModule()

    self.fc = nn.Linear(1024, 136)

  def forward(self, x):
    x = self.entry_block(x)
    x = self.middel_block(x)
    x = self.exit_block(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return x