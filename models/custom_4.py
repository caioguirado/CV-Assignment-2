import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Same as custom_1 but with kernel_size = 3
'''
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=3, 
                                out_channels=10, 
                                kernel_size=3, 
                                stride=1, 
                                padding=0)
        self.conv_2 = nn.Conv2d(in_channels=10, 
                                out_channels=20, 
                                kernel_size=3, 
                                stride=1, 
                                padding=0)
        self.pool = nn.MaxPool2d(4, 1)
        self.fc_1 = nn.Linear(38**2 * 20, 120)
        self.fc_2 = nn.Linear(120, 250)
        self.fc_3 = nn.Linear(250, 7)

    def forward(self, x):        
        x = self.conv_1(x) #46x46
        x = F.relu(x)
        x = self.pool(x) #43x43
        x = self.conv_2(x) #41x41
        x = F.relu(x)
        x = self.pool(x) #38x38
        x = torch.flatten(x, 1)
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2(x)
        x = F.relu(x)
        x = self.fc_3(x)
        
        return x


model = CNN()