import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=3, 
                                out_channels=10, 
                                kernel_size=5, 
                                stride=1, 
                                padding=0)
        self.conv_2 = nn.Conv2d(in_channels=10, 
                                out_channels=20, 
                                kernel_size=5, 
                                stride=1, 
                                padding=0)
        self.pool = nn.MaxPool2d(4, 1)
        self.fc_1 = nn.Linear(34**2 * 20, 250)
        self.fc_2 = nn.Linear(250, 120)
        self.fc_2 = nn.Linear(120, 120)
        self.fc_2 = nn.Linear(120, 120)
        self.fc_3 = nn.Linear(120, 7)

    def forward(self, x):        
        x = self.conv_1(x) #44x44
        x = F.relu(x)
        x = self.pool(x) #41x41
        x = self.conv_2(x) #37x37
        x = F.relu(x)
        x = self.pool(x) #34x34
        x = torch.flatten(x, 1)
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2(x)
        x = F.relu(x)
        x = self.fc_3(x)
        
        return x


model = CNN()