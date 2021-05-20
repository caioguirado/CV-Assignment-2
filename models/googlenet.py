import torch
import torchvision.models as models


model = models.googlenet(pretrained=True)
model.fc = nn.Linear(in_features=1024, out_features=7, bias=True)