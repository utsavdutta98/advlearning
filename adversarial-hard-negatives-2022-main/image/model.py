import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50, resnet18


class Model(nn.Module):
    def __init__(self, feature_dim=2):
        super(Model, self).__init__()

        self.f = []
        for name, module in resnet18().named_children():
            if name == 'conv1':
                module = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)

class MobileNetWrapper(nn.Module):
    def __init__(self, feature_dim=2):
        super(MobileNetWrapper, self).__init__()
        self.f = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=False)

        self.g = nn.Sequential(nn.Linear(1000, 256, bias=False), nn.BatchNorm1d(512),
                        nn.ReLU(inplace=True), nn.Linear(256, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = F.normalize(feature, dim=-1)
        return feature, out
