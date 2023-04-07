
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout

Incep = models.inception_v3(pretrained=True)
# modules = list(Incep.children())[:-1]      # delete the last fc layer.

print(Incep)