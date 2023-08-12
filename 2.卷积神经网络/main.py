import torch
from torch import nn
layer = nn.Conv2d(1, 3, kernel_size=3, stride=2, padding=1)
# layer = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)
# layer = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=0)
x = torch.rand(1, 1, 28, 28)
# out = layer.forward(x)
out = layer(x)
print(out.shape)
# print(layer.weight)
# print(layer.weight.shape)
# print(layer.bias.shape)

x = out
layer = nn.MaxPool2d(2, stride=2)
out = layer(x)
print(out.shape)