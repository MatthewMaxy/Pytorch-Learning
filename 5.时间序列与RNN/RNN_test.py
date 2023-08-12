from torch import nn
import torch
# RNN
"""
rnn = nn.RNN(input_size=100, hidden_size=10, num_layers=2)

x = torch.randn(5, 3, 100)
out, h = rnn(x, torch.zeros(2, 3, 10))

print(out.shape, h.shape)
"""

# RNNCell
x = torch.randn(5, 3, 100)
cell1 = nn.RNNCell(100, 30)
cell2 = nn.RNNCell(30, 20)
h1 = torch.zeros(3, 30)
h2 = torch.zeros(3, 20)
for xt in x:
	h1 = cell1(xt, h1)
	h2 = cell2(h1, h2)

print(h2.shape)