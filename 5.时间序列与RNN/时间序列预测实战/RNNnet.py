from torch import nn, optim
import torch
import numpy as np
import matplotlib.pyplot as plt

input_size = 1
hidden_size = 16
output_size = 1
lr = 0.01
num_time_steps = 50

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.rnn = nn.RNN(
			input_size=input_size,
			hidden_size=hidden_size,
			num_layers=1,
			batch_first=True
		)
		self.linear = nn.Linear(hidden_size, output_size)

	def forward(self, _x, hidden):
		# _x [1, 49, 1]
		out, hidden = self.rnn(_x, hidden)
		# out [1, 49, 16] h1,h2,...ht 最上面一层
		# hidden [1, 1, 16] ht 最右边
		out = out.view(-1, hidden_size)

		# 利用所有的h做全连接 out => [49, 16]
		out = self.linear(out)
		# out=>[49, 1]   =添加一维(与y运算)=> [1, 49, 1]
		out = out.unsqueeze(dim=0)
		return out, hidden


model = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr)

hidden_prev = torch.zeros(1, 1, hidden_size)

for iter in range(60000):

	# 起点随机0-10中的一个
	start = np.random.randint(10, size=1)[0]

	# 创建一个包含 50 个均匀间隔值的数组
	# 这些值从 start 开始，每个时间步长为 0.2，一直到比 start 大 10 的位置结束
	time_steps = np.linspace(start, start + 10, num_time_steps)

	data = np.sin(time_steps)
	data = data.reshape(num_time_steps, 1)
	# x [batch, sequence_length, input_dim ]
	x = torch.tensor(data[:-1]).float().view(1, num_time_steps - 1, 1)
	y = torch.tensor(data[1:]).float().view(1, num_time_steps - 1, 1)

	output, hidden_prev = model(x, hidden_prev)
	hidden_prev = hidden_prev.detach()

	loss = criterion(output, y)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	if iter % 100 == 0:
		print("Iteration:{} loss {}".format(iter, loss.item()))

start = np.random.randint(10, size=1)[0]
time_steps = np.linspace(start, start + 10, num_time_steps)
data = np.sin(time_steps)
data = data.reshape(num_time_steps, 1)
x = torch.tensor(data[:-1]).float().view(1, num_time_steps - 1, 1)
y = torch.tensor(data[1:]).float().view(1, num_time_steps - 1, 1)

predictions = []
input = x[:, 0, :]
for _ in range(x.shape[1]):
	input = input.view(1, 1, 1)
	(pred, hidden_prev) = model(input, hidden_prev)
	input = pred
	predictions.append(pred.detach().numpy().ravel()[0])

x = x.data.numpy().ravel()
y = y.data.numpy()
plt.scatter(time_steps[:-1], x.ravel(), s=90)
plt.plot(time_steps[:-1], x.ravel())

plt.scatter(time_steps[1:], predictions)
plt.show()