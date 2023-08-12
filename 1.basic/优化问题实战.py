import numpy as np
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import Axes3D
from torch.nn import functional as F

# # 绘制图像
# def himmelblau(x):
# 	return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
#
#
# x = np.arange(-6, 6, 0.1)
# y = np.arange(-6, 6, 0.1)
# X, Y = np.meshgrid(x, y)
# Z = himmelblau([X, Y])
#
# # 绘制图像
# fig = plt.figure('himmelblau')
#
# ax = fig.add_axes(Axes3D(fig))
# ax.plot_surface(X, Y, Z)
# ax.view_init(60, -30)
# plt.show()
#
# x = torch.tensor([0., -2.], requires_grad=True)
# optimizer = torch.optim.Adam([x], lr=1e-3)
# for step in range(20000):
# 	pred = himmelblau(x)
# 	optimizer.z
# 	pred.backward()
# 	optimizer.step()
#
# 	if step % 1000 == 0:
# 		print('step {}: x={}, f(x)={}'
# 		      .format(step, x.tolist(), pred.item()))

# cross entropy
x = torch.randn(1,784)
w = torch.randn(10,784)

logits = x@w.t()
pred = F.softmax(logits, dim=1)
pred_log = torch.log(pred)

print(F.cross_entropy(logits, torch.tensor([3])))
print(F.nll_loss(pred_log,torch.tensor([3])))