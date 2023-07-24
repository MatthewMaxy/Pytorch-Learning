import torch
import numpy as np
# 基本类型检验
# a = torch.randn(2, 3)
# print(a)
# print(a.type())
# print(type(a))
# print(isinstance(a, torch.FloatTensor))

# cuda类型检验
# data = torch.randn(2, 3)
# print(isinstance(data, torch.cuda.FloatTensor))
# data = data.cuda()
# print(isinstance(data, torch.cuda.FloatTensor))

# 不同Dimension的张量
# # dim = 0 常用于loss评估
# data_0 = torch.tensor(1.)
# print(data_0.size())
# # dim = 1 用于bias或Linear Input
# data_1 = torch.tensor([1.])     # dim=1, size = 1
# data_1_2 = torch.tensor([1., 2.])   # dim=1, size = 2
# print(data_1.size())
# print(data_1_2.size())
# data_r = torch.FloatTensor(1)   # dim=1, size = 1
# print(data_r)
# data_r = torch.FloatTensor(2)   # dim=1, size = 2
# print(data_r)
# data_np = np.array([1, 1])
# print(data_np)
# data_np_t = torch.from_numpy(data_np)
# print(data_np_t.type())
# print(data_np_t.size())
# # dim = 2 Linear Input batch
# a = torch.randn(2, 3)
# print(a)
# print(a.size())
# # dim = 3 RNN Input Batch
# a = torch.rand(2, 3, 4)
# print(a)
# print(a.size())
# # dim = 4
# a = torch.rand(2, 3, 64, 64)
# print(a)
# print(a.size())
# print(a.numel())    # 2*3*64*64
# print(a.dim())
# a = torch.Tensor(2,3)
# print(a)

# # rand_like使用
# a = torch.rand(3, 3)
# print(a)
# b = torch.rand_like(a)
# print(b)
# a_int = torch.randint(1,5,[3,3])
# print(a_int)

# # randn
# data = torch.normal(mean=torch.full([10], 0.0), std=torch.arange(1, 0, -0.1))
# print(data)
# data = torch.normal(mean=torch.full([10], 0.0), std=torch.arange(1, 0, -0.1))
# print(data)

# # arrange and linspace
# data = torch.arange(0, 10, 3)
# print(data)
# # [0 ~ 1]
# data = torch.linspace(0, 1, steps=4)
# print(data)
# # [10^0 ~ 10^-1]
# data = torch.logspace(0, -1, steps=11)
# print(data)

# # eye
# data = torch.eye(3,3)
# print(data)
# data = torch.eye(3,4)
# print(data)

# # shuffle perm
# a = torch.rand(3, 3)
# print(a)
# idx = torch.randperm(3)
# print(idx)
# b = a[idx]
# print(b)

# 索引与切片
# a = torch.rand(4, 3, 28, 28)
# print(a.shape)
# # print(a[0,0,1,2])
# # b = a[:2,1:,:28,1:28]
# # print(b.shape)
# b = a[:,:,0:28:2,0:28:2]    # 隔行隔列采样
# c = a[:,:,::2,::2]
# print(b.shape)
# print(c.shape)
# 具体某个维度采样 第一个参数为操作维度，第二个参数必须为tensor
# b = a.index_select(2, torch.arange(0,28,2)).index_select(3,torch.arange(0,28,2))
# print(b.shape)
# ···表示匹配后剩余维度全取
# b = a[2, ..., ::2]
# print(b.shape)

# # mask掩码
# x = torch.randn(3, 4)
# print(x)
# mask = x.ge(0.5)    # ge为great and equal 大于等于
# print(mask)
# mask_sel = torch.masked_select(x, mask)
# print(mask_sel)
# print(mask_sel.shape)   # 会变成一维--打平
#
# # take(先打平再操作)
# src = torch.tensor([[4, 3, 5],[6, 7, 8]])
# src_take = torch.take(src, torch.tensor([0,2,5]))
# print(src_take)

# # view reshape numel不变即可，但要考虑物理意义
# a = torch.randn(4, 1, 28, 28)
# print(a.shape)
# b = a.view(4,28*28)
# print(b.shape)
# # print(b)

# # squeeze 和 unsqueeze
# # a = torch.randn(4, 1, 28, 28)
# # for i in range(-5, 5):
# # 	print(f"unsqueeze值{i}:",a.unsqueeze(i).shape)
# b = torch.rand(32)
# f = torch.rand(4, 32, 14, 14)
# # 若想对b，f操作，首先要增维（unsqueeze）然后扩张
# b_ = b.unsqueeze(1).unsqueeze(2).unsqueeze(0)
# print(b_.shape)
#
# # squeeze
# print(b_.squeeze().shape)   # 挤压所有维度维1
# print(b_.squeeze(0).shape)
# print(b_.squeeze(1).shape)  # 无法挤压的不报错但没用
# print(b_.squeeze(-1).shape)

# # Expand / repeat
# # Expand只能1-》N
# a = torch.rand(1,32,1,1)
# print(a.shape)
# print(a.expand(-1,32,14,14).shape)  # -1表示不变
# # print(a.expand(4,32,14,14))
# print(a.expand(4,32,14,14).shape)
# # repeat 参数表示拷贝次数
# print(a.repeat(4,1,14,14).shape)
# print(a.repeat(4,32,14,14).shape)

# # .t()转置
# a = torch.rand(3,4)
# print(a)
# print(a.t())

# # transpose()和contiguous()
# a = torch.rand(4,1,32,28)
# a1 = torch.transpose(a,1,3)
# print(a1.shape)

# # 多维数组在空间内存储方式
# t = torch.arange(24).reshape(2, 3, 4)
# print(t)
# print(t.flatten())
# 故transpose后需要先contiguous()[相当于拷贝一个新的连续空间给交换后的张量]再view()
# # permute
# a = torch.rand(4,3,28,32)
# print(a.shape)
# b = a.permute(0,2,3,1)
# print(b.shape)

