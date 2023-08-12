import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from resnet import ResNet18
from torch import nn, optim

def main():
	cifar_train = DataLoader(
		datasets.CIFAR10('cifar', True, download=True,
		                 transform=transforms.Compose([
			                 transforms.Resize((32, 32)),
			                 transforms.ToTensor(),
			                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
			                                      std=[0.229, 0.224, 0.225],)
		                 ])),
		batch_size=32, shuffle=True)

	cifar_test = DataLoader(
		datasets.CIFAR10('cifar', False, download=True,
		                 transform=transforms.Compose([
			                 transforms.Resize((32, 32)),
			                 transforms.ToTensor(),
			                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
			                                      std=[0.229, 0.224, 0.225], )
		                 ])),
		batch_size=32, shuffle=True
	)

	device = torch.device('cuda')
	model = ResNet18().to(device)
	# print(model)
	criteon = nn.CrossEntropyLoss().to(device)
	optimizer = optim.Adam(model.parameters(), lr=1e-3)
	for epoch in range(1000):

		model.train()
		for batchidx, (x, label) in enumerate(cifar_train):
			x, label = x.to(device), label.to(device)
			logits = model(x)
			loss = criteon(logits, label)

			# backward
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		print(epoch, loss.item())

		model.eval()
		# test
		with torch.no_grad():
			total_correct = 0
			total_num = 0
			for x, label in cifar_test:
				x, label = x.to(device), label.to(device)
				logits = model(x)
				pred = logits.argmax(dim=1)
				total_correct += torch.eq(pred, label).float().sum().item()
				total_num += x.size(0)

			acc = total_correct/total_num
			print(epoch, "acc:", acc)


if __name__ == '__main__':
	main()