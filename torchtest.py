import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2

n_l = 128
n_gf = 16
n_df = 16
nc = 1


def show_image(tensor_img):
	img = tensor_img[0][0].cpu().numpy()
	img = np.array((img+1) * 127.5, dtype="uint8")
	# img = np.transpose(img, (1, 2, 0))
	cv2.imshow("Image", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def write_image(tensor_img, img_name):
	img = tensor_img[0][0].cpu().numpy()
	img = np.array((img+1) * 127.5, dtype="uint8")
	# img = np.transpose(img, (1, 2, 0))
	cv2.imwrite("outputs/" + img_name + ".png", img)


class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()
		self.model = nn.Sequential(
			nn.ConvTranspose2d(   n_l, n_gf*4, 4, 1, 0, bias=False),
			# Output is 128 x 4 x 4
			nn.ReLU(),
			nn.ConvTranspose2d(n_gf*4, n_gf*2,  4, 2, 1, bias=False),
			# Output is 64 x 8 x 8
			nn.ReLU(),
			nn.ConvTranspose2d(n_gf*2,   n_gf,  4, 2, 2, bias=False),
			# Output is 32 x 14 x 14
			nn.ReLU(),
			nn.ConvTranspose2d(  n_gf,    n_c,  4, 2, 1, bias=False),
			# Output is 16 x 28 x 28
			nn.Tanh())

	def forward(self, x):
		x = self.model(x)
		return x


class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.model = nn.Sequential(
			nn.Conv2d(n_c, 16, 5, bias=False),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),
			nn.Conv2d(16, 32, 3, bias=False),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),
			# nn.Conv2d(32, 64, 3, bias=False),
			# nn.ReLU(),
			nn.Flatten(),
			nn.Linear(32 * 5 * 5, 64),
			nn.ReLU(),
			nn.Linear(64, 1),
			nn.Sigmoid())

	def forward(self, x):
		x = self.model(x)
		return x


transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize([0.0], [0.5])])

train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

netG = Generator().to(device)
netD = Discriminator().to(device)

criterion = nn.BCELoss()
optimizerG = optim.SGD(netG.parameters(), lr=0.001, momentum=0.9)
optimizerD = optim.SGD(netD.parameters(), lr=0.001, momentum=0.9)

# Fixed noise for G
fixed_noise = torch.randn(1, n_l, 1, 1, device=device)
real_label = 1
fake_label = 0


for epoch in range(100):
	for i, (data, _) in enumerate(train_loader, 0):
		################################
		# Update Discriminator network #
		################################
		netD.zero_grad()
		# training with real data
		real = data.to(device)
		batch_size = real.size(0)
		label = torch.full((batch_size, ), fill_value=real_label, dtype=torch.float, device=device)
		output = netD(real)
		# return real, output, label
		errD_real = criterion(output, label)
		errD_real.backward()
		D_x = output.mean().item()
		# training with fake data
		noise = torch.randn(batch_size, n_l, 1, 1, device=device)
		fake = netG(noise)
		label.fill_(fake_label)
		output = netD(fake.detach())
		errD_fake = criterion(output, label)
		errD_fake.backward()
		D_G_z1 = output.mean().item()
		# end Discriminator part
		errD = errD_real + errD_fake
		optimizerD.step()

		############################
		# Update Generator network #
		############################
		netG.zero_grad()
		label.fill_(real_label) # opposite labels for Generator cost
		output = netD(fake.detach())
		errG = criterion(output, label)
		errG.backward()
		D_G_z2 = output.mean().item()
		optimizerG.step()

		# Calculating Discriminator Accuracy (Did it recognize the samples as fake?)
		accuracy = np.sum(output.cpu().detach().numpy() < 0.5) / batch_size * 100

		output_string = "Epoch[{}/{}] Iteration[{}/{}] Loss_D: {:3.2f}, Loss_G: {:3.2f}, D(x): {:3.2f}, D(G(z)): {:3.2f}, Accuracy: {:2.2f}%".format(
			epoch, 10, i, len(train_set), errD.item(), errG.item(), D_x, D_G_z1/D_G_z2, accuracy)
		print(output_string)

		if i % 1000 == 0:
			fake_sample = netG(fixed_noise)
			write_image(fake_sample.detach(), "ep_{}_it_{}".format(epoch, i))

