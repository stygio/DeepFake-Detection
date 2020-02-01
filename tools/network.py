import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os

import tools.miscellaneous as misc
from tools.preprocessing import create_batch
from models.xception import xception


def save_model():
	pass


def train_fc_layer(real_video_dir, fake_video_dir, epochs = 1, batch_size = 16, model = "Xception"):
	# Generators for random file path in real/fake video directories
	real_video_paths = misc.get_random_file_path(real_video_dir)
	fake_video_paths = misc.get_random_file_path(fake_video_dir)
	# Pytorch device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Setup chosen CNN model for training of FC layer
	network = None
	if model == "Xception":
		network = xception(pretrained = True).to(device)
	else:
		raise Exception("Invalid model chosen.")
	criterion = nn.BCELoss()
	optimizer = optim.SGD(network.fc_binary.parameters(), lr=0.001, momentum=0.9)
	# Label tensors
	real_labels = torch.full((batch_size, ), fill_value = 1, dtype = torch.float, device = device)
	fake_labels = torch.full((batch_size, ), fill_value = 0, dtype = torch.float, device = device)
	
	for epoch in range(epochs):
		iterations = 3
		for iteration in range(iterations):
			real_batch, fake_batch = None, None
			network.zero_grad()

			# While there is no real_batch, try to create one
			while not torch.is_tensor(real_batch):
				real_video = None
				try:
					real_video = next(real_video_paths)
					real_batch = create_batch(video_path = real_video, device = device, batch_size = batch_size)
				except AttributeError as Error:
					# No faces error
					print("DEBUG: {}".format(Error))
				except ValueError as Error:
					# Multiple faces error
					print("DEBUG: {}".format(Error))
					# ToDo: Move the file to a special folder for videos with multiple faces

			# While there is no fake_batch, try to create one
			while not torch.is_tensor(fake_batch):
				fake_video = None
				try:
					fake_video = next(fake_video_paths)
					fake_batch = create_batch(video_path = fake_video, device = device, batch_size = batch_size)
				except AttributeError as Error:
					# No faces error
					print("DEBUG: {}".format(Error))
				except ValueError as Error:
					# Multiple faces error
					print("DEBUG: {}".format(Error))
					# ToDo: Move the file to a special folder for videos with multiple faces

			# Training with real data
			output_real_samples = network(real_batch)
			err_real = criterion(output_real_samples, real_labels)
			err_real.backward()
			D_real = output_real_samples.mean().item()
			# Training with fake data
			output_fake_samples = network(fake_batch)
			err_fake = criterion(output_fake_samples, fake_labels)
			err_fake.backward()
			D_fake = output_fake_samples.mean().item()
			# Optimizer step
			optimizer.step()

			# Calculating accuracy for real samples
			acc_real = np.sum(output_real_samples.cpu().detach().numpy() >= 0.5) / batch_size * 100
			# Calculating accuracy for fake samples
			acc_fake = np.sum(output_fake_samples.cpu().detach().numpy() < 0.5) / batch_size * 100

			output_string = "Epoch [{}/{}] Iteration [{}/{}] Loss(Real): {}, Loss(Fake): {}, D_real: {}, D_fake: {}, Acc(Real): {}%, Acc(Fake): {}%".format(
				epoch, epochs, iteration, iterations, err_real.item(), err_fake.item(), D_real, D_fake, acc_real, acc_fake)
			print(output_string)
