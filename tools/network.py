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


"""
Function to retrieve a batch in tensor form
	video_path_generator - generator object which returns paths to video samples
	device               - PyTorch device
	batch_size           - size of returned batch (# of consecutive frames from the video)
"""
def get_batch(video_path_generator, device, batch_size):
	# While there is no batch, try to create one
	batch, video_path = None, None
	while not torch.is_tensor(batch):
		try:
			video_path = next(video_path_generator)
			batch = create_batch(video_path = video_path, device = device, batch_size = batch_size)
		except AttributeError as Error:
			# No faces error
			print("DEBUG: {}".format(Error))
		except ValueError as Error:
			# Multiple faces error
			print("DEBUG: {}".format(Error))
			# Move the file to a special folder for videos with multiple faces
			misc.put_file_in_folder(file_path = video_path, folder = "multiple_faces")

	return batch, video_path


"""
Function for training the final fully connected layer of chosen model.
	real_video_dir - directory with real training samples (videos) 
	fake_video_dir - directory with fake training samples (videos)
	epochs         - # of epochs to train the model
	batch_size     - size of training batches (training will use both a real and fake batch of this size)
	model          - chosen model to be trained
"""
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
	# Set requires_grad to False for all layers except final FC layer
	network.freeze_layers()
	network.unfreeze_fc_layer()
	# Loss function and optimizer
	criterion = nn.BCELoss()
	optimizer = optim.SGD(network.fc_binary.parameters(), lr = 0.1, momentum = 0.9)
	# Label tensors
	real_labels = torch.full((batch_size, ), fill_value = 1, dtype = torch.float, device = device)
	real_labels = real_labels.view(-1,1)
	fake_labels = torch.full((batch_size, ), fill_value = 0, dtype = torch.float, device = device)
	fake_labels = fake_labels.view(-1,1)
	
	for epoch in range(epochs):
		iterations = 2
		for iteration in range(iterations):
			network.zero_grad()
			torch.cuda.empty_cache()

			# Training with real data
			real_batch, chosen_video = get_batch(video_path_generator = real_video_paths, device = device, batch_size = batch_size)
			# print("DEBUG: Retrieved REAL batch from '{}'".format(chosen_video))
			output_real_samples = network(real_batch.detach())
			# Delete the batch to conserve memory
			del real_batch
			torch.cuda.empty_cache()
			# Compute loss and do backpropagation
			err_real = criterion(output_real_samples, real_labels)
			err_real.backward()
			real_avg = output_real_samples.mean().item()
			# Calculating accuracy for real samples
			acc_real = np.sum(output_real_samples.cpu().detach().numpy() >= 0.5) / batch_size * 100

			# Training with fake data
			fake_batch, chosen_video = get_batch(video_path_generator = fake_video_paths, device = device, batch_size = batch_size)
			# print("DEBUG: Retrieved FAKE batch from '{}'".format(chosen_video))
			output_fake_samples = network(fake_batch.detach())
			# Delete the batch to conserve memory
			del fake_batch
			torch.cuda.empty_cache()
			# Compute loss and do backpropagation
			err_fake = criterion(output_fake_samples, fake_labels)
			err_fake.backward()
			fake_avg= output_fake_samples.mean().item()
			# Calculating accuracy for fake samples
			acc_fake = np.sum(output_fake_samples.cpu().detach().numpy() < 0.5) / batch_size * 100
			
			# Optimizer step applying gradients from real and fake batch results
			optimizer.step()

			output_string = ">> Epoch [{}/{}] Iteration [{}/{}] REAL - Acc: {:05.2f}%, MeanOut: {:3.2}, Loss: {:3.2f} | FAKE - Acc: {:05.2f}%, MeanOut: {:3.2f}, Loss: {:3.2f} | Overall Accuracy: {:05.2f}%".format(
				epoch, epochs-1, iteration, iterations-1, acc_real, real_avg, err_real.item(), acc_fake, fake_avg, err_fake.item(), (acc_real+acc_fake)/2)
			print(output_string)

		# Save the network after every epoch
		misc.save_network(network_state_dict = network.state_dict(), model_type = model)
