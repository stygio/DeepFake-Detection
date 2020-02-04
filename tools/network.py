import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os

import tools.miscellaneous as misc
from tools.preprocessing import create_homogenous_batch, create_disparate_batch
from models.xception import xception
from models.inception_v3 import inception_v3
from models.resnet152 import resnet152
from models.resnext101 import resnext101

"""
Function to retrieve a batch in tensor form
	video_path_generator - generator object which returns paths to video samples
	model_type           - model type name
	device               - PyTorch device
	batch_size           - size of returned batch (# of consecutive frames from the video)
"""
def get_homogenous_batch(video_path_generator, model_type, device, batch_size):
	# While there is no batch, try to create one
	batch, video_path = None, None
	while not torch.is_tensor(batch):
		try:
			video_path = next(video_path_generator)
			batch = create_homogenous_batch(video_path = video_path, 
				model_type = model_type, device = device, batch_size = batch_size)
		except AttributeError as Error:
			# No faces error
			print("DEBUG: {}".format(Error))
		except ValueError as Error:
			# Multiple faces error
			print("DEBUG: {}".format(Error))
			# Move the file to a special folder for videos with multiple faces
			misc.put_file_in_folder(file_path = video_path, folder = "multiple_faces")
		except AssertionError as Error:
			# Video length error
			print("DEBUG: {}".format(Error))
			# Move the file to a special folder for short/corrupt videos
			misc.put_file_in_folder(file_path = video_path, folder = "bad_samples")

	return batch, video_path


"""
Function to retrieve a batch in tensor form
	real_video_generator - generator object which returns paths to real video samples
	fake_video_generator - generator object which returns paths to fake video samples
	model_type           - model type name
	device               - PyTorch device
	batch_size           - size of returned batch (# of consecutive frames from the video)
"""
def get_disparate_batch(real_video_generator, fake_video_generator, model_type, device, batch_size):
	batch, labels = create_disparate_batch(
		real_video_generator = real_video_generator, fake_video_generator = fake_video_generator, model_type = model_type, device = device, batch_size = batch_size)

	return batch, labels


# """
# Function for training the final fully connected layer of chosen model.
# 	real_video_dirs - list of directories with real training samples (videos) 
# 	fake_video_dirs - list of directories with fake training samples (videos)
# 	epochs          - # of epochs to train the model
# 	batch_size      - size of training batches (training will use both a real and fake batch of this size)
# 	model           - chosen model to be trained
# """
# def train_fc_layer_homogenous_batches(real_video_dirs, fake_video_dirs, epochs = 1, batch_size = 16, model = "xception"):
# 	# Generators for random file path in real/fake video directories
# 	real_video_paths = misc.get_random_file_path(real_video_dirs)
# 	fake_video_paths = misc.get_random_file_path(fake_video_dirs)
# 	# Pytorch device
# 	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 	# Setup chosen CNN model for training of FC layer
# 	network = None
# 	if model == "xception":
# 		network = xception(pretrained = True).to(device)
# 	else:
# 		raise Exception("Invalid model chosen.")
# 	# Set requires_grad to False for all layers except final FC layer
# 	network.freeze_layers()
# 	network.unfreeze_fc_layer()
# 	# Loss function and optimizer
# 	criterion = nn.BCELoss()
# 	optimizer = optim.SGD(network.fc.parameters(), lr = 0.01, momentum = 0.9)
# 	# Label tensors
# 	real_labels = torch.full((batch_size, ), fill_value = 1, dtype = torch.float, device = device)
# 	real_labels = real_labels.view(-1,1)
# 	fake_labels = torch.full((batch_size, ), fill_value = 0, dtype = torch.float, device = device)
# 	fake_labels = fake_labels.view(-1,1)

# 	log_header = "Epoch,Iteration,Acc(R),MeanOut(R),Loss(R),Acc(F),MeanOut(F),Loss(F),Acc(Overall),\n"
# 	log_file = misc.create_log(model_type = model, header_string = log_header)

# 	for epoch in range(epochs):
# 		iterations = 500
# 		for iteration in range(iterations):
# 			torch.cuda.empty_cache()

# 			network.zero_grad()
# 			# Training with real data
# 			real_batch, chosen_video = get_homogenous_batch(video_path_generator = real_video_paths, model_type = model, device = device, batch_size = batch_size)
# 			# print("DEBUG: Retrieved REAL batch from '{}'".format(chosen_video))
# 			output_real_samples = network(real_batch.detach())
# 			# Delete the batch to conserve memory
# 			del real_batch
# 			torch.cuda.empty_cache()
# 			# Compute loss and do backpropagation
# 			err_real = criterion(output_real_samples, real_labels)
# 			err_real.backward()
# 			real_avg = output_real_samples.mean().item()
# 			# Calculating accuracy for real samples
# 			acc_real = np.sum(output_real_samples.cpu().detach().numpy() >= 0.5) / batch_size * 100
# 			# Optimizer step applying gradients from real batch results
# 			optimizer.step()

# 			network.zero_grad()
# 			# Training with fake data
# 			fake_batch, chosen_video = get_homogenous_batch(video_path_generator = fake_video_paths, model_type = model, device = device, batch_size = batch_size)
# 			# print("DEBUG: Retrieved FAKE batch from '{}'".format(chosen_video))
# 			output_fake_samples = network(fake_batch.detach())
# 			# Delete the batch to conserve memory
# 			del fake_batch
# 			torch.cuda.empty_cache()
# 			# Compute loss and do backpropagation
# 			err_fake = criterion(output_fake_samples, fake_labels)
# 			err_fake.backward()
# 			fake_avg= output_fake_samples.mean().item()
# 			# Calculating accuracy for fake samples
# 			acc_fake = np.sum(output_fake_samples.cpu().detach().numpy() < 0.5) / batch_size * 100
# 			# Optimizer step applying gradients from fake batch results
# 			optimizer.step()
			

# 			# Write iteration results to console
# 			output_string = ">> Epoch [{}/{}] Iteration [{}/{}] REAL - Acc: {:05.2f}%, MeanOut: {:3.2}, Loss: {:3.2f} | FAKE - Acc: {:05.2f}%, MeanOut: {:3.2f}, Loss: {:3.2f} | Overall Accuracy: {:05.2f}%".format(
# 				epoch, epochs-1, iteration, iterations-1, acc_real, real_avg, err_real.item(), acc_fake, fake_avg, err_fake.item(), (acc_real+acc_fake)/2)
# 			print(output_string)

# 			# Write iteration results to log file
# 			log_string = "{},{},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},\n".format(
# 				epoch, iteration, acc_real, real_avg, err_real.item(), acc_fake, fake_avg, err_fake.item(), (acc_real+acc_fake)/2)
# 			misc.add_to_log(log_file = log_file, log_string = log_string)

# 		# Save the network after every epoch
# 		misc.save_network(network_state_dict = network.state_dict(), model_type = model)


# """
# Function for training the final fully connected layer of chosen model.
# 	real_video_dirs - list of directories with real training samples (videos) 
# 	fake_video_dirs - list of directories with fake training samples (videos)
# 	epochs          - # of epochs to train the model
# 	batch_size      - size of training batches (training will use both a real and fake batch of this size)
# 	model           - chosen model to be trained
# """
# def train_fc_layer_disparate_batches(real_video_dirs, fake_video_dirs, epochs = 1, batch_size = 16, model = "xception"):
# 	# Generators for random file path in real/fake video directories
# 	real_video_paths = misc.get_random_file_path(real_video_dirs)
# 	fake_video_paths = misc.get_random_file_path(fake_video_dirs)
# 	# Pytorch device
# 	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 	# Setup chosen CNN model for training of FC layer
# 	network = None
# 	if model == "xception":
# 		network = xception(pretrained = True).to(device)
# 	else:
# 		raise Exception("Invalid model chosen.")
# 	# Set requires_grad to False for all layers except final FC layer
# 	network.freeze_layers()
# 	network.unfreeze_fc_layer()
# 	# Loss function and optimizer
# 	criterion = nn.BCELoss()
# 	optimizer = optim.SGD(network.fc.parameters(), lr = 0.01, momentum = 0.9)	

# 	log_header = "Epoch,Iteration,Loss,Accuracy,\n"
# 	log_file = misc.create_log(model_type = model, header_string = log_header)

# 	for epoch in range(epochs):
# 		iterations = 500
# 		for iteration in range(iterations):
# 			torch.cuda.empty_cache()

# 			network.zero_grad()
# 			# Training with mixed data
# 			batch, labels = get_disparate_batch(
# 				real_video_generator = real_video_paths, fake_video_generator = fake_video_paths, 
# 				model_type = model, device = device, batch_size = batch_size)
# 			output = network(batch.detach())
# 			# Delete the batch to conserve memory
# 			del batch
# 			torch.cuda.empty_cache()
# 			# Compute loss and do backpropagation
# 			err = criterion(output, labels)
# 			err.backward()
# 			# Calculating accuracy for mixed samples
# 			o = output.cpu().detach().numpy()
# 			l = labels.cpu().detach().numpy()
# 			acc = np.sum(np.round(o) == np.round(l)) / batch_size * 100
# 			# Optimizer step applying gradients from results
# 			optimizer.step()			

# 			# Write iteration results to console
# 			output_string = ">> Epoch [{}/{}] Iteration [{}/{}] Loss: {:3.2f} | Accuracy: {:05.2f}%".format(
# 				epoch, epochs-1, iteration, iterations-1, err.item(), acc)
# 			print(output_string)

# 			# Write iteration results to log file
# 			log_string = "{},{},{:.2f},{:.2f},\n".format(
# 				epoch, epochs-1, iteration, iterations-1, err.item(), acc)
# 			misc.add_to_log(log_file = log_file, log_string = log_string)

# 		# Save the network after every epoch
# 		misc.save_network(network_state_dict = network.state_dict(), model_type = model)


"""
Function for training the final fully connected layer of chosen model.
	real_video_dirs - list of directories with real training samples (videos) 
	fake_video_dirs - list of directories with fake training samples (videos)
	epochs          - # of epochs to train the model
	batch_size      - size of training batches (training will use both a real and fake batch of this size)
	model           - chosen model to be trained
"""
def train_fc_layer(real_video_dirs, fake_video_dirs, 
					epochs = 1, iterations = 500, batch_size = 32, batch_type = "disparate", 
					lr = 0.01, momentum = 0.9, model = "xception"):
	
	# Generators for random file path in real/fake video directories
	real_video_paths = misc.get_random_file_path(real_video_dirs)
	fake_video_paths = misc.get_random_file_path(fake_video_dirs)
	# Pytorch device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Setup chosen CNN model for training of FC layer
	network = None
	if model == "xception":
		network = xception(pretrained = True).to(device)
	elif model == "inception_v3":
		network = inception_v3(pretrained = True).to(device)
	elif model == "resnet152":
		network = resnet152(pretrained = True).to(device)
	elif model == "resnext101":
		network = resnext101(pretrained = True).to(device)
	else:
		raise Exception("Invalid model chosen.")
	# Set requires_grad to False for all layers except final FC layer
	for param in network.parameters():
		param.requires_grad = False
	for param in network.fc.parameters():
		param.requires_grad = True

	# Loss function and optimizer
	criterion = nn.BCEWithLogitsLoss()
	optimizer = optim.SGD(network.fc.parameters(), lr = lr, momentum = momentum)	

	if batch_type == "homogenous":
		# Label tensors
		real_labels = torch.full((batch_size, ), fill_value = 1, dtype = torch.float, device = device)
		real_labels = real_labels.view(-1,1)
		fake_labels = torch.full((batch_size, ), fill_value = 0, dtype = torch.float, device = device)
		fake_labels = fake_labels.view(-1,1)
		# Create log file
		log_header = "Epoch,Iteration,Acc(R),MeanOut(R),Loss(R),Acc(F),MeanOut(F),Loss(F),Acc(Overall),\n"
		log_file = misc.create_log(model_type = model, lr = lr, momentum = momentum, header_string = log_header)
		# Run training loop
		for epoch in range(epochs):
			for iteration in range(iterations):
				torch.cuda.empty_cache()

				network.zero_grad()
				# Training with real data
				real_batch, chosen_video = get_homogenous_batch(video_path_generator = real_video_paths, model_type = model, device = device, batch_size = batch_size)
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
				# Optimizer step applying gradients from real batch results
				optimizer.step()

				network.zero_grad()
				# Training with fake data
				fake_batch, chosen_video = get_homogenous_batch(video_path_generator = fake_video_paths, model_type = model, device = device, batch_size = batch_size)
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
				# Optimizer step applying gradients from fake batch results
				optimizer.step()
				

				# Write iteration results to console
				output_string = ">> Epoch [{}/{}] Iteration [{}/{}] REAL - Acc: {:05.2f}%, MeanOut: {:3.2}, Loss: {:3.2f} | FAKE - Acc: {:05.2f}%, MeanOut: {:3.2f}, Loss: {:3.2f} | Overall Accuracy: {:05.2f}%".format(
					epoch, epochs-1, iteration, iterations-1, acc_real, real_avg, err_real.item(), acc_fake, fake_avg, err_fake.item(), (acc_real+acc_fake)/2)
				print(output_string)

				# Write iteration results to log file
				log_string = "{},{},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},\n".format(
					epoch, iteration, acc_real, real_avg, err_real.item(), acc_fake, fake_avg, err_fake.item(), (acc_real+acc_fake)/2)
				misc.add_to_log(log_file = log_file, log_string = log_string)

			# Save the network after every epoch
			misc.save_network(network_state_dict = network.state_dict(), model_type = model)

	elif batch_type == "disparate":
		# Create log file
		log_header = "Epoch,Iteration,Loss,Accuracy,\n"
		log_file = misc.create_log(model_type = model, lr = lr, momentum = momentum, header_string = log_header)
		# Run training loop
		for epoch in range(epochs):
			for iteration in range(iterations):
				torch.cuda.empty_cache()

				network.zero_grad()
				# Training with mixed data
				batch, labels = get_disparate_batch(
					real_video_generator = real_video_paths, fake_video_generator = fake_video_paths, 
					model_type = model, device = device, batch_size = batch_size)
				output = network(batch.detach())
				# Delete the batch to conserve memory
				del batch
				torch.cuda.empty_cache()
				# Compute loss and do backpropagation
				err = criterion(output, labels)
				err.backward()
				# Calculating accuracy for mixed samples
				o = output.cpu().detach().numpy()
				l = labels.cpu().detach().numpy()
				acc = np.sum(np.round(o) == np.round(l)) / batch_size * 100
				# Optimizer step applying gradients from results
				optimizer.step()			

				# Write iteration results to console
				output_string = ">> Epoch [{}/{}] Iteration [{}/{}] Loss: {:3.2f} | Accuracy: {:05.2f}%".format(
					epoch, epochs-1, iteration, iterations-1, err.item(), acc)
				print(output_string)

				# Write iteration results to log file
				log_string = "{},{},{:.2f},{:.2f},\n".format(
					epoch, iteration, err.item(), acc)
				misc.add_to_log(log_file = log_file, log_string = log_string)

			# Save the network after every epoch
			misc.save_network(network_state_dict = network.state_dict(), model_type = model)

	else:
		raise Exception("Invalid batch_type: {}".format(batch_type))