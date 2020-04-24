import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import json
import cv2
import random
import time
from tqdm import tqdm

import tools.miscellaneous as misc
from tools import preprocessing
from models.model_helpers import get_model
from tools.batch import BatchGenerator

kaggle_validation_folders = [ "dfdc_train_part_45",
							  "dfdc_train_part_46",]

kaggle_test_folders = [	"dfdc_train_part_47",
						"dfdc_train_part_48",
						"dfdc_train_part_49"]


# """
# Function for training chosen model on faceforensics data.
# 	real_video_dirs - list of directories with real training samples (videos) 
# 	fake_video_dirs - list of directories with fake training samples (videos)
# 	epochs          - # of epochs to train the model
# 	batch_size      - size of training batches (training will use both a real and fake batch of this size)
# 	model_name      - chosen model to be trained
# 	only_fc_layer   - trains all weights or only the final fully connected layer
# """
# def train_faceforensics(real_video_dirs, fake_video_dirs, model_name = "xception", model_weights_path = None, 
# 			epochs = 1, iterations = 500, batch_size = 32, batch_type = "disparate", 
# 			lr = 0.001, momentum = 0.9, only_fc_layer = True):
	
# 	# Generators for random file path in real/fake video directories
# 	real_video_paths = misc.get_random_file_path(real_video_dirs)
# 	fake_video_paths = misc.get_random_file_path(fake_video_dirs)
# 	# Pytorch device
# 	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 	# Setup chosen CNN model for training
# 	network = get_model(model_name, model_weights_path).to(device)
# 	if only_fc_layer:
# 		for param in network.parameters():
# 			param.requires_grad = False
# 		for param in network.fc.parameters():
# 			param.requires_grad = True

# 	# Loss function and optimizer
# 	criterion = nn.BCEWithLogitsLoss()
# 	# optimizer = optim.SGD(network.fc.parameters(), lr = lr, momentum = momentum)	
# 	optimizer = optim.Adam(network.fc.parameters())	

# 	if batch_type == "homogenous":
# 		# Label tensors
# 		real_labels = torch.full((batch_size, ), fill_value = 1, dtype = torch.float, device = device)
# 		real_labels = real_labels.view(-1,1)
# 		fake_labels = torch.full((batch_size, ), fill_value = 0, dtype = torch.float, device = device)
# 		fake_labels = fake_labels.view(-1,1)
# 		# Create log file
# 		log_header = "Epoch,Iteration,Acc(R),MeanOut(R),Loss(R),Acc(F),MeanOut(F),Loss(F),Acc(Overall),\n"
# 		log_file = misc.create_log(model_type = model_name, lr = lr, momentum = momentum, header_string = log_header)
# 		# Run training loop
# 		for epoch in range(epochs):
# 			for iteration in range(iterations):
# 				torch.cuda.empty_cache()

# 				network.zero_grad()
# 				# Training with real data
# 				real_batch, chosen_video = get_homogenous_batch(video_path_generator = real_video_paths, model_type = model_name, device = device, batch_size = batch_size)
# 				# print("DEBUG: Retrieved REAL batch from '{}'".format(chosen_video))
# 				output_real_samples = network(real_batch.detach())
# 				if model_name == 'inception_v3':
# 					output_real_samples = output_real_samples[0]
# 				# Delete the batch to conserve memory
# 				del real_batch
# 				torch.cuda.empty_cache()
# 				# Compute loss and do backpropagation
# 				err_real = criterion(output_real_samples, real_labels)
# 				err_real.backward()
# 				real_avg = output_real_samples.mean().item()
# 				# Calculating accuracy for real samples
# 				acc_real = np.sum(output_real_samples.cpu().detach().numpy() >= 0.5) / batch_size * 100
# 				# Optimizer step applying gradients from real batch results
# 				optimizer.step()

# 				network.zero_grad()
# 				# Training with fake data
# 				fake_batch, chosen_video = get_homogenous_batch(video_path_generator = fake_video_paths, model_type = model_name, device = device, batch_size = batch_size)
# 				# print("DEBUG: Retrieved FAKE batch from '{}'".format(chosen_video))
# 				output_fake_samples = network(fake_batch.detach())
# 				if model_name == 'inception_v3':
# 					output_fake_samples = output_fake_samples[0]
# 				# Delete the batch to conserve memory
# 				del fake_batch
# 				torch.cuda.empty_cache()
# 				# Compute loss and do backpropagation
# 				err_fake = criterion(output_fake_samples, fake_labels)
# 				err_fake.backward()
# 				fake_avg= output_fake_samples.mean().item()
# 				# Calculating accuracy for fake samples
# 				acc_fake = np.sum(output_fake_samples.cpu().detach().numpy() < 0.5) / batch_size * 100
# 				# Optimizer step applying gradients from fake batch results
# 				optimizer.step()
				

# 				# Write iteration results to console
# 				output_string = ">> Epoch [{}/{}] Iteration [{}/{}] REAL - Acc: {:05.2f}%, MeanOut: {:3.2}, Loss: {:3.2f} | FAKE - Acc: {:05.2f}%, MeanOut: {:3.2f}, Loss: {:3.2f} | Overall Accuracy: {:05.2f}%".format(
# 					epoch, epochs-1, iteration, iterations-1, acc_real, real_avg, err_real.item(), acc_fake, fake_avg, err_fake.item(), (acc_real+acc_fake)/2)
# 				print(output_string)

# 				# Write iteration results to log file
# 				log_string = "{},{},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},\n".format(
# 					epoch, iteration, acc_real, real_avg, err_real.item(), acc_fake, fake_avg, err_fake.item(), (acc_real+acc_fake)/2)
# 				misc.add_to_log(log_file = log_file, log_string = log_string)

# 			# Save the network after every epoch
# 			misc.save_network(network_state_dict = network.state_dict(), model_type = model_name)

# 	elif batch_type == "disparate":
# 		# Create log file
# 		log_header = "Epoch,Iteration,Loss,Accuracy,\n"
# 		log_file = misc.create_log(model_type = model_name, lr = lr, momentum = momentum, header_string = log_header)
# 		# Run training loop
# 		for epoch in range(epochs):
# 			for iteration in range(iterations):
# 				torch.cuda.empty_cache()

# 				network.zero_grad()
# 				# Training with mixed data
# 				batch, labels = get_disparate_batch(
# 					real_video_generator = real_video_paths, fake_video_generator = fake_video_paths, 
# 					model_type = model_name, device = device, batch_size = batch_size)
# 				output = network(batch.detach())
# 				if model_name == 'inception_v3':
# 					output = output[0]
# 				# Delete the batch to conserve memory
# 				del batch
# 				torch.cuda.empty_cache()
# 				# Compute loss and do backpropagation
# 				err = criterion(output, labels)
# 				err.backward()
# 				# Calculating accuracy for mixed samples
# 				o = output.cpu().detach().numpy()
# 				l = labels.cpu().detach().numpy()
# 				acc = np.sum(np.round(o) == np.round(l)) / batch_size * 100
# 				# Optimizer step applying gradients from results
# 				optimizer.step()			

# 				# Write iteration results to console
# 				output_string = ">> Epoch [{}/{}] Iteration [{}/{}] Loss: {:3.2f} | Accuracy: {:05.2f}%".format(
# 					epoch, epochs-1, iteration, iterations-1, err.item(), acc)
# 				print(output_string)

# 				# Write iteration results to log file
# 				log_string = "{},{},{:.2f},{:.2f},\n".format(
# 					epoch, iteration, err.item(), acc)
# 				misc.add_to_log(log_file = log_file, log_string = log_string)

# 			# Save the network after every epoch
# 			misc.save_network(network_state_dict = network.state_dict(), model_type = model_name, training_dataset = "faceforensics")
# 	else:
# 		raise Exception("Invalid batch_type: {}".format(batch_type))


class Network:

	def __init__(self, model_name = "xception", model_weights_path = None):
		
		# Model name
		self.model_name = model_name
		# Choose torch device
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# Setup chosen CNN model
		self.network = get_model(model_name, model_weights_path).to(self.device)
		for param in self.network.parameters():
				param.requires_grad = False
		# Loss function and optimizer
		self.criterion = nn.BCEWithLogitsLoss()


	"""
	Save the model weights
		dataset_name  - dataset used for training (kaggle, face_forsensics)
		batch_type	  - batch type used for training (single, dual)
		only_fc_layer - training type ie. full model/only fc layer (True, False)
	"""
	def save_model(self, dataset_name, only_fc_layer):
		
		model_dir = "models/saved_models/"
		# Construct filename
		filename = self.model_name + "_" + dataset_name
		filename += "_fc" if only_fc_layer else "_full"
		filename += misc.timestamp() + ".pt"
		filename = os.path.join(model_dir, filename)
		# Save network
		print("Saving network as '{}'".format(filename))
		torch.save(self.network.state_dict(), filename)


	# """
	# Function for training chosen model on kaggle data.
	# 	kaggle_dataset_path	- path to Kaggle DFDC dataset on local machine
	# """
	# def train_kaggle(self, kaggle_dataset_path, 
	# 		epochs = 5, iterations = 50, batch_size = 10, batch_type = "dual", 
	# 		lr = 0.001, momentum = 0.9, only_fc_layer = False):
		
	# 	# Setup gradients
	# 	if only_fc_layer:
	# 		for param in self.network.fc.parameters():
	# 			param.requires_grad = True
	# 	else:
	# 		for param in self.network.parameters():
	# 			param.requires_grad = True
	# 	# Creating batch generator
	# 	BG = BatchGenerator(self.model_name, self.device, batch_size)
	# 	# Initializing optimizer
	# 	optimizer = optim.Adam(self.network.fc.parameters())
	# 	# optimizer = optim.SGD(network.fc.parameters(), lr = lr, momentum = momentum)
	# 	# Generator for random folder paths in real/fake video directories
	# 	kaggle_folders = [os.path.join(kaggle_dataset_path, x) for x in os.listdir(kaggle_dataset_path) if x not in kaggle_test_folders]
	# 	folder_paths = misc.get_random_folder_from_list(kaggle_folders)
	# 	del kaggle_folders
	# 	# Create log file
	# 	filename = self.model_name + "_kaggle"
	# 	filename += "_fc" if only_fc_layer else "_full"
	# 	log_header = "Folder,FakeVideo,RealVideo,Epoch,Iteration,Loss,Accuracy,\n"
	# 	log_file = misc.create_log(filename, header_string = log_header)
		
	# 	# Run training loop, a folder of videos is an epoch
	# 	for epoch in range(1, epochs+1):
	# 		# Get the next folder of videos
	# 		folder_path = next(folder_paths, None)
	# 		if folder_path == None:
	# 			print("DEBUG: No more folders.")
	# 			break
	# 		# Possibly replace with a generator?
	# 		videos = os.listdir(folder_path)
	# 		random.shuffle(videos)
	# 		videos = (x for x in videos if x not in ["metadata.json", "multiple_faces", "bad_samples"])
	# 		metadata = os.path.join(folder_path, "metadata.json")
	# 		metadata = json.load(open(metadata))

	# 		# Each processed video/set of videos is an iteration
	# 		iteration = 0
	# 		while iteration <= iterations-1:
	# 			video = next(videos)
	# 			accuracies = []
	# 			errors = []
	# 			# Get label from metadata.json
	# 			label = metadata[video]['label']

	# 			start_time = time.time()

	# 			if batch_type == "single":
	# 				# Get label tensor for this video
	# 				if label == 'REAL':
	# 					labels = torch.tensor([1]*batch_size, device = self.device, requires_grad = False, dtype = torch.float)
	# 				else:
	# 					labels = torch.tensor([0]*batch_size, device = self.device, requires_grad = False, dtype = torch.float)
	# 				labels = labels.view(-1,1)
	# 				# Get batch generator
	# 				video_path = os.path.join(folder_path, video)
	# 				batch_generator = BG.single_video_batch(video_path = video_path)
	# 			elif batch_type == "dual":
	# 				# Assert the batch_size is even
	# 				assert batch_size % 2 == 0, "Uneven batch_size equal to {}".format(batch_size)
	# 				# <batch_type> dual only constructs batches from videos with <label> FAKE
	# 				if label == "FAKE":
	# 					# Get label tensor
	# 					labels = torch.tensor([0]*int(batch_size/2) + [1]*int(batch_size/2), device = self.device, requires_grad = False, dtype = torch.float)
	# 					labels = labels.view(-1,1)
	# 					video_path_1 = os.path.join(folder_path, video)
	# 					video_path_2 = os.path.join(folder_path, metadata[video]['original'])
	# 					batch_generator = BG.dual_video_batch(video_path_1, video_path_2)
	# 			else:
	# 				raise Exception("Invalid batch_type: {}".format(batch_type))

	# 			init_time = time.time() - start_time

	# 			# In case of <batch_type> dual, only run training if the handled video is fake and it's source video exists
	# 			# Otherwise, a dual batch can't be constructed from the fake and its original
	# 			if batch_type != "dual" or (label == "FAKE" and os.path.exists(video_path_2)): 
	# 				iteration += 1
	# 				batches = 0
	# 				# if batch_type == "single":
	# 				# 	print(">> Epoch [{}/{}] Iteration [{}] Processing: {}".format(
	# 				# 					epoch, epochs, iteration, video_path))
	# 				# elif batch_type == "dual":
	# 				# 	print(">> Epoch [{}/{}] Iteration [{}] Processing: {} and {}".format(
	# 				# 					epoch, epochs, iteration, 
	# 				# 					video_path_1, video_path_2))

	# 				while True:
	# 					torch.cuda.empty_cache()
	# 					# Try to get next batch
	# 					try:
	# 						# start_time_1 = time.time()

	# 						batch = next(batch_generator)

	# 						# batch_creation_time = time.time() - start_time_1
	# 						# start_time_2 = time.time()

	# 						self.network.zero_grad()
	# 						output = self.network(batch.detach())
	# 						if self.model_name == 'inception_v3':
	# 							output = output[0]
	# 						# Delete the batch to conserve memory
	# 						del batch
	# 						torch.cuda.empty_cache()
	# 						# Compute loss and do backpropagation
	# 						err = self.criterion(output, labels)
	# 						err.backward()

	# 						# Calculating accuracy for mixed samples
	# 						o = output.cpu().detach().numpy()
	# 						o = 1 / (1 + np.exp(-o)) # Applying the sigmoid to the output
	# 						l = labels.cpu().detach().numpy()
	# 						acc = np.sum(np.round(o) == np.round(l)) / batch_size * 100
	# 						# Optimizer step applying gradients from results
	# 						optimizer.step()
	# 						# Add accuracy, error to lists & increment iteration
	# 						accuracies.append(acc)
	# 						errors.append(err.item())

	# 						batches += 1

	# 						# network_time = time.time() - start_time_2
	# 						# total_batch_time = time.time() - start_time_1
	# 						# output_string = ">> Loss: {:3.2f} | Accuracy: {:05.2f}%".format(
	# 						# 	err.item(), acc)
	# 						# output_string = ">> DEBUG: <elapsed time> Batch: {:.3f} | Network: {:.3f} | Total: {:.3f}".format(
	# 						# 	batch_creation_time, network_time, total_batch_time)
	# 						# print(output_string)

	# 					# If there are no more sequences to retrieve or the file cannot be opened
	# 					except StopIteration:
	# 						if batches > 0:
	# 							err = sum(errors)/batches
	# 							acc = sum(accuracies)/batches

	# 							# Write iteration results to console
	# 							output_string = ">> Epoch [{}/{}] Iteration [{}] Loss: {:3.2f} | Accuracy: {:05.2f}%".format(
	# 								epoch, epochs, iteration, err, acc)
	# 							print(output_string)

	# 							# Write iteration results to log file
	# 							log_string = "{},{},{},{},{},{:.2f},{:.2f},\n".format(
	# 								folder_path.split("\\")[-2], 
	# 								os.path.basename(video_path_1), 
	# 								os.path.basename(video_path_2), 
	# 								epoch, iteration, err, acc)
	# 							misc.add_to_log(log_file = log_file, log_string = log_string)
	# 						break

	# 		self.save_model("kaggle", batch_type, only_fc_layer)


	# """
	# Function for training chosen model on kaggle data.
	# 	kaggle_dataset_path	- path to Kaggle DFDC dataset on local machine
	# """
	# def train_kaggle(self, kaggle_dataset_path, epochs = 5, batch_size = 10, 
	# 		lr = 0.001, momentum = 0.9, only_fc_layer = False, start_folder = None):
		
	# 	# Assert the batch_size is even
	# 	assert batch_size % 2 == 0, "Uneven batch_size equal to {}".format(batch_size)

	# 	# Setup gradients
	# 	if only_fc_layer:
	# 		for param in self.network.fc.parameters():
	# 			param.requires_grad = True
	# 	else:
	# 		for param in self.network.parameters():
	# 			param.requires_grad = True
		
	# 	# Creating batch generator
	# 	BG = BatchGenerator(self.model_name, self.device, batch_size)
		
	# 	# Initializing optimizer
	# 	optimizer = optim.Adam(self.network.fc.parameters())
	# 	# optimizer = optim.SGD(network.fc.parameters(), lr = lr, momentum = momentum)

	# 	# Get label tensor
	# 	labels = torch.tensor([0]*int(batch_size/2) + [1]*int(batch_size/2), device = self.device, requires_grad = False, dtype = torch.float)
	# 	labels = labels.view(-1,1)
		
	# 	# List of sorted folders in the kaggle directory
	# 	kaggle_folders = [os.path.join(kaggle_dataset_path, x) for x in os.listdir(kaggle_dataset_path) if x not in kaggle_validation_folders + kaggle_test_folders]
	# 	kaggle_folders = sorted(kaggle_folders, key = lambda d: int(d.split('_')[-1]))

	# 	# Create log file
	# 	filename = self.model_name + "_kaggle"
	# 	filename += "_fc" if only_fc_layer else "_full"
	# 	log_header = "Epoch,Folder,FakeVideo,RealVideo,Loss,Accuracy,\n"
	# 	log_file = misc.create_log(filename, header_string = log_header)
		
	# 	# Run training loop, a folder of videos is an epoch
	# 	for epoch in range(1, epochs+1):

	# 		# Start training from requested folder
	# 		# Useful in case of training being interrupted previously before completing an epoch
	# 		if epoch == 1 and start_folder != None:
	# 			folder_paths = kaggle_folders[start_folder:]
	# 		else:
	# 			folder_paths = kaggle_folders

	# 		# Get the next folder of videos
	# 		for folder_path in folder_paths:
	# 			videos = os.listdir(folder_path)
	# 			random.shuffle(videos)
	# 			videos = [x for x in videos if x not in ["metadata.json", "bounding_boxes", "bad_samples"]]
	# 			metadata = os.path.join(folder_path, "metadata.json")
	# 			metadata = json.load(open(metadata))
	# 			bb_path = os.path.join(folder_path, "bounding_boxes")
				
	# 			# Grab list of fake videos in the folder
	# 			fake_videos = []
	# 			for video in videos:
	# 				# Check if video is labeled as fake
	# 				if metadata[video]['label'] == 'FAKE':
	# 					fake_videos.append(video)
				
	# 			# Iterations in this folder
	# 			accuracies = []
	# 			errors = []
	# 			progress_bar = tqdm(fake_videos, desc = os.path.split(folder_path)[1])
	# 			for video in progress_bar:
	# 				# Absolute paths to videos
	# 				fake_video_path = os.path.join(folder_path, video)
	# 				real_video_path = os.path.join(folder_path, metadata[video]['original'])
	# 				# Retrieving dict of bounding boxes for faces
	# 				bounding_boxes  = os.path.join(bb_path, os.path.splitext(os.path.basename(real_video_path))[0]) + ".json"
	# 				bounding_boxes = json.load(open(bounding_boxes))
	# 				# Check if the video contains frames with multiple faces
	# 				multiple_faces = bounding_boxes['multiple_faces']

	# 				# Only run training for fake videos, with an existing original and a single face
	# 				if os.path.exists(real_video_path) and not multiple_faces:
	# 					# Get batch
	# 					batch = BG.training_batch(fake_video_path, real_video_path, boxes = bounding_boxes, epoch = epoch)
	# 					# print(batch.cpu().detach().numpy())
	# 					# print(">> Epoch [{}/{}] Processing: {} and {}".format(
	# 					# 			epoch, epochs, fake_video_path, real_video_path))

	# 					self.network.zero_grad()
	# 					output = self.network(batch.detach())
	# 					if self.model_name == 'inception_v3':
	# 						output = output[0]
	# 					# Compute loss and do backpropagation
	# 					err = self.criterion(output, labels)
	# 					err.backward()
	# 					# Optimizer step applying gradients from results
	# 					optimizer.step()

	# 					# Get loss
	# 					err = err.item()
	# 					# Calculating accuracy for mixed samples
	# 					o = output.cpu().detach().numpy()
	# 					o = 1 / (1 + np.exp(-o)) # Applying the sigmoid to the output
	# 					l = labels.cpu().detach().numpy()
	# 					acc = np.sum(np.round(o) == np.round(l)) / batch_size * 100
	# 					# Add accuracy, error to lists & increment iteration
	# 					errors.append(err)
	# 					accuracies.append(acc)
	# 					# Refresh tqdm postfix
	# 					postfix_dict = {'loss': round(np.mean(errors), 2), 'acc': round(np.mean(accuracies), 2)}
	# 					progress_bar.set_postfix(postfix_dict, refresh = False)

	# 					# Log results
	# 					log_string = "{},{},{},{},{:.2f},{:.2f},\n".format(
	# 								epoch, os.path.split(folder_path)[1], 
	# 								os.path.basename(fake_video_path), os.path.basename(real_video_path), 
	# 								err, acc)
	# 					misc.add_to_log(log_file = log_file, log_string = log_string)

	# 			# Save the model weights after each folder
	# 			self.save_model("kaggle_" + folder_path.split('_')[-1], only_fc_layer)


		"""
	Function for training chosen model on kaggle data.
		kaggle_dataset_path	- path to Kaggle DFDC dataset on local machine
	"""
	def train_kaggle(self, kaggle_dataset_path, epochs = 5, batch_size = 10, 
			lr = 0.001, momentum = 0.9, only_fc_layer = False, start_folder = None):
		
		# Assert the batch_size is even
		assert batch_size % 2 == 0, "Uneven batch_size equal to {}".format(batch_size)

		# Setup gradients
		if only_fc_layer:
			for param in self.network.fc.parameters():
				param.requires_grad = True
		else:
			for param in self.network.parameters():
				param.requires_grad = True
		
		# Creating batch generator
		BG = BatchGenerator(self.model_name, self.device, batch_size)
		
		# Initializing optimizer
		optimizer = optim.Adam(self.network.fc.parameters())
		# optimizer = optim.SGD(network.fc.parameters(), lr = lr, momentum = momentum)

		# Get label tensor
		labels = torch.tensor([0]*int(batch_size/2) + [1]*int(batch_size/2), device = self.device, requires_grad = False, dtype = torch.float)
		labels = labels.view(-1,1)
		
		# List of sorted folders in the kaggle directory
		kaggle_folders = [os.path.join(kaggle_dataset_path, x) for x in os.listdir(kaggle_dataset_path) if x not in kaggle_validation_folders + kaggle_test_folders]
		kaggle_folders = sorted(kaggle_folders, key = lambda d: int(d.split('_')[-1]))

		# Create log file
		filename = self.model_name + "_kaggle"
		filename += "_fc" if only_fc_layer else "_full"
		log_header = "Epoch,Folder,FakeVideo,RealVideo,Loss,Accuracy,\n"
		log_file = misc.create_log(filename, header_string = log_header)


		training_samples = []
		for folder_path in kaggle_folders:
			videos = os.listdir(folder_path)
			videos = [x for x in videos if x not in ["metadata.json", "bounding_boxes", "bad_samples"]]
			metadata = os.path.join(folder_path, "metadata.json")
			metadata = json.load(open(metadata))
			# Added tuples of fake and corresponding real videos to the training_samples
			for video in videos:
				# Check if video is labeled as fake
				if metadata[video]['label'] == 'FAKE':
					fake_video_path = os.path.join(folder_path, video)
					real_video_path = os.path.join(folder_path, metadata[video]['original'])
					training_samples.append((fake_video_path, real_video_path))
		
		# Run training loop
		for epoch in range(1, epochs+1):
			accuracies = []
			errors = []
			
			# Shuffle training_samples and initialize progress bar
			random.shuffle(training_samples)
			progress_bar = tqdm(training_samples, desc = "epoch {}".format(epoch))
			# Grab samples and run iterations
			for fake_video_path, real_video_path in progress_bar:
				
				# Retrieving dict of bounding boxes for faces
				bb_path = os.path.join(os.path.dirname(real_video_path), "bounding_boxes")
				bounding_boxes  = os.path.join(bb_path, os.path.splitext(os.path.basename(real_video_path))[0]) + ".json"
				bounding_boxes = json.load(open(bounding_boxes))
				# Check if the video contains frames with multiple faces
				multiple_faces = bounding_boxes['multiple_faces']

				# Only run training for fake videos, with an existing original and a single face
				if os.path.exists(real_video_path) and not multiple_faces:
					# Get batch
					batch = BG.training_batch(fake_video_path, real_video_path, boxes = bounding_boxes, epoch = epoch, n = 15)
					# print(">> Epoch [{}/{}] Processing: {} and {}".format(
					# 			epoch, epochs, fake_video_path, real_video_path))

					self.network.zero_grad()
					output = self.network(batch.detach())
					if self.model_name == 'inception_v3':
						output = output[0]
					# Compute loss and do backpropagation
					err = self.criterion(output, labels)
					err.backward()
					# Optimizer step applying gradients from results
					optimizer.step()

					# Get loss
					err = err.item()
					# Calculating accuracy for mixed samples
					o = output.cpu().detach().numpy()
					o = 1 / (1 + np.exp(-o)) # Applying the sigmoid to the output
					l = labels.cpu().detach().numpy()
					acc = np.sum(np.round(o) == np.round(l)) / batch_size * 100
					# Add accuracy, error to lists & increment iteration
					errors.append(err)
					accuracies.append(acc)
					# Refresh tqdm postfix
					postfix_dict = {'loss': round(np.mean(errors), 2), 'acc': round(np.mean(accuracies), 2)}
					progress_bar.set_postfix(postfix_dict, refresh = False)

					# Log results
					log_string = "{},{},{},{},{:.2f},{:.2f},\n".format(
								epoch, os.path.split(folder_path)[1], 
								os.path.basename(fake_video_path), os.path.basename(real_video_path), 
								err, acc)
					misc.add_to_log(log_file = log_file, log_string = log_string)

			# Save the model weights after each folder
			self.save_model("kaggle_" + epoch, only_fc_layer)


	def evaluate_kaggle(self, kaggle_dataset_path, mode, batch_size):
		
		# Creating batch generator
		BG = BatchGenerator(self.model_name, self.device, batch_size)
		
		# List of absolute paths to evaluation folders
		if mode == 'val':
			folders = kaggle_validation_folders
		elif mode == 'test':
			folders = kaggle_test_folders
		else:
			raise Exception("Invalid mode for <evaluate_kaggle>.")
		folder_paths = [os.path.join(kaggle_dataset_path, x) for x in folders]

		# Get the next folder of videos
		for folder_path in folder_paths:
			videos = os.listdir(folder_path)
			videos = [x for x in videos if x not in ["metadata.json", "bounding_boxes", "bad_samples"]]
			metadata = os.path.join(folder_path, "metadata.json")
			metadata = json.load(open(metadata))
			bb_path = os.path.join(folder_path, "bounding_boxes")
			
			# Iterations in this folder
			overall_err = []
			overall_acc = []
			progress_bar = tqdm(videos, desc = os.path.split(folder_path)[1])
			for video in progress_bar:
				# Absolute path to video
				video_path = os.path.join(folder_path, video)
				# Retrieving dict of bounding boxes for faces
				bounding_boxes  = os.path.join(bb_path, os.path.splitext(os.path.basename(video_path))[0]) + ".json"
				bounding_boxes = json.load(open(bounding_boxes))
				# Retrieve the video's label
				label = metadata[video]['label']
				# Check if the video contains frames with multiple faces
				multiple_faces = bounding_boxes['multiple_faces']

				# Only run training for fake videos, with an existing original and a single face
				if not multiple_faces:
					video_err = []
					video_acc = []
					batch_generator = BG.evaluation_batch(video_path, boxes = bounding_boxes)
					while True:
						try:
							# Get batch
							batch = next(batch_generator)
							# Feed batch through network
							output = self.network(batch.detach())
							if self.model_name == 'inception_v3':
								output = output[0]
							# Get label tensor for this video
							if label == 'REAL':
								labels = torch.tensor([1]*batch_size, device = self.device, requires_grad = False, dtype = torch.float)
							else:
								labels = torch.tensor([0]*batch_size, device = self.device, requires_grad = False, dtype = torch.float)
							labels = labels.view(-1,1)
							# Compute loss and do backpropagation
							err = self.criterion(output, labels)

							# Get loss
							err = err.item()
							# Calculating accuracy for mixed samples
							o = output.cpu().detach().numpy()
							o = 1 / (1 + np.exp(-o)) # Applying the sigmoid to the output
							l = labels.cpu().detach().numpy()
							acc = np.sum(np.round(o) == np.round(l)) / batch_size * 100
							# Add accuracy, error to lists & increment iteration
							video_err.append(err)
							video_acc.append(acc)
						
						except StopIteration:
							# Append average of video loss and accuracy
							overall_err.append(np.mean(video_err))
							overall_acc.append(np.mean(video_acc))
							break

					# Refresh tqdm postfix
					postfix_dict = {'loss': round(np.mean(overall_err), 2), 'acc': round(np.mean(overall_acc), 2)}
					progress_bar.set_postfix(postfix_dict, refresh = False)

