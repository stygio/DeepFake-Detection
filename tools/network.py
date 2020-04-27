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


class Network:

	def __init__(self, model_name, model_weights_path = None, training = False):
		
		# Model name
		self.model_name = model_name
		# Choose torch device
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# Setup chosen CNN model
		self.network = get_model(model_name, training, model_weights_path).to(self.device)
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


		"""
	Function for training chosen model on kaggle data.
		kaggle_dataset_path	- path to Kaggle DFDC dataset on local machine
	"""
	def train_kaggle(self, kaggle_dataset_path, epochs = 5, batch_size = 10, 
			lr = 0.00001, momentum = 0.9, only_fc_layer = False, start_folder = None):
		
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
								epoch, os.path.split(os.path.dirname(real_video_path))[1], 
								os.path.basename(fake_video_path), os.path.basename(real_video_path), 
								err, acc)
					misc.add_to_log(log_file = log_file, log_string = log_string)

			# Save the model weights after each folder
			self.save_model("kaggle_" + str(epoch), only_fc_layer)


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

