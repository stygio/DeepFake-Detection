import torch
import torch.nn as nn
import torch.optim as optim

import os
import json
import cv2
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import lines
from sklearn.metrics import balanced_accuracy_score

from tools.batch import BatchGenerator
from models import model_helpers
import tools.miscellaneous as misc
from radam.radam import RAdam


class Network:

	def __init__(self, model_name, model_weights_path = None):
		
		# Model name
		self.model_name = model_name
		# Choose torch device
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# Setup chosen CNN model
		self.network = model_helpers.get_model(model_name, model_weights_path).to(self.device)
		# Loss function and optimizer
		self.criterion = nn.BCEWithLogitsLoss()


	"""
	Save the model weights
		dataset_name  	 - dataset used for training (kaggle, face_forsensics)
		training_level - training type/level {classifier, higher, lower}
	"""
	def save_model(self, dataset_name, training_level):
		
		model_dir = "models/saved_models/"
		# Construct filename
		filename = self.model_name + "_" + dataset_name
		filename += training_level
		filename += misc.timestamp() + ".pt"
		filename = os.path.join(model_dir, filename)
		# Save network
		print("Saving network as '{}'".format(filename))
		torch.save(self.network.state_dict(), filename)


	"""
	Plots the gradients flowing through different layers in the net during training.
	Can be used for checking for possible gradient vanishing / exploding problems.
	"""
	def plot_grad_flow(self):

		named_parameters = self.network.named_parameters()
		ave_grads = []
		max_grads= []
		layers = []
		for n, p in named_parameters:
			if(p.requires_grad) and ("bias" not in n):
				layers.append(n)
				ave_grads.append(p.grad.abs().mean())
				max_grads.append(p.grad.abs().max())
		plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
		plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
		plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
		plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
		plt.xlim(left=0, right=len(ave_grads))
		plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
		plt.xlabel("Layers")
		plt.ylabel("average gradient")
		plt.title("Gradient flow")
		plt.grid(True)
		plt.legend([lines.Line2D([0], [0], color="c", lw=4),
					lines.Line2D([0], [0], color="b", lw=4),
					lines.Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
		plt.show()


	# """
	# Function for training the network
	# 	dataset    		- choice of dataset to train on
	# 	dataset_path	- path to dataset on local machine
	# """
	# def train(self, dataset, dataset_path, epochs = 10, batch_size = 14, 
	# 		lr = 0.001, momentum = 0.9, training_level = 'classifier'):
		
	# 	# Display network parameter division ratios
	# 	model_helpers.count_parameters(self.network)

	# 	# Assert the batch_size is even
	# 	assert batch_size % 2 == 0, "Uneven batch_size: {}".format(batch_size)
	# 	# Assert valid dataset choise
	# 	assert dataset in ['faceforensics', 'kaggle'], "Invalid dataset choice: {}".format(dataset)
	# 	# Assert valid dataset patch
	# 	assert os.path.isdir(dataset_path), "Invalid dataset path: {}".format(dataset_path)
	# 	# Assert valid training_level
	# 	assert training_level in ['classifier', 'higher', 'lower'], "Invalid training level choice: {}".format(training_level)
		
	# 	# Creating batch generator
	# 	BG = BatchGenerator(self.model_name, self.device, batch_size)

	# 	# Initializing optimizer with appropriate lr
	# 	classifier_lr = lr
	# 	higher_level_lr = 0.1 * classifier_lr
	# 	lower_level_lr = 0.1 * higher_level_lr
	# 	# optimizer = optim.SGD([
	# 	# 	{'params': self.network.classifier_parameters(), 'lr': classifier_lr},
	# 	# 	{'params': self.network.higher_level_parameters(), 'lr': higher_level_lr},
	# 	# 	{'params': self.network.lower_level_parameters(), 'lr': lower_level_lr}
	# 	# ], momentum = momentum)
	# 	optimizer = RAdam([
	# 		{'params': self.network.classifier_parameters(), 'lr': classifier_lr},
	# 		{'params': self.network.higher_level_parameters(), 'lr': higher_level_lr},
	# 		{'params': self.network.lower_level_parameters(), 'lr': lower_level_lr}], weight_decay = 1e-4)

	# 	# Get label tensor
	# 	labels = torch.tensor([0]*int(batch_size/2) + [1]*int(batch_size/2), device = self.device, requires_grad = False, dtype = torch.float)
	# 	labels = labels.view(-1,1)
		
	# 	# Create log file w/ train/val information per epoch
	# 	filename = self.model_name + "_" + dataset + "_"
	# 	filename += training_level + "_epochs"
	# 	log_header = "epoch,train_loss,train_acc,val_loss,val_acc\n"
	# 	epoch_log = misc.create_log(filename, header_string = log_header)
	# 	# Create log file w/ information from validation runs
	# 	filename = self.model_name + "_" + dataset + "_"
	# 	filename += training_level + "_validation"
	# 	log_header = "Epoch,File,Label,AvgOutput,Loss,Acc\n"
	# 	validation_log = misc.create_log(filename, header_string = log_header)

	# 	# Assemble a list of training samples
	# 	training_samples = []
	# 	if dataset == 'faceforensics':
	# 		# List of sorted folders in the faceforensics directory
	# 		original_sequences = os.path.join(dataset_path, 'original_sequences')
	# 		real_folder = os.path.join(original_sequences, 'c23', 'videos')
	# 		manipulated_sequences = os.path.join(dataset_path, 'manipulated_sequences')
	# 		fake_folders = [os.path.join(manipulated_sequences, x) for x in os.listdir(manipulated_sequences)]
	# 		fake_folders = [os.path.join(x, 'c23', 'videos') for x in fake_folders]
	# 		# Collect training samples
	# 		for folder_path in fake_folders:
	# 			videos = os.listdir(folder_path)
	# 			videos = [x for x in videos if x not in ["metadata.json", "bounding_boxes", "bad_samples", "multiple_faces", "images"]]
	# 			metadata = os.path.join(folder_path, "metadata.json")
	# 			metadata = json.load(open(metadata))
	# 			# Added tuples of fake and corresponding real videos to the training_samples
	# 			for video in videos:
	# 				# Check if video is labeled as fake
	# 				if metadata[video]['split'] == 'train':
	# 					fake_video_path = os.path.join(folder_path, video)
	# 					real_video_path = os.path.join(real_folder, metadata[video]['original'])
	# 					training_samples.append((fake_video_path, real_video_path))
	# 	elif dataset == 'kaggle':
	# 		# List of sorted folders in the kaggle directory
	# 		kaggle_folders = [os.path.join(dataset_path, x) for x in os.listdir(dataset_path)]
	# 		kaggle_folders = sorted(kaggle_folders, key = lambda d: int(d.split('_')[-1]))
	# 		# Collect training samples
	# 		for folder_path in kaggle_folders:
	# 			videos = os.listdir(folder_path)
	# 			videos = [x for x in videos if x not in ["metadata.json", "bounding_boxes", "bad_samples", "multiple_faces", "images"]]
	# 			metadata = os.path.join(folder_path, "metadata.json")
	# 			metadata = json.load(open(metadata))
	# 			# Added tuples of fake and corresponding real videos to the training_samples
	# 			for video in videos:
	# 				# Check if video is labeled as fake
	# 				if metadata[video]['label'] == 'FAKE' and metadata[video]['split'] == 'train':
	# 					fake_video_path = os.path.join(folder_path, video)
	# 					real_video_path = os.path.join(folder_path, metadata[video]['original'])
	# 					training_samples.append((fake_video_path, real_video_path))
		
	# 	# Run training loop
	# 	for epoch in range(1, epochs+1):
	# 		accuracies = []
	# 		errors = []

	# 		# Set to training mode
	# 		self.network.train()

	# 		# Unfreezing gradients
	# 		if training_level == 'lower':
	# 			self.network.unfreeze_lower_level()
	# 		if training_level == 'lower' or 'higher':
	# 			self.network.unfreeze_higher_level()
	# 		self.network.unfreeze_classifier()
			
	# 		# # Initializing optimizer with appropriate lr
	# 		# classifier_lr = lr * 0.8**(epoch-1)
	# 		# higher_level_lr = 1. * classifier_lr
	# 		# lower_level_lr = 1. * higher_level_lr
	# 		# optimizer = optim.SGD([
	# 		# 	{'params': self.network.classifier_parameters(), 'lr': classifier_lr},
	# 		# 	{'params': self.network.higher_level_parameters(), 'lr': higher_level_lr},
	# 		# 	{'params': self.network.lower_level_parameters(), 'lr': lower_level_lr}
	# 		# ], momentum = momentum)

	# 		# Shuffle training_samples and initialize progress bar
	# 		random.shuffle(training_samples)
	# 		progress_bar = tqdm(training_samples, desc = "epoch {}".format(epoch))
			
	# 		# Grab samples and run iterations
	# 		for fake_video_path, real_video_path in progress_bar:

	# 			# Collecting necessary data for fake and real samples. For some of them the face images may already be extracted to speed up preprocessing.
	# 			fake_faces_path = misc.get_images_path(fake_video_path)
	# 			if os.path.isdir(fake_faces_path):
	# 				fake_data = (fake_video_path, os.path.join(fake_faces_path, '0'), 'images')
	# 			else:
	# 				fake_bb_path = misc.get_boundingbox_path(fake_video_path)
	# 				fake_boxes = json.load(open(fake_bb_path))
	# 				fake_data = (fake_video_path, fake_boxes, 'boxes')

	# 			real_faces_path = misc.get_images_path(real_video_path)
	# 			if os.path.isdir(real_faces_path):
	# 				real_data = (real_video_path, os.path.join(real_faces_path, '0'), 'images')
	# 			else:
	# 				real_bb_path = misc.get_boundingbox_path(real_video_path)
	# 				real_boxes = json.load(open(real_bb_path))
	# 				real_data = (real_video_path, real_boxes, 'boxes')

	# 			# Only run training for fake videos, with an existing original and a single face
	# 			if os.path.exists(real_video_path):
	# 				# Get batch
	# 				try:
	# 					batch = BG.training_batch(fake_data = fake_data, real_data = real_data, epoch = epoch, total_epochs = epochs)
	# 				except:
	# 					print('Fake video: {}, Real video: {}'.format(fake_video_path, real_video_path))
	# 					raise

	# 				self.network.zero_grad()
	# 				output = self.network(batch.detach())
	# 				# Compute loss and do backpropagation
	# 				err = self.criterion(output, labels)
	# 				err.backward()
	# 				# self.plot_grad_flow()
	# 				# Optimizer step applying gradients from results
	# 				optimizer.step()

	# 				# Get loss
	# 				err = err.item()
	# 				# Calculating accuracy for mixed samples
	# 				o = output.cpu().detach().numpy()
	# 				o = 1 / (1 + np.exp(-o)) # Applying the sigmoid to the output
	# 				l = labels.cpu().detach().numpy()
	# 				acc = np.sum(np.round(o) == np.round(l)) / batch_size * 100
	# 				# Add accuracy, error to lists & increment iteration
	# 				errors.append(err)
	# 				accuracies.append(acc)
	# 				# Refresh tqdm postfix
	# 				postfix_dict = {'loss': round(np.mean(errors), 2), 'acc': round(np.mean(accuracies), 2)}
	# 				progress_bar.set_postfix(postfix_dict, refresh = False)

	# 		# Clean CUDA cache
	# 		torch.cuda.empty_cache()
	# 		# Save the model weights after each folder
	# 		self.save_model(dataset + "_" + str(epoch) + "_", training_level)

	# 		# Run validation
	# 		val_dict = self.evaluate(dataset, dataset_path, 'val', batch_size = batch_size,
	# 				val_log_info = (validation_log, epoch))
	# 		val_loss = val_dict['loss']
	# 		val_acc = val_dict['acc']

	# 		# Add to epoch log
	# 		log_string = "{},{:.2f},{:.2f},{:.2f},{:.2f},\n".format(
	# 					epoch, np.mean(errors), np.mean(accuracies), val_loss, val_acc)
	# 		misc.add_to_log(log_file = epoch_log, log_string = log_string)


	def get_data_dict(self, video_path):
		# Collecting necessary data samples. 
		# For some of them the face images may already be extracted to speed up preprocessing.
		data_dict = {}
		faces_path = misc.get_images_path(video_path)
		
		if os.path.isdir(faces_path):
			data_dict['type'] = 'images'
			data_dict['images_path'] = os.path.join(faces_path, '0')
			data_dict['length'] = max([int(os.path.splitext(x)[0]) for x in os.listdir(data_dict['images_path'])])
		else:
			data_dict['type'] = 'video'
			data_dict['video_path'] = video_path
			data_dict['boundingbox_path'] = misc.get_boundingbox_path(video_path)
			data_dict['length'] = cv2.VideoCapture(video_path).get(7)

		return data_dict


	"""
	Function for training the network
		dataset    		- choice of dataset to train on
		dataset_path	- path to dataset on local machine
	"""
	def train(self, dataset, dataset_path, epochs = 10, batch_size = 14, 
			lr = 0.001, momentum = 0.9, training_level = 'classifier', training_type = 'dual'):
		
		# Display network parameter division ratios
		model_helpers.count_parameters(self.network)

		# Assert the batch_size is even
		assert batch_size % 2 == 0, "Uneven batch_size: {}".format(batch_size)
		# Assert valid dataset choise
		assert dataset in ['faceforensics', 'kaggle'], "Invalid dataset choice: {}".format(dataset)
		# Assert valid dataset patch
		assert os.path.isdir(dataset_path), "Invalid dataset path: {}".format(dataset_path)
		# Assert valid training_level
		assert training_level in ['classifier', 'higher', 'lower'], "Invalid training level choice: {}".format(training_level)
		
		# Creating batch generator
		BG = BatchGenerator(self.model_name, self.device, batch_size)

		# Initializing optimizer with appropriate lr
		classifier_lr = lr
		higher_level_lr = 0.1 * classifier_lr
		lower_level_lr = 0.1 * higher_level_lr
		# optimizer = optim.SGD([
		# 	{'params': self.network.classifier_parameters(), 'lr': classifier_lr},
		# 	{'params': self.network.higher_level_parameters(), 'lr': higher_level_lr},
		# 	{'params': self.network.lower_level_parameters(), 'lr': lower_level_lr}
		# ], momentum = momentum)
		optimizer = RAdam([
			{'params': self.network.classifier_parameters(), 'lr': classifier_lr},
			{'params': self.network.higher_level_parameters(), 'lr': higher_level_lr},
			{'params': self.network.lower_level_parameters(), 'lr': lower_level_lr}], weight_decay = 1e-4)

		# Get label tensor
		if training_type == 'dual':
			labels = torch.tensor([0]*int(batch_size/2) + [1]*int(batch_size/2),
					device = self.device, requires_grad = False, dtype = torch.float)
		elif training_type == 'various':
			labels = torch.tensor([0, 1] * int(batch_size/2), 
					device = self.device, requires_grad = False, dtype = torch.float)
		else:
			raise Exception('Invalid training_type.')
		labels = labels.view(-1,1)
		
		# Create log file w/ train/val information per epoch
		filename = self.model_name + "_" + dataset + "_"
		filename += training_level + "_epochs"
		log_header = "epoch,train_loss,train_acc,val_loss,val_acc\n"
		epoch_log = misc.create_log(filename, header_string = log_header)
		# Create log file w/ information from validation runs
		filename = self.model_name + "_" + dataset + "_"
		filename += training_level + "_validation"
		log_header = "Epoch,File,Label,AvgOutput,Loss,Acc\n"
		validation_log = misc.create_log(filename, header_string = log_header)

		# Get list of training samples
		training_samples = misc.get_training_samples(dataset, dataset_path)
		
		# Run training loop
		for epoch in range(1, epochs+1):
			accuracies = []
			errors = []

			# Set to training mode
			self.network.train()

			# Unfreezing gradients
			if training_level == 'lower':
				self.network.unfreeze_lower_level()
			if training_level == 'lower' or 'higher':
				self.network.unfreeze_higher_level()
			self.network.unfreeze_classifier()

			# Shuffle training_samples and initialize progress bar
			random.shuffle(training_samples)
			progress_bar = tqdm(training_samples, desc = "epoch {}".format(epoch))
			
			data = []
			# Grab samples and run iterations
			for fake_video_path, real_video_path in progress_bar:

				if training_type == 'dual':
					fake_dict = self.get_data_dict(fake_video_path)
					# Calculating step of frames to skip in video (subtract total_epochs to ensure there are enough frames)
					frame_step = int((fake_dict['length'] - epochs) / int(batch_size / 2))
					end = epoch + int(batch_size/2) * (frame_step - 1) + 1
					# List of frames [epoch:n:end] to be grabbed from the video
					frame_numbers = list(range(epoch, end, frame_step))
					fake_dict['frame_numbers'] = frame_numbers
					data.append(fake_dict)

					real_dict = self.get_data_dict(real_video_path)
					frame_step = int((real_dict['length'] - epochs) / int(batch_size / 2))
					end = epoch + int(batch_size/2) * (frame_step - 1) + 1
					frame_numbers = list(range(epoch, end, frame_step))
					real_dict['frame_numbers'] = frame_numbers
					data.append(real_dict)

				elif training_type == 'various' and len(data) < batch_size:
					fake_dict = self.get_data_dict(fake_video_path)
					fake_dict['frame_nr'] = (8 * epoch) % fake_dict['length'] + int(8 * epoch / fake_dict['length'])
					data.append(fake_dict)
					real_dict = self.get_data_dict(real_video_path)
					real_dict['frame_nr'] = (8 * epoch) % real_dict['length'] + int(8 * epoch / real_dict['length'])
					data.append(real_dict)

				# Only run training for fake videos, with an existing original and a single face
				if training_type == 'dual' or (training_type == 'various' and len(data) == batch_size):
					# Get batch
					if training_type == 'dual':
						batch = BG.training_batch_video_pair(data = data)
					elif training_type == 'various':
						batch = BG.training_batch_various_videos(data = data)
					data = []

					self.network.zero_grad()
					output = self.network(batch.detach())
					# Compute loss and do backpropagation
					err = self.criterion(output, labels)
					err.backward()
					# self.plot_grad_flow()
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

			# Clean CUDA cache
			torch.cuda.empty_cache()
			# Save the model weights after each folder
			self.save_model(dataset + "_" + str(epoch) + "_", training_level)

			# Run validation
			val_dict = self.evaluate(dataset, dataset_path, 'val', batch_size = batch_size,
					val_log_info = (validation_log, epoch))
			val_loss = val_dict['loss']
			val_acc = val_dict['acc']

			# Add to epoch log
			log_string = "{},{:.2f},{:.2f},{:.2f},{:.2f},\n".format(
						epoch, np.mean(errors), np.mean(accuracies), val_loss, val_acc)
			misc.add_to_log(log_file = epoch_log, log_string = log_string)


	"""
	Function for evaluation of the model on the chosen dataset.
		dataset 	 	- one of {kaggle, faceforensics}
		dataset_path 	- absolute path to the dataset
		mode 		 	- one of {val, test}
		batch_size	 	- number of frames to be grabbed from each video
		val_log_info 	- (filename, epoch_number) info for logging
	"""
	def evaluate(self, dataset, dataset_path, mode, batch_size = 24, val_log_info = None):
		
		self.network.eval()
		
		# Disable gradients
		for param in self.network.parameters():
			param.requires_grad = False

		# Creating batch generator
		BG = BatchGenerator(self.model_name, self.device, batch_size)
		
		evaluation_samples = []

		if dataset == 'faceforensics':
			# Folders in the faceforensics directory
			original_sequences = os.path.join(dataset_path, 'original_sequences')
			real_folder = os.path.join(original_sequences, 'c23', 'videos')
			manipulated_sequences = os.path.join(dataset_path, 'manipulated_sequences')
			fake_folders = [os.path.join(manipulated_sequences, x) for x in os.listdir(manipulated_sequences)]
			fake_folders = [os.path.join(x, 'c23', 'videos') for x in fake_folders]

			# Originals
			videos = os.listdir(real_folder)
			videos = [x for x in videos if x not in ["metadata.json", "bounding_boxes", "bad_samples", "multiple_faces", "images"]]
			metadata = os.path.join(real_folder, "metadata.json")
			metadata = json.load(open(metadata))
			for video in videos:
				# Check if video belongs to the correct dataset split
				if metadata[video]['split'] == mode:
					real_video_path = os.path.join(real_folder, video)
					evaluation_samples.append((real_video_path, 'REAL'))
			# Fake videos
			for folder_path in fake_folders:
				videos = os.listdir(folder_path)
				videos = [x for x in videos if x not in ["metadata.json", "bounding_boxes", "bad_samples", "multiple_faces", "images"]]
				metadata = os.path.join(folder_path, "metadata.json")
				metadata = json.load(open(metadata))
				for video in videos:
					# Check if video belongs to the correct dataset split
					if metadata[video]['split'] == mode:
						fake_video_path = os.path.join(folder_path, video)
						evaluation_samples.append((fake_video_path, 'FAKE'))

		elif dataset == 'kaggle':
			# Folders in the kaggle directory
			folder_paths = [os.path.join(dataset_path, x) for x in os.listdir(dataset_path)]

			# Iterate through folders and construct list of samples
			for folder_path in folder_paths:
				videos = os.listdir(folder_path)
				videos = [x for x in videos if x not in ["metadata.json", "bounding_boxes", "bad_samples", "multiple_faces", "images"]]
				metadata = os.path.join(folder_path, "metadata.json")
				metadata = json.load(open(metadata))
				for video in videos:
					# Check if video belongs to the correct dataset split
					if metadata[video]['split'] == mode:
						video_path = os.path.join(folder_path, video)
						evaluation_samples.append((video_path, metadata[video]['label']))

		# Initialize lists or err/acc & progress bar
		overall_err = []
		overall_acc = []
		random.shuffle(evaluation_samples)
		progress_bar = tqdm(evaluation_samples, desc = '{} ({} set)'.format(dataset, mode))

		# Predictions for balanced accuracy calculation
		output_true = []
		output_pred = []
		
		# Run evaluation loop
		for video_path, label in progress_bar:			
			# Retrieving dict of bounding boxes for faces
			bb_path = os.path.join(os.path.dirname(video_path), "bounding_boxes")
			bounding_boxes  = os.path.join(bb_path, os.path.splitext(os.path.basename(video_path))[0]) + ".json"
			bounding_boxes = json.load(open(bounding_boxes))
			# Check if the video contains frames with multiple faces
			multiple_faces = bounding_boxes['multiple_faces']

			# Only run training for fake videos, with an existing original and a single face
			if not multiple_faces:

				# Get batch
				batch = BG.evaluation_batch(video_path, boxes = bounding_boxes)
				# Feed batch through network
				output = self.network(batch.detach())
				# Get label tensor for this video
				if label == 'REAL':
					labels = torch.tensor([1]*batch_size, device = self.device, requires_grad = False, dtype = torch.float)
				else:
					labels = torch.tensor([0]*batch_size, device = self.device, requires_grad = False, dtype = torch.float)
				labels = labels.view(-1,1)
				# Compute loss
				err = self.criterion(output, labels)

				# Get loss
				err = err.item()
				# Calculating accuracy for mixed samples
				o = output.cpu().detach().numpy()
				o = 1 / (1 + np.exp(-o)) # Applying the sigmoid to the output
				l = labels.cpu().detach().numpy()
				acc = np.sum(np.round(o) == np.round(l)) / batch_size * 100
				avg_output = np.mean(o)
				# Add accuracy, error to lists & increment iteration
				overall_err.append(err)
				overall_acc.append(acc)

				# Refresh tqdm postfix
				postfix_dict = {'loss': round(np.mean(overall_err), 2), 'acc': round(np.mean(overall_acc), 2)}
				progress_bar.set_postfix(postfix_dict, refresh = False)

				output_true.append(1 if label == 'REAL' else 0)
				output_pred.append(1 if avg_output > 0.5 else 0)

				# Log results if val_log_info exists
				if val_log_info:
					validation_log_filename, epoch = val_log_info
					log_string = "{},{},{},{:.2f},{:.2f},{:.2f},\n".format(epoch,video_path, label, avg_output, err, acc)
					misc.add_to_log(log_file = validation_log_filename, log_string = log_string)

		print('Balanced accuracy score: {:.2f}%'.format(balanced_accuracy_score(output_true, output_pred) * 100))
		torch.cuda.empty_cache()
		return {'loss': np.mean(overall_err), 'acc': np.mean(overall_acc)}

