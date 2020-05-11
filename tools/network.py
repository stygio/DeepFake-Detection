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

import tools.miscellaneous as misc
from models.model_helpers import get_model
from tools.batch import BatchGenerator


class Network:

	def __init__(self, model_name, model_weights_path = None):
		
		# Model name
		self.model_name = model_name
		# Choose torch device
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# Setup chosen CNN model
		self.network = get_model(model_name, model_weights_path).to(self.device)
		# Loss function and optimizer
		self.criterion = nn.BCEWithLogitsLoss()

		# print(self.network)


	"""
	Save the model weights
		dataset_name  	 - dataset used for training (kaggle, face_forsensics)
		finetuning_level - training type/level of finetuning {classifier, higher, lower}
	"""
	def save_model(self, dataset_name, finetuning_level):
		
		model_dir = "models/saved_models/"
		# Construct filename
		filename = self.model_name + "_" + dataset_name
		filename += finetuning_level
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


	"""
	Function for training chosen model on faceforensics data.
		ff_dataset_path	- path to FaceForensics++ dataset on local machine
	"""
	def train_faceforensics(self, ff_dataset_path, epochs = 1, batch_size = 24, 
			lr = 0.0001, momentum = 0.9, finetuning_level = 'classifier'):
		
		self.network.train()

		# Assert the batch_size is even
		assert batch_size % 2 == 0, "Uneven batch_size equal to {}".format(batch_size)
		
		# Creating batch generator
		BG = BatchGenerator(self.model_name, self.device, batch_size)
		
		# Unfreezing gradients
		if finetuning_level == 'lower':
			self.network.unfreeze_lower_level()
		if finetuning_level == 'lower' or 'higher':
			self.network.unfreeze_higher_level()
		self.network.unfreeze_classifier()

		# Get label tensor
		labels = torch.tensor([0]*int(batch_size/2) + [1]*int(batch_size/2), device = self.device, requires_grad = False, dtype = torch.float)
		labels = labels.view(-1,1)
		
		# Create log file w/ information from every iteration
		filename = self.model_name + "_ff_"
		filename += finetuning_level + "_iterations"
		log_header = "Epoch,Folder,FakeVideo,RealVideo,Loss,Accuracy,\n"
		iteration_log = misc.create_log(filename, header_string = log_header)
		# Create log file w/ train/val information for epoch
		filename = self.model_name + "_ff_"
		filename += finetuning_level + "_epochs"
		log_header = "epoch,train_loss,train_acc,val_loss,val_acc\n"
		epoch_log = misc.create_log(filename, header_string = log_header)

		# List of sorted folders in the kaggle directory
		original_sequences = os.path.join(ff_dataset_path, 'original_sequences')
		real_folder = os.path.join(original_sequences, 'c23', 'videos')
		manipulated_sequences = os.path.join(ff_dataset_path, 'manipulated_sequences')
		fake_folders = [os.path.join(manipulated_sequences, x) for x in os.listdir(manipulated_sequences)]
		fake_folders = [os.path.join(x, 'c23', 'videos') for x in fake_folders]

		training_samples = []
		for folder_path in fake_folders:
			videos = os.listdir(folder_path)
			videos = [x for x in videos if x not in ["metadata.json", "bounding_boxes", "bad_samples", "multiple_faces", "images"]]
			metadata = os.path.join(folder_path, "metadata.json")
			metadata = json.load(open(metadata))
			# Added tuples of fake and corresponding real videos to the training_samples
			for video in videos:
				# Check if video is labeled as fake
				if metadata[video]['split'] == 'train':
					fake_video_path = os.path.join(folder_path, video)
					real_video_path = os.path.join(real_folder, metadata[video]['original'])
					training_samples.append((fake_video_path, real_video_path))
		
		# Run training loop
		for epoch in range(1, epochs+1):
			accuracies = []
			errors = []
			
			# Initializing optimizer with appropriate lr
			classifier_lr = lr * 0.8**(epoch-1)
			higher_level_lr = 0.1 * classifier_lr
			lower_level_lr = 0.01 * classifier_lr
			optimizer = optim.SGD([
				{'params': self.network.classifier_parameters(), 'lr': classifier_lr},
				{'params': self.network.higher_level_parameters(), 'lr': higher_level_lr},
				{'params': self.network.lower_level_parameters(), 'lr': lower_level_lr}
			], momentum = momentum)
			# optimizer = optim.SGD(self.network.parameters(), lr = classifier_lr, momentum = momentum)

			# Shuffle training_samples and initialize progress bar
			random.shuffle(training_samples)
			progress_bar = tqdm(training_samples, desc = "epoch {}".format(epoch))
			
			# Grab samples and run iterations
			for fake_video_path, real_video_path in progress_bar:
				# Keep track of whether one or both of the videos contains multiple faces
				multiple_faces = False

				# Collecting necessary data for fake and real samples. For some of them the face images may already be extracted to speed up preprocessing.
				fake_faces_path = misc.get_images_path(fake_video_path)
				if os.path.isdir(fake_faces_path):
					fake_data = (fake_video_path, os.path.join(fake_faces_path, '0'), 'images')
					multiple_faces = multiple_faces or os.path.isfile(os.path.join(fake_faces_path, 'multiple_faces'))
				else:
					fake_bb_path = misc.get_boundingbox_path(fake_video_path)
					fake_boxes = json.load(open(fake_bb_path))
					fake_data = (fake_video_path, fake_boxes, 'boxes')
					multiple_faces = multiple_faces or fake_boxes['multiple_faces']

				real_faces_path = misc.get_images_path(real_video_path)
				if os.path.isdir(real_faces_path):
					real_data = (real_video_path, os.path.join(real_faces_path, '0'), 'images')
					multiple_faces = multiple_faces or os.path.isfile(os.path.join(real_faces_path, 'multiple_faces'))
				else:
					real_bb_path = misc.get_boundingbox_path(real_video_path)
					real_boxes = json.load(open(real_bb_path))
					real_data = (real_video_path, real_boxes, 'boxes')
					multiple_faces = multiple_faces or real_boxes['multiple_faces']

				# Only run training for fake videos, with an existing original and a single face
				if os.path.exists(real_video_path) and not multiple_faces:
					# Get batch
					try:
						batch = BG.training_batch(fake_data = fake_data, real_data = real_data, epoch = epoch)
					except:
						print('Fake video: {}, Real video: {}'.format(fake_video_path, real_video_path))
						raise

					self.network.zero_grad()
					output = self.network(batch.detach())
					# if self.model_name == 'inception_v3':
					# 	output = output[0]
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

					# Log results
					log_string = "{},{},{},{},{:.2f},{:.2f},\n".format(
								epoch, os.path.split(os.path.dirname(real_video_path))[1], 
								os.path.basename(fake_video_path), os.path.basename(real_video_path), 
								err, acc)
					misc.add_to_log(log_file = iteration_log, log_string = log_string)

			# Save the model weights after each folder
			self.save_model("ff_" + str(epoch) + "_", finetuning_level)

			# Run validation
			val_dict = self.evaluate('faceforensics', ff_dataset_path, 'val')
			val_loss = val_dict['loss']
			val_acc = val_dict['acc']

			# Add to epoch log
			log_string = "{},{:.2f},{:.2f},{:.2f},{:.2f},\n".format(
						epoch, np.mean(errors), np.mean(accuracies), val_loss, val_acc)
			misc.add_to_log(log_file = epoch_log, log_string = log_string)


	"""
	Function for training chosen model on kaggle data.
		kaggle_dataset_path	- path to Kaggle DFDC dataset on local machine
	"""
	def train_kaggle(self, kaggle_dataset_path, epochs = 1, batch_size = 24, 
			lr = 0.0001, momentum = 0.9, only_fc_layer = False):
		
		self.network.train()

		# Assert the batch_size is even
		assert batch_size % 2 == 0, "Uneven batch_size equal to {}".format(batch_size)
		
		# Creating batch generator
		BG = BatchGenerator(self.model_name, self.device, batch_size)
		
		# Initializing optimizer
		# optimizer = optim.Adam()
		if only_fc_layer:
			self.network.unfreeze_classifier()
		optimizer = optim.SGD(self.network.parameters(), lr = lr, momentum = momentum)

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
			videos = [x for x in videos if x not in ["metadata.json", "bounding_boxes", "bad_samples", "multiple_faces", "images"]]
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
					batch = BG.training_batch(fake_video_path, real_video_path, fake_boxes = bounding_boxes, real_boxes = bounding_boxes, epoch = epoch)
					# print(">> Epoch [{}/{}] Processing: {} and {}".format(
					# 			epoch, epochs, fake_video_path, real_video_path))

					self.network.zero_grad()
					output = self.network(batch.detach())
					# if self.model_name == 'inception_v3':
					# 	output = output[0]
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
					if len(errors) < 5000:
						training_loss = round(np.mean(errors), 2)
						training_acc = round(np.mean(accuracies), 2)
					else:
						training_loss = round(np.mean(errors[-5000:]), 2)
						training_acc = round(np.mean(accuracies[-5000:]), 2)
					postfix_dict = {'loss': training_loss, 'acc': training_acc}
					progress_bar.set_postfix(postfix_dict, refresh = False)

					# Log results
					log_string = "{},{},{},{},{:.2f},{:.2f},\n".format(
								epoch, os.path.split(os.path.dirname(real_video_path))[1], 
								os.path.basename(fake_video_path), os.path.basename(real_video_path), 
								err, acc)
					misc.add_to_log(log_file = log_file, log_string = log_string)

			# Save the model weights after each folder
			self.save_model("kaggle_" + str(epoch), only_fc_layer)


	"""
	Function for evaluation of the model on the chosen dataset.
		dataset 	 - one of {kaggle, faceforensics}
		dataset_path - absolute path to the dataset
		mode 		 - one of {val, test}
		batch_size	 - number of frames to be grabbed from each video
	"""
	def evaluate(self, dataset, dataset_path, mode, batch_size = 24):
		
		self.network.eval()

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
				# Add accuracy, error to lists & increment iteration
				overall_err.append(err)
				overall_acc.append(acc)

				# Refresh tqdm postfix
				postfix_dict = {'loss': round(np.mean(overall_err), 2), 'acc': round(np.mean(overall_acc), 2)}
				progress_bar.set_postfix(postfix_dict, refresh = False)

		return {'loss': np.mean(overall_err), 'acc': np.mean(overall_acc)}

