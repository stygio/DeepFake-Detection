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

	def __init__(self, model_name, model_weights_path = None, pretrained = True):
		
		# Model name
		self.model_name = model_name
		# Choose torch device
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# Setup chosen CNN model
		self.network = model_helpers.get_model(model_name, model_weights_path, pretrained).to(self.device)
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
	def train(self, dataset, dataset_path, epochs = 10, batch_size = 12, 
			lr = 0.001, momentum = 0.9, training_level = 'classifier', training_type = 'various'):
		
		# Display network parameter division ratios
		model_helpers.count_parameters(self.network)

		# Assert the batch_size is even
		assert batch_size % 2 == 0, "Uneven batch_size: {}".format(batch_size)
		# Assert valid dataset choise
		assert dataset in ['faceforensics', 'kaggle'], "Invalid dataset choice: {}".format(dataset)
		# Assert valid dataset patch
		assert os.path.isdir(dataset_path), "Invalid dataset path: {}".format(dataset_path)
		# Assert valid training_level
		assert training_level in ['classifier', 'higher', 'lower', 'full'], "Invalid training level choice: {}".format(training_level)
		
		# Creating batch generator
		BG = BatchGenerator(self.model_name, self.device, batch_size)

		# Initializing optimizer with appropriate lr
		classifier_lr = lr
		higher_level_lr = 1. * classifier_lr
		lower_level_lr = 1. * higher_level_lr
		if training_level == 'full':
			optimizer = RAdam(self.network.parameters(), lr = lr, weight_decay = 0)
		else:
			optimizer = RAdam([
				{'params': self.network.classifier_parameters(), 'lr': classifier_lr},
				{'params': self.network.higher_level_parameters(), 'lr': higher_level_lr},
				{'params': self.network.lower_level_parameters(), 'lr': lower_level_lr}], weight_decay = 0)
		# # optimizer = optim.SGD([
		# # 	{'params': self.network.classifier_parameters(), 'lr': classifier_lr},
		# # 	{'params': self.network.higher_level_parameters(), 'lr': higher_level_lr},
		# # 	{'params': self.network.lower_level_parameters(), 'lr': lower_level_lr}
		# # ], momentum = momentum)
		scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.5, patience = 0)

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
		log_header = "epoch,train_loss,train_acc,val_loss,val_acc,val_bal_acc\n"
		epoch_log = misc.create_log(filename, header_string = log_header)
		# Create log file w/ information from validation runs
		filename = self.model_name + "_" + dataset + "_"
		filename += training_level + "_validation"
		log_header = "Epoch,Folder,Video,Label,AvgOutput,Loss,Acc\n"
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
			if training_level == 'full':
				for param in self.network.parameters():
					param.requires_grad = True
			if training_level == 'lower':
				self.network.unfreeze_lower_level()
			if training_level == 'lower' or 'higher':
				self.network.unfreeze_higher_level()
			if training_level == 'lower' or 'higher' or 'classifier':
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
					end = epoch + int(batch_size/2 - 1) * frame_step + 1
					# List of frames [epoch:n:end] to be grabbed from the video
					frame_numbers = list(range(epoch, end, frame_step))
					fake_dict['frame_numbers'] = frame_numbers
					data.append(fake_dict)

					real_dict = self.get_data_dict(real_video_path)
					frame_step = int((real_dict['length'] - epochs) / int(batch_size / 2))
					end = epoch + int(batch_size/2 - 1) * frame_step + 1
					frame_numbers = list(range(epoch, end, frame_step))
					real_dict['frame_numbers'] = frame_numbers
					data.append(real_dict)

				elif training_type == 'various' and len(data) < batch_size:
					fake_dict = self.get_data_dict(fake_video_path)
					fake_dict['frame_nr'] = (8 * epoch) % (fake_dict['length'] - epochs) + int(8 * epoch / fake_dict['length'])
					data.append(fake_dict)
					real_dict = self.get_data_dict(real_video_path)
					real_dict['frame_nr'] = (8 * epoch) % (real_dict['length'] - epochs) + int(8 * epoch / real_dict['length'])
					data.append(real_dict)

				# Only run training for fake videos, with an existing original and a single face
				if training_type == 'dual' or (training_type == 'various' and len(data) == batch_size):
					# Get batch
					if training_type == 'dual':
						batch = BG.multiple_frames_per_video(data = data)
					elif training_type == 'various':
						batch = BG.single_frame_per_video(data = data)
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
			val_dict = self.evaluate(dataset, dataset_path, 'val', batch_size = 7,
					val_log_info = (validation_log, epoch))
			val_loss = val_dict['loss']
			val_acc = val_dict['acc']
			val_balanced_acc = val_dict['balanced_acc']

			# Scheduler step
			scheduler.step(val_loss)

			# Add to epoch log
			log_string = "{},{:.2f},{:.2f},{:.2f},{:.2f},\n".format(
						epoch, np.mean(errors), np.mean(accuracies), val_loss, val_acc, val_balanced_acc)
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
		
		# Initialize lists or err/acc & progress bar
		overall_err = []
		overall_acc = []
		evaluation_samples = misc.get_evaluation_samples(dataset, dataset_path, mode)		
		random.shuffle(evaluation_samples)
		progress_bar = tqdm(evaluation_samples, desc = '{} ({} set)'.format(dataset, mode))

		# Predictions for balanced accuracy calculation
		output_true, output_pred = [], []
		
		# Run evaluation loop
		for video_path, label in progress_bar:

			data_dict = self.get_data_dict(video_path)
			# Calculating step of frames to skip in video (subtract total_epochs to ensure there are enough frames)
			frame_step = int((data_dict['length'] - 10) / int(batch_size))
			end = 10 + int(batch_size - 1) * frame_step + 1
			# List of frames [epoch:n:end] to be grabbed from the video
			frame_numbers = list(range(10, end, frame_step))
			data_dict['frame_numbers'] = frame_numbers

			# Get batch
			batch = BG.multiple_frames_per_video([data_dict])
			# Get label tensor for this video
			label = 0 if label == 'FAKE' else 1
			labels = torch.tensor([label] * batch_size, device = self.device, requires_grad = False, dtype = torch.float)
			labels = labels.view(-1,1)
			# Feed batch through network
			output = self.network(batch.detach())
			
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

			# For balanced accuracy calculation
			output_true.append(label)
			output_pred.append(1 if avg_output > 0.5 else 0)

			# Log results if val_log_info exists
			if val_log_info:
				validation_log_filename, epoch = val_log_info
				video = os.path.basename(video_path)
				folder = video_path.split('\\')[-4]
				log_string = "{},{},{},{},{:.2f},{:.2f},{:.2f},\n".format(epoch, folder, video, label, avg_output, err, acc)
				misc.add_to_log(log_file = validation_log_filename, log_string = log_string)

		torch.cuda.empty_cache()
		bal_acc = balanced_accuracy_score(output_true, output_pred) * 100
		print('Balanced accuracy score: {:.2f}%'.format(bal_acc))
		return {'loss': np.mean(overall_err), 'acc': np.mean(overall_acc), 'balanced_acc': bal_acc}

