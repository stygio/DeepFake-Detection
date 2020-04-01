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

import tools.miscellaneous as misc
from tools import preprocessing
from models.model_helpers import get_model
from tools import opencv_helpers


kaggle_test_folders = [	"dfdc_train_part_45",
						"dfdc_train_part_46",
						"dfdc_train_part_47",
						"dfdc_train_part_48",
						"dfdc_train_part_49"]


# """
# Function to retrieve a homogenous batch in tensor form (FaceForensics dataset)
# 	video_path_generator - generator object which returns paths to video samples
# 	model_type           - model type name
# 	device               - PyTorch device
# 	batch_size           - size of returned batch (# of consecutive frames from the video)
# """
# def get_homogenous_batch(video_path_generator, model_type, device, batch_size):
# 	# While there is no batch, try to create one
# 	batch, video_path = None, None
# 	while not torch.is_tensor(batch):
# 		try:
# 			video_path = next(video_path_generator)
# 			batch = preprocessing.create_homogenous_batch(video_path = video_path, 
# 				model_type = model_type, device = device, batch_size = batch_size)
# 		except AttributeError as Error:
# 			# No faces error
# 			print("DEBUG: {}".format(Error))
# 		except ValueError as Error:
# 			# Multiple faces error
# 			print("DEBUG: {}".format(Error))
# 			# Move the file to a special folder for videos with multiple faces
# 			misc.put_file_in_folder(file_path = video_path, folder = "multiple_faces")
# 		except (AssertionError, IndexError) as Error:
# 			# Video access or video length error
# 			print("DEBUG: {}".format(Error))
# 			# Move the file to a special folder for corrupt/short videos
# 			misc.put_file_in_folder(file_path = video_path, folder = "bad_samples")

# 	return batch, video_path


# """
# Function to retrieve a disparate batch in tensor form (FaceForensics dataset)
# 	real_video_generator - generator object which returns paths to real video samples
# 	fake_video_generator - generator object which returns paths to fake video samples
# 	model_type           - model type name
# 	device               - PyTorch device
# 	batch_size           - size of returned batch (# of consecutive frames from the video)
# """
# def get_disparate_batch(real_video_generator, fake_video_generator, model_type, device, batch_size):
# 	batch, labels = preprocessing.create_disparate_batch(
# 		real_video_generator = real_video_generator, fake_video_generator = fake_video_generator, model_type = model_type, device = device, batch_size = batch_size)

# 	return batch, labels


"""
Function to retrieve a generator of batches in tensor form (Kaggle dataset)
	video_path           - path to the video from which frames will be captured
	model_type           - model type name
	device               - PyTorch device
	batch_size           - size of returned batch (# of consecutive frames from the video)
"""
def get_kaggle_batch_single(video_path, model_type, device, batch_size):
	video_handle = cv2.VideoCapture(video_path)
	# Try..Except to handle the video_handle failure case
	try:
		assert video_handle.isOpened() == True, "VideoCapture() failed to open {}".format(video_path)
		video_length = video_handle.get(7)
		video_handle.release()
		cv2.destroyAllWindows()

		# Iterate through the video yielding batches of <batch_size> frames
		start_frame = 0
		error = False
		while start_frame + batch_size <= video_length and not error:
			batch = None
			while not torch.is_tensor(batch):
				try:
					batch = preprocessing.create_homogenous_batch(video_path = video_path, 
						model_type = model_type, device = device, batch_size = batch_size, start_frame = start_frame)
					start_frame += batch_size
					yield batch
				
				except IndexError as Error:
					# Requested segment is invalid (goes out of bounds of video length)
					error = True
					break
				except AttributeError as Error:
					# No faces error
					print("DEBUG: {}".format(Error))
					start_frame += batch_size
				except ValueError as Error:
					# Multiple faces error
					print("DEBUG: {}".format(Error))
					# Move the file to a special folder for videos with multiple faces
					misc.put_file_in_folder(file_path = video_path, folder = "multiple_faces")
					error = True
					break
				except AssertionError as Error:
					# Corrupt file error
					print("DEBUG: {}".format(Error))
					# Move the file to a special folder for corrupt videos
					misc.put_file_in_folder(file_path = video_path, folder = "bad_samples")
					error = True
					break
	
	except AssertionError:
		# Release the video file
		video_handle.release()
		cv2.destroyAllWindows()
		# Move the file to a special folder for short/corrupt videos
		misc.put_file_in_folder(file_path = video_path, folder = "bad_samples")
		yield None


"""
Function to retrieve a generator of batches in tensor form (Kaggle dataset)
The batches are constructed from two seperate videos
	video_path_1         - path to the video from which frames will be captured
	video_path_2         - path to the video from which frames will be captured
	model_type           - model type name
	device               - PyTorch device
	batch_size           - size of returned batch (# of consecutive frames from the video)
"""
def get_kaggle_batch_dual(video_path_1, video_path_2, model_type, device, batch_size):
	# Open the two videos
	video_handle_1 = cv2.VideoCapture(video_path_1)
	video_handle_2 = cv2.VideoCapture(video_path_2)
	# Try..Except to handle the video_handle failure case
	try:
		assert video_handle_1.isOpened() == True, "VideoCapture() failed to open {}".format(video_path_1)
		assert video_handle_2.isOpened() == True, "VideoCapture() failed to open {}".format(video_path_2)

		# Iterate through the video yielding batches of <batch_size> frames
		frame_generator_1 = opencv_helpers.yield_video_frames(video_handle_1, int(batch_size/2))
		frame_generator_2 = opencv_helpers.yield_video_frames(video_handle_2, int(batch_size/2))
		error = False
		while not error:
			batch = None
			while not torch.is_tensor(batch) and not error:
				batch1, batch2 = None, None
				while not torch.is_tensor(batch1):
					try:
						batch1 = preprocessing.create_homogenous_batch(next(frame_generator_1), model_type = model_type, device = device)
					# Video is done (frame_generator_1 finished)
					except StopIteration:
						del frame_generator_1
						del frame_generator_2
						video_handle_1.release()
						video_handle_2.release()
						error = True
						break
					# No faces error
					except AttributeError as Error:
						print(">> DEBUG: {}".format(Error))
					# Multiple faces error
					except ValueError as Error:
						print(">> DEBUG: {}".format(Error))
						del frame_generator_1
						del frame_generator_2
						video_handle_1.release()
						video_handle_2.release()
						# Move the file to a special folder for videos with multiple faces
						misc.put_file_in_folder(file_path = video_path_1, folder = "multiple_faces")
						error = True
						break
				while not torch.is_tensor(batch2) and not error:
					try:
						batch2 = preprocessing.create_homogenous_batch(next(frame_generator_2), model_type = model_type, device = device)
					# Video is done (frame_generator_2 finished)
					except StopIteration:
						del frame_generator_1
						del frame_generator_2
						video_handle_1.release()
						video_handle_2.release()
						error = True
						break
					# No faces error
					except AttributeError as Error:
						print(">> DEBUG: {}".format(Error))
					# Multiple faces error
					except ValueError as Error:
						print(">> DEBUG: {}".format(Error))
						del frame_generator_1
						del frame_generator_2
						video_handle_1.release()
						video_handle_2.release()
						# Move the file to a special folder for videos with multiple faces
						misc.put_file_in_folder(file_path = video_path_2, folder = "multiple_faces")
						error = True
						break
				if torch.is_tensor(batch1) and torch.is_tensor(batch2):
					batch = torch.cat((batch1, batch2), 0)
					yield batch
	
	except AssertionError:
		# Move the corrupt file to a special folder for short/corrupt videos
		if video_handle_1.isOpened() == False:
			video_handle_1.release()
			misc.put_file_in_folder(file_path = video_path_1, folder = "bad_samples")
		elif video_handle_2.isOpened() == False:
			video_handle_1.release()
			video_handle_2.release()
			misc.put_file_in_folder(file_path = video_path_2, folder = "bad_samples")
		yield None


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


"""
Function for training chosen model on kaggle data.
	real_video_dirs - list of directories with real training samples (videos) 
	fake_video_dirs - list of directories with fake training samples (videos)
	epochs          - # of epochs to train the model
	batch_size      - size of training batches (training will use both a real and fake batch of this size)
	model_name      - chosen model to be trained
	only_fc_layer   - trains all weights or only the final fully connected layer
"""
def train_kaggle(kaggle_dataset_path, model_name = "xception", model_weights_path = None,
			epochs = 1, iterations = 50, batch_size = 32, batch_type = "dual", 
			lr = 0.001, momentum = 0.9, only_fc_layer = True):
	
	# Generator for random folder paths in real/fake video directories
	kaggle_folders = [os.path.join(kaggle_dataset_path, x) for x in os.listdir(kaggle_dataset_path) if x not in kaggle_test_folders]
	folder_paths = misc.get_random_folder_from_list(kaggle_folders)
	del kaggle_folders

	# Initializing mobilenet for face recognition
	preprocessing.initialize_mobilenet()

	# Choose torch device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Setup chosen CNN model for training
	network = get_model(model_name, model_weights_path).to(device)
	if only_fc_layer:
		for param in network.parameters():
			param.requires_grad = False
		for param in network.fc.parameters():
			param.requires_grad = True

	# Loss function and optimizer
	criterion = nn.BCEWithLogitsLoss()
	# optimizer = optim.SGD(network.fc.parameters(), lr = lr, momentum = momentum)	
	optimizer = optim.Adam(network.fc.parameters())	

	# Create log file
	filename = model_name + "_kaggle"
	filename += "_fc" if only_fc_layer else "_full"
	log_header = "Folder,FakeVideo,RealVideo,Epoch,Iteration,Loss,Accuracy,\n"
	log_file = misc.create_log(filename, header_string = log_header)
	
	# Run training loop, a folder of videos is an epoch
	for epoch in range(epochs):
		# Get the next folder of videos
		folder_path = next(folder_paths, None)
		if folder_path == None:
			print("DEBUG: No more folders.")
			break
		# Possibly replace with a generator?
		videos = os.listdir(folder_path)
		random.shuffle(videos)
		videos = (x for x in videos if x not in ["metadata.json", "multiple_faces", "bad_samples"])
		metadata = os.path.join(folder_path, "metadata.json")
		metadata = json.load(open(metadata))

		# Each processed video/set of videos is an iteration
		iteration = 0
		while iteration <= iterations-1:
			video = next(videos)
			accuracies = []
			errors = []
			# Get label from metadata.json
			label = metadata[video]['label']

			start_time = time.time()

			if batch_type == "single":
				# Get label tensor for this video
				if label == 'REAL':
					labels = torch.tensor([1]*batch_size, device = device, requires_grad = False, dtype = torch.float)
				else:
					labels = torch.tensor([0]*batch_size, device = device, requires_grad = False, dtype = torch.float)
				labels = labels.view(-1,1)
				# Get batch generator
				video_path = os.path.join(folder_path, video)
				batch_generator = get_kaggle_batch_single(video_path = video_path, model_type = model_name, device = device, batch_size = batch_size)
			elif batch_type == "dual":
				# Assert the batch_size is even
				assert batch_size % 2 == 0, "Uneven batch_size equal to {}".format(batch_size)
				# <batch_type> dual only constructs batches from videos with <label> FAKE
				if label == "FAKE":
					# Get label tensor
					labels = torch.tensor([0]*int(batch_size/2) + [1]*int(batch_size/2), device = device, requires_grad = False, dtype = torch.float)
					labels = labels.view(-1,1)
					video_path_1 = os.path.join(folder_path, video)
					video_path_2 = os.path.join(folder_path, metadata[video]['original'])
					batch_generator = get_kaggle_batch_dual(video_path_1, video_path_2, model_type = model_name, device = device, batch_size = batch_size)
			else:
				raise Exception("Invalid batch_type: {}".format(batch_type))

			init_time = time.time() - start_time

			# In case of <batch_type> dual, only run training if the handled video is fake and it's source video exists
			# Otherwise, a dual batch can't be constructed from the fake and its original
			if batch_type != "dual" or (label == "FAKE" and os.path.exists(video_path_2)): 
				iteration += 1
				batches = 0
				if batch_type == "single":
					print(">> Epoch [{}/{}] Iteration [{}] Processing: {}".format(
									epoch, epochs-1, iteration, os.path.join(video_path)))
				elif batch_type == "dual":
					print(">> Epoch [{}/{}] Iteration [{}] Processing: {} and {}".format(
									epoch, epochs-1, iteration, 
									video_path_1, video_path_2))

				while True:
					torch.cuda.empty_cache()
					# Try to get next batch
					try:
						# start_time_1 = time.time()

						batch = next(batch_generator)
						if not torch.is_tensor(batch):
							# File cannot be opened
							break

						# batch_creation_time = time.time() - start_time_1
						# start_time_2 = time.time()

						network.zero_grad()
						output = network(batch.detach())
						if model_name == 'inception_v3':
							output = output[0]
						# Delete the batch to conserve memory
						del batch
						torch.cuda.empty_cache()
						# Compute loss and do backpropagation
						err = criterion(output, labels)
						err.backward()

						# Calculating accuracy for mixed samples
						o = output.cpu().detach().numpy()
						o = 1 / (1 + np.exp(-o)) # Applying the sigmoid to the output
						l = labels.cpu().detach().numpy()
						acc = np.sum(np.round(o) == np.round(l)) / batch_size * 100
						# Optimizer step applying gradients from results
						optimizer.step()
						# Add accuracy, error to lists & increment iteration
						accuracies.append(acc)
						errors.append(err.item())

						batches += 1

						# network_time = time.time() - start_time_2
						# total_batch_time = time.time() - start_time_1
						# output_string = ">> Loss: {:3.2f} | Accuracy: {:05.2f}%".format(
						# 	err.item(), acc)
						# output_string = ">> DEBUG: <elapsed time> Batch: {:.3f} | Network: {:.3f} | Total: {:.3f}".format(
						# 	batch_creation_time, network_time, total_batch_time)
						# print(output_string)

					# If there are no more sequences to retrieve or the file cannot be opened
					except StopIteration:
						if batches > 0:
							err = sum(errors)/batches
							acc = sum(accuracies)/batches

							# Write iteration results to console
							output_string = ">> Epoch [{}/{}] Iteration [{}] Loss: {:3.2f} | Accuracy: {:05.2f}%".format(
								epoch, epochs-1, iteration, err, acc)
							print(output_string)

							# Write iteration results to log file
							log_string = "{},{},{},{},{},{:.2f},{:.2f},\n".format(
								folder_path.split("\\")[-2], 
								os.path.basename(video_path_1), 
								os.path.basename(video_path_2), 
								epoch, iteration, err, acc)
							misc.add_to_log(log_file = log_file, log_string = log_string)
						break

		# Save the network after every epoch
		model_filename_string = model_name + "_kaggle_" + batch_type
		model_filename_string += "_fc" if only_fc_layer else "_full"
		misc.save_network(network_state_dict = network.state_dict(), base_filename = model_filename_string)