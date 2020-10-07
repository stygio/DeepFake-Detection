import cv2
import torch
import time
import os
import numpy as np
import json

from models import transform
from tools import preprocessing, opencv_helpers
from tools.custom_errors import NoFaceDetected, MultipleFacesDetected, CorruptVideoError
import tools.miscellaneous as misc

class BatchGenerator:
	
	def __init__(self, model_type, device, batch_size):
		self.model_transform = transform.model_transforms[model_type]
		self.device = device
		self.batch_size = batch_size
		# Initializing mobilenet for face recognition
		preprocessing.initialize_mobilenet(0.4)

	# Data transform to be used in training
	def training_transform(self, data, random_erasing = False):
		if random_erasing:
			data = transform.random_erasing(self.model_transform(transform.data_augmentation(transform.to_PIL(data))))
		else:
			data = self.model_transform(transform.data_augmentation(transform.to_PIL(data)))
		return data

	# Data transform to be used in evaluation/detection
	def evaluation_transform(self, data):
		data = self.model_transform(transform.to_PIL(data))
		return data


	def show_tensor(self, tensor):
		# Convert to numpy array
		image = np.moveaxis(tensor.cpu().detach().numpy(), 0, -1)
		# Denormalize from normalized tensor
		image[:,:,0] = image[:,:,0] * 0.5 + 0.5
		image[:,:,1] = image[:,:,1] * 0.5 + 0.5
		image[:,:,2] = image[:,:,2] * 0.5 + 0.5
		cv2.imshow("tensor", image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()


	"""
	Function to retrieve a generator of training batches in tensor form (Kaggle dataset)
	The batches contain sequences of consecutive frames from a two videos (half of the batch from each one)
		fake_video_path	- path to the altered video
		real_video_path	- path to the original video the fake is based on
	"""
	def multiple_frames_per_video(self, data, mode):
		faces = []

		for data_dict in data:
			if data_dict['type'] == 'images':
				for frame_nr in data_dict['frame_numbers']:
					face = cv2.imread(os.path.join(data_dict['images_path'], '{}.png'.format(frame_nr)))
					# preprocessing.show_test_img(face)
					face = self.training_transform(face, random_erasing = True) if mode == 'train' else self.evaluation_transform(face)
					# self.show_tensor(face)
					faces.append(face)
			else:
				try:
					video_handle = cv2.VideoCapture(data_dict['video_path'])
					# Check that the video was opened successfully
					if video_handle.isOpened() == False:
						raise CorruptVideoError("cv2.VideoCapture() failed to open {}".format(data_dict['video_path']))
					# Frames from the video
					frames = opencv_helpers.specific_frames(video_handle, data_dict['frame_numbers'])
					video_handle.release()
					# Retrieve boundingbox information and crop image
					boxes = json.load(open(data_dict['boundingbox_path']))
					for i, frame_nr in enumerate(data_dict['frame_numbers']):
						top 	= boxes[str(frame_nr)]['0']['top']
						bottom 	= boxes[str(frame_nr)]['0']['bottom']
						left 	= boxes[str(frame_nr)]['0']['left']
						right 	= boxes[str(frame_nr)]['0']['right']
						face = preprocessing.crop_image(frames[i], (top, bottom, left, right))
						# preprocessing.show_test_img(face)
						face = self.training_transform(face, random_erasing = True) if mode == 'train' else self.evaluation_transform(face)
						# self.show_tensor(face)
						faces.append(face)
				except CorruptVideoError:
					video_handle.release()
					# Move the file to a folder for corrupt videos
					misc.put_file_in_folder(file_path = data_dict['video_path'], folder = "bad_samples")
					raise

		batch = torch.stack(faces).to(self.device)
		return batch


	"""
	Function to retrieve a generator of training batches in tensor form (Kaggle dataset)
	The batches contain sequences of consecutive frames from a two videos (half of the batch from each one)
		fake_video_path	- path to the altered video
		real_video_path	- path to the original video the fake is based on
	"""
	def single_frame_per_video(self, data):
		faces = []

		for data_dict in data:
			if data_dict['type'] == 'images':
				face = cv2.imread(os.path.join(data_dict['images_path'], '{}.png'.format(int(data_dict['frame_nr']))))
				# preprocessing.show_test_img(face)
				face = self.training_transform(face, random_erasing = True)
				# self.show_tensor(face)
				faces.append(face)
			else:
				try:
					video_handle = cv2.VideoCapture(data_dict['video_path'])
					# Check that the video was opened successfully
					if video_handle.isOpened() == False:
						raise CorruptVideoError("cv2.VideoCapture() failed to open {}".format(data_dict['video_path']))
					# Frame from the video
					frame = opencv_helpers.specific_frames(video_handle, [int(data_dict['frame_nr'])])[0]
					video_handle.release()
					# Retrieve boundingbox information and crop image
					boxes = json.load(open(data_dict['boundingbox_path']))
					top 	= boxes[str(int(data_dict['frame_nr']))]['0']['top']
					bottom 	= boxes[str(int(data_dict['frame_nr']))]['0']['bottom']
					left 	= boxes[str(int(data_dict['frame_nr']))]['0']['left']
					right 	= boxes[str(int(data_dict['frame_nr']))]['0']['right']
					face = preprocessing.crop_image(frame, (top, bottom, left, right))
					# preprocessing.show_test_img(face)
					face = self.training_transform(face, random_erasing = True)
					# self.show_tensor(face)
					faces.append(face)
				except CorruptVideoError:
					video_handle.release()
					# Move the file to a folder for corrupt videos
					misc.put_file_in_folder(file_path = data_dict['video_path'], folder = "bad_samples")
					raise

		batch = torch.stack(faces).to(self.device)
		return batch


	"""
	Function to retrieve a generator of batches in tensor form (Kaggle dataset)
	The batches contain sequences of consecutive frames from a two videos (half of the batch from each one)
		video_path - path to the video
		boxes	   - dictionary of bounding boxes in the video	
	"""
	def full_video_batch(self, video_path, boxes):
		# Open the video
		video_handle = cv2.VideoCapture(video_path)
		try:
			# Check that video was opened correctly
			if video_handle.isOpened() == False:
				raise CorruptVideoError("cv2.VideoCapture() failed to open {}".format(video_path))

			# Generator for consecutive sequences of <batch_size> frames
			frame_generator = opencv_helpers.yield_video_frames(video_handle, int(self.batch_size))
			# Iterate through the video yielding batches of <batch_size> frames
			error = False
			start_frame = 0
			while not error:
				try:
					batch = next(frame_generator)
					start_frame += int(self.batch_size)

					# Retrieve boundingbox information and crop images
					faces = []
					for i in range(int(self.batch_size)):
						top 	= boxes[str(start_frame + i)]['0']['top']
						bottom 	= boxes[str(start_frame + i)]['0']['bottom']
						left 	= boxes[str(start_frame + i)]['0']['left']
						right 	= boxes[str(start_frame + i)]['0']['right']
						face = preprocessing.crop_image(batch[i], (top, bottom, left, right))
						faces.append(self.evaluation_transform(face))

					batch = torch.stack(faces).to(self.device)
					yield batch

				# Video is done (frame_generator finished)
				except StopIteration:
					del frame_generator
					video_handle.release()
					error = True
					break
		
		except CorruptVideoError as Error:
			# Release the video file
			video_handle.release()
			# Move the file to a folder for corrupt videos
			misc.put_file_in_folder(file_path = video_path, folder = "bad_samples")