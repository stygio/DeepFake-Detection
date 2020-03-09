"""
Library of functions for image preprocessing
"""

import numpy as np
import random
import cv2
from PIL import Image
import face_recognition
import torch

from tools import opencv_helpers
from tools.miscellaneous import put_file_in_folder
from models import transform

crop_factor = 1.3


def show_test_img(test_img):
	cv2.imshow("test", test_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


# Image preprocessing: face detection, cropping, resizing
def get_faces(img, isPath = False):
	# Load image and resize if it's too big (otherwise we run into an out-of-memory error with CUDA)
	if isPath:
		img = cv2.imread(img)
	if np.shape(img)[0] > 720:
		scale_factor = 720/np.shape(img)[0] # percent of original size
		width = int(img.shape[1] * scale_factor)
		height = int(img.shape[0] * scale_factor)
		dim = (width, height)
		# resize image
		img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	# print("DEBUG: Retrieved image shape: {}".format(np.shape(rgb_img)))
	
	# Acquire face_locations, which is a list of tuples with locations
	# of bounding boxes specified as (top, right, bottom, left)
	face_locations = face_recognition.face_locations(rgb_img, model="cnn")
	faces = []
	face_positions = []
	for face in face_locations:
		# Retrieve original bounding box
		(top, right, bottom, left) = face
		crop_height = bottom - top
		crop_width = right - left
		# Get the face's position in the image
		face_Y = top + (crop_height / 2)
		face_X = left + (crop_width / 2)
		# Modify bounds by crop_factor
		top = top - int((crop_factor-1) * crop_height / 2)
		bottom = bottom + int((crop_factor-1) * crop_height/ 2)
		left = left - int((crop_factor-1) * crop_width / 2)
		right = right + int((crop_factor-1) * crop_width / 2)
		# Calculate square crop dimensions
		crop_height = bottom - top
		crop_width = right - left
		crop_diff = abs(crop_height - crop_width)
		# Height of bounding box is larger than its width, extend horizontally
		if crop_height > crop_width:
			left = left - int(crop_diff/2)
			right = right + int((crop_diff+1)/2)		# Compensating for cases where cropp_diff is an odd number
		# Width of bounding box is larger than its height, extend vertically
		elif crop_width > crop_height:
			top = top - int(crop_diff/2)
			bottom = bottom + int((crop_diff+1)/2)	# Compensating for cases where cropp_diff is an odd number
		
		# Crop, making sure new dimensions don't go out of bounds
		(img_height, img_width, _) = np.shape(img)
		cropped_img = img[max(top, 0):min(bottom, img_height-1), max(left, 0):min(right, img_width-1)]

		# print("DEBUG: crop_height: {}, crop_width: {}, crop_diff: {}".format(crop_height, crop_width, crop_diff))
		# print("DEBUG: top: {}, bottom: {}, left: {}, right: {}".format(top, bottom, left, right))

		# Handle cases where the new box will extend beyond image dimensions, requiring padding
		(crop_height, crop_width, _) = np.shape(cropped_img)
		if top < 0:
			padding = np.zeros((abs(top), crop_width, 3), dtype = "uint8")
			cropped_img = cv2.vconcat([padding, cropped_img])
		elif left < 0:
			padding = np.zeros((crop_height, abs(left), 3), dtype = "uint8")
			cropped_img = cv2.hconcat([padding, cropped_img])
		elif bottom > img_height-1:
			padding = np.zeros((bottom - (img_height-1), crop_width, 3), dtype = "uint8")
			cropped_img = cv2.vconcat([cropped_img, padding])
		elif right > img_width-1:
			padding = np.zeros((crop_height, right - (img_width-1), 3), dtype = "uint8")
			cropped_img = cv2.hconcat([padding, cropped_img])

		if np.shape(cropped_img[0]) != np.shape(cropped_img[1]):
			print("DEBUG: Cropped image is not a square! Shape: {}".format(np.shape(cropped_img)))
		
		# Append transformed face and its position in the image
		faces.append(cropped_img)
		face_positions.append((face_Y, face_X))

	# Throw an AssertionError if <faces> is an empty list:
	assert faces, "No faces detected."

	return faces, face_positions


# Create a batch of face images from a point in the video
def create_homogenous_batch(video_path, model_type, device, batch_size, start_frame = None):
	tensor_transform = transform.model_transforms[model_type]

	video_handle = cv2.VideoCapture(video_path)
	# Try..Except to handle the video_handle failure case
	try:
		assert video_handle.isOpened() == True, "VideoCapture() failed to open the video"
		video_length = video_handle.get(7)
		# If start_frame is not given choose random start_frame in the range of the video length in frames
		if start_frame == None:
			start_frame = random.randint(0, (video_length - 1) - batch_size)
		else:
			if start_frame + batch_size - 1 > video_length:
				raise IndexError("Requested segment of video is too long: last_frame {} > video length {}".format(
					start_frame + batch_size - 1, video_length))

		# Grab a frame sequence
		frames = opencv_helpers.loadFrameSequence(video_handle, start_frame, sequence_length = batch_size)
		video_handle.release()
		cv2.destroyAllWindows()
		# Process the frames to retrieve only the faces, and construct the batch
		batch = []
		for i, frame in enumerate(frames):
			# Retrieve detected faces and their positions. Throw an <AssertionError> in case of no detected faces.
			faces, face_positions = [], []
			try:
				faces, face_positions = get_faces(frame)
			except AssertionError:
				raise AttributeError("No faces detected in {}".format(video_path))
			# Check whether 1 face was detected. If more - throw a ValueError
			if len(face_positions) == 1:
				tensor_img = tensor_transform(Image.fromarray(faces[0]))
				batch.append(tensor_img)
			else:
				# ToDo: Multiple faces, choose closest one
				raise ValueError("Multiple faces detected in {}".format(video_path))
	except:
		# An error occured
		video_handle.release()
		cv2.destroyAllWindows()
		raise

	# Stack list of tensors into a single tensor on device
	batch = torch.stack(batch).to(device)
	
	return batch


# Create a batch of face images from various videos
def create_disparate_batch(real_video_generator, fake_video_generator, model_type, device, batch_size = 16):
	tensor_transform = transform.model_transforms[model_type]

	# Process the frames to retrieve only the faces, and construct the batch
	batch = []
	labels = []
	while len(batch) < batch_size:
		video_path = None
		label = None
		if random.random() < 0.5:
			video_path = next(real_video_generator)
			label = 1
		else:
			video_path = next(fake_video_generator)
			label = 0

		# Grab a frame
		video_handle = cv2.VideoCapture(video_path)
		if (video_handle.isOpened() == True):
			# VideoCapture() succesfully opened the video
			video_length = video_handle.get(7)
			frame = opencv_helpers.getRandomFrame(video_handle)
			video_handle.release()
			cv2.destroyAllWindows()

			# Retrieve detected faces and their positions. Throw an <AssertionError> in case of no detected faces.
			faces, face_positions = [], []
			try:
				faces, face_positions = get_faces(frame)
			except AssertionError:
				print("No faces detected in {}".format(video_path))
			# Check whether 1 face was detected. If more - throw a ValueError
			if len(face_positions) == 1:
				tensor_img = tensor_transform(Image.fromarray(faces[0]))
				batch.append(tensor_img)
				labels.append(label)
			elif len(face_positions) >= 2:
				# ToDo: Multiple faces, choose closest one
				print("Multiple faces detected in {}".format(video_path))
				put_file_in_folder(file_path = video_path, folder = "multiple_faces")
		else:
			# VideoCapture() failed to open the video
			print("VideoCapture() failed to open {}".format(video_path))
			video_handle.release()
			cv2.destroyAllWindows()
			put_file_in_folder(file_path = video_path, folder = "bad_samples")

	# Stack list of tensors into a single tensor on device
	batch = torch.stack(batch).to(device)
	# Create label tensor
	labels = torch.tensor(labels, device = device, requires_grad = False, dtype = torch.float)
	labels = labels.view(-1,1)
	
	return batch, labels