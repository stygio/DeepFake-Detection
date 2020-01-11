# import torch
# import torch.nn as nn
# import torch.nn.functional as functional
# import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
# from xception import xception
import numpy as np
import cv2
import os
import random
import face_recognition


def show_test_img(test_img):
	cv2.imshow("test", test_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# Return a generator of random real/fake image directories
def get_random_directory(real_or_fake):
	if real_or_fake == "real":
		dir_list = os.listdir(real_img_dir)
		# random.shuffle(dir_list)
		for name in dir_list:
			yield os.path.join(real_img_dir, name)
	elif real_or_fake == "fake":
		dir_list = os.listdir(fake_img_dir)
		# random.shuffle(dir_list)
		for name in dir_list:
			yield os.path.join(fake_img_dir, name)
	else:
		raise Exception("Invalid value passed with <real_or_fake>")

# Image preprocessing: face detection, cropping, resizing
def preprocess_image(img_path, resize_dim = (299, 299)):
	img = cv2.imread(img_path)
	rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	
	# Acquire face_locations, which is a list of tuples with locations
	# of bounding boxes specified as (top, right, bottom, left)
	face_locations = face_recognition.face_locations(rgb_img, model="cnn")
	faces = []
	for face in face_locations:
		(top, right, bottom, left) = face
		crop_height = bottom - top
		crop_width = right - left
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

		print("Debug: crop_height: {}, crop_width: {}, crop_diff: {}".format(crop_height, crop_width, crop_diff))
		print("Debug: top: {}, bottom: {}, left: {}, right: {}".format(top, bottom, left, right))

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
		print("Debug: Cropped image shape: {}".format(np.shape(cropped_img)))
		
		# Resize
		resized_img = cv2.resize(img, resize_dim, interpolation = cv2.INTER_AREA)
		print("Debug: Resized image shape: {}".format(np.shape(resized_img)))
		show_test_img(resized_img)

		faces.append(resized_img)

	return faces


# real_img_dir = "E:\\FaceForensics_Dataset\\original_sequences\\c23\\images"
real_img_dir = "C:\\Users\\Andrzej\\Pictures\\tempgarbage"
fake_img_dir = "E:\\FaceForensics_Dataset\\manipulated_sequences\\DeepFakeDetection\\c23\\images"
real_img_dirs = get_random_directory("real")
fake_img_dirs = get_random_directory("fake")
crop_factor = 1.2

def test():
	chosen_dir = next(real_img_dirs)
	for filename in os.listdir(chosen_dir):
		full_path = os.path.join(chosen_dir, filename)
		print("Processing file: {}".format(full_path))
		faces = preprocess_image(img_path = full_path)
		yield faces
		
test()