"""
Library of functions for image preprocessing
"""

import numpy as np
import cv2
import random
import face_recognition
from torch import from_numpy as torch_from_numpy

crop_factor = 1.2

def show_test_img(test_img):
	cv2.imshow("test", test_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# Image preprocessing: face detection, cropping, resizing
def get_faces(img_path, resize_dim = (299, 299)):
	# Load image and resize if it's too big
	img = cv2.imread(img_path)
	if np.shape(img)[0] > 720:
		scale_factor = 720/np.shape(img)[0] # percent of original size
		width = int(img.shape[1] * scale_factor)
		height = int(img.shape[0] * scale_factor)
		dim = (width, height)
		# resize image
		img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	print(np.shape(rgb_img))
	
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

		# print("Debug: crop_height: {}, crop_width: {}, crop_diff: {}".format(crop_height, crop_width, crop_diff))
		# print("Debug: top: {}, bottom: {}, left: {}, right: {}".format(top, bottom, left, right))

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
		# print("Debug: Cropped image shape: {}".format(np.shape(cropped_img)))
		
		# Resize
		resized_img = cv2.resize(img, resize_dim, interpolation = cv2.INTER_AREA)
		# print("Debug: Resized image shape: {}".format(np.shape(resized_img)))
		# show_test_img(resized_img)

		faces.append(resized_img)

	return np.array(faces)


# Reshape array of faces and create tensor
def faces_to_tensor(faces, device):
	data = np.moveaxis(faces, -1, 1)
	data = torch_from_numpy(data).float().to(device)
	return data