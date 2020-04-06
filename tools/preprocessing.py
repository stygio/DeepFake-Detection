"""
Library of functions for image preprocessing
"""

import numpy as np
import random
import cv2
from PIL import Image
import torch
import time
import tensorflow as tf
import os

from tools import opencv_helpers
from tools.miscellaneous import put_file_in_folder
from models import transform

crop_factor = 1.3
sess, image_tensor, boxes_tensor, scores_tensor, num_detections = None, None, None, None, None


def show_test_img(test_img):
	cv2.imshow("test", test_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def initialize_mobilenet():
	global sess, image_tensor, boxes_tensor, scores_tensor, num_detections
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
	
	detection_graph = tf.Graph()
	with detection_graph.as_default():
		od_graph_def = tf.compat.v1.GraphDef()
		with tf.io.gfile.GFile('models/mobilenet_face/frozen_inference_graph_face.pb', 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')
		config = tf.compat.v1.ConfigProto()
		config.gpu_options.allow_growth = False
		config.gpu_options.per_process_gpu_memory_fraction=0.4
		sess = tf.compat.v1.Session(graph=detection_graph, config=config)
		image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
		boxes_tensor = detection_graph.get_tensor_by_name('detection_boxes:0')    
		scores_tensor = detection_graph.get_tensor_by_name('detection_scores:0')
		num_detections = detection_graph.get_tensor_by_name('num_detections:0')


def get_mobilenet_faces(image):
	# Minimum threshold to qualify a detected object as a face
	mobilenet_score_threshold = 0.4

	(im_height,im_width)=image.shape[:-1]
	imgs=np.array([image])
	(boxes, scores) = sess.run(
		[boxes_tensor, scores_tensor],
		feed_dict={image_tensor: imgs})

	# Grab the box which the highest scoring face
	max_ = np.where(scores == scores.max())[0][0]
	box = boxes[0][max_]
	ymin, xmin, ymax, xmax = box
	(left, right, top, bottom) = (xmin * im_width, xmax * im_width, 
								ymin * im_height, ymax * im_height)
	face = (int(top), int(right), int(bottom), int(left))
	faces = [face]

	# Append any other faces
	for i, box in enumerate(boxes[0]):
		# Check whether the box's score is above the threshold and isn't the max score (that one is already added)
		if scores[0][i] > mobilenet_score_threshold and i != max_:
			ymin, xmin, ymax, xmax = box
			(left, right, top, bottom) = (xmin * im_width, xmax * im_width, 
										ymin * im_height, ymax * im_height)
			face = (int(top), int(right), int(bottom), int(left))
			faces.append(face)

	return faces


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
	face_locations = get_mobilenet_faces(rgb_img)

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
		
		# Append transformed face and its position in the image
		faces.append(cropped_img)
		face_positions.append((face_Y, face_X))

	return faces, face_positions

