import numpy as np
import random
import cv2
from PIL import Image
import os
import json
from shutil import copyfile as copyfile
from tqdm import tqdm

from tools import preprocessing, opencv_helpers

kaggle_path = "D:/Kaggle_Dataset"
face_forensics_path = "D:/FaceForensics_Dataset"

test_video = 'd:/faceforensics_dataset/original_sequences/c23/videos/172.mp4'


def get_json_path(video_path):
	return  os.path.join(os.path.dirname(video_path), 'bounding_boxes', os.path.splitext(os.path.basename(video_path))[0]) + '.json'


def fix_bb_file(video_path, multiple_face_threshold = 0.1):
	# Get the boundingboxes
	bb_path = get_json_path(video_path)
	metadata_dict = json.load(open(bb_path))
	
	# Get number of last frame in boundbingbox file (they are numbered from 0)
	last_frame = max([int(f) for f in metadata_dict.keys() if f != 'multiple_faces'])

	# Get video handle and length
	video_handle = cv2.VideoCapture(video_path)
	video_length = video_handle.get(7)
	
	# Check if there are missing frames in the file
	# The video_length is the number of frames, last frame is the number of the last frame, though they start at 0
	if video_length > last_frame + 1:
		
		# If we grab batch_size frames in segments the last part of the video 
		# will have a segment with a number of frames <= batch_size
		leftover_frames = video_length - 1 - last_frame
		leftover_start = last_frame + 1
		current_frame = leftover_start
		# Handle leftover frames (not enough for a batch of batch_size frames)
		frames = opencv_helpers.load_video_segment(video_handle, start_frame = leftover_start, segment_length = leftover_frames)
		video_handle.release()

		# Process the frames
		batch_of_boxes = preprocessing.get_bounding_boxes(frames)
		for n, boxes in enumerate(batch_of_boxes):
			metadata_dict[current_frame] = {}
			for i, box in enumerate(boxes):
				metadata_dict[current_frame][i] = {}
				metadata_dict[current_frame][i]['top'] = box[0]
				metadata_dict[current_frame][i]['bottom'] = box[1]
				metadata_dict[current_frame][i]['left'] = box[2]
				metadata_dict[current_frame][i]['right'] = box[3]
				# preprocessing.show_test_img(preprocessing.crop_image(frames[n], box))
			current_frame += 1

	# Checking whether there are frames in the video with multiple faces detected
	# Detected faces are saved as keys starting from '0', so '1' means there is another
	multiple_faces = ['1' in bb.keys() for bb in metadata_dict.values() if type(bb) is dict].count(True)
	multiple_faces = True if multiple_faces/video_length > multiple_face_threshold else False
	# Remove existing entry from dict and replace it
	metadata_dict.pop('multiple_faces', None)
	metadata_dict['multiple_faces'] = multiple_faces

	# Replace the file with new version
	os.remove(bb_path)
	json.dump(metadata_dict, open(bb_path, "w+"))


def fix_boundingboxes(dataset, mobilenet_gpu_allocation = 0.75):
	# Initialize face recognition mobilenet
	preprocessing.initialize_mobilenet(mobilenet_gpu_allocation)

	# Retrieve folder_paths based on dataset
	if dataset == 'faceforensics':
		 # Folders in the faceforensics directory
		original_sequences = os.path.join(face_forensics_path, 'original_sequences')
		real_folder = os.path.join(original_sequences, 'c23', 'videos')
		manipulated_sequences = os.path.join(face_forensics_path, 'manipulated_sequences')
		fake_folders = [os.path.join(manipulated_sequences, x) for x in os.listdir(manipulated_sequences)]
		fake_folders = [os.path.join(x, 'c23', 'videos') for x in fake_folders]
		folder_paths = [real_folder] + fake_folders
	elif dataset == 'kaggle':
		folder_paths = [os.path.join(kaggle_path, x) for x in os.listdir(kaggle_path)]
	else:
		raise Exception('Invalid dataset choice: ' + dataset)

	for folder_path in folder_paths:
		videos = os.listdir(folder_path)
		videos = [x for x in videos if x not in ["metadata.json", "bounding_boxes", "bad_samples", "multiple_faces"]]

		for video in tqdm(videos, desc = folder_path):
			video_path = os.path.join(folder_path, video)
			fix_bb_file(video_path)


fix_boundingboxes('faceforensics')
