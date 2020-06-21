import os
import json
from shutil import rmtree
from tqdm import tqdm

from tools.opencv_helpers import extract_faces_from_video

kaggle_path = "D:/Kaggle_Dataset"
face_forensics_path = "D:/FaceForensics_Dataset"


# Extract faces from the chosen dataset
def extract_faces(dataset):
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
		folder_paths = sorted(folder_paths, key = lambda d: int(d.split('_')[-1]))
	else:
		raise Exception('Invalid dataset choice: ' + dataset)

	# For each folder
	for folder_path in folder_paths:
		# Get videos in folder_path
		videos = os.listdir(folder_path)
		videos = [x for x in videos if x not in ['metadata.json', 'bounding_boxes', 'bad_samples', 'multiple_faces', 'images']]

		# Only retrieve faces from real videos
		if dataset == 'kaggle':
			metadata = json.load(open(os.path.join(folder_path, 'metadata.json')))
			videos = [x for x in videos if metadata[x]['label'] == 'REAL']

		# Main extraction loop for each video
		for video in tqdm(videos, desc = folder_path):
			video_path = os.path.join(folder_path, video)
			bb_path = os.path.join(folder_path, 'bounding_boxes', os.path.splitext(video)[0]) + '.json'
			target_folder = os.path.join(folder_path, 'images', os.path.splitext(video)[0])

			# Check whether the video is tagged for multiple faces
			multiple_faces = json.load(open(bb_path))['multiple_faces']
			
			# Only extract if the video hasn't been extracted already & has a single face
			if not os.path.isdir(target_folder) and not multiple_faces:
				# In case of a KeyboardInterrupt remove the folder currently being made
				try:
					os.makedirs(target_folder)
					extract_faces_from_video(video_path, bb_path, target_folder)
				except KeyboardInterrupt:
					rmtree(target_folder)
					raise
	

extract_faces('faceforensics')