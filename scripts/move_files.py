import os
import shutil
import json
from tqdm import tqdm


kaggle_path = "D:/Kaggle_Dataset"
face_forensics_path = "D:/FaceForensics_Dataset"


def move_folders():
	# Move files from 'dataset_path/*/folders' back to 'dataset_path/*''
	folders = ["bad_samples", "multiple_faces"]
	for directory in tqdm(os.listdir(dataset_path), desc = "Progress"):
		directory = os.path.join(dataset_path, directory)
		for folder in folders:
			folder_path = os.path.join(directory, folder)
			if os.path.isdir(folder_path):
				for file in os.listdir(folder_path):
					file_path = os.path.join(folder_path, file)
					new_file_path = os.path.join(directory, file)
					os.replace(file_path, new_file_path)


def delete_folders():
	# Delete 'dataset_path/*/folders'
	folders = ["multiple_faces"]
	for directory in tqdm(os.listdir(dataset_path), desc = "Progress"):
		directory = os.path.join(dataset_path, directory)
		for folder in folders:
			folder_path = os.path.join(directory, folder)
			if os.path.isdir(folder_path):
				shutil.rmtree(folder_path)


# For faceforensics
def short_videos(dataset_path):
	# Move videos which are too short to bad samples
	import cv2
	min_length = 60
	# Paths to fake videos (multiple folders of fakes)
	fake_video_paths = os.path.join(dataset_path, 'manipulated_sequences')
	fake_video_paths = [os.path.join(fake_video_paths, x) for x in os.listdir(fake_video_paths)]
	fake_video_paths = [os.path.join(x, 'c23', 'videos') for x in fake_video_paths]

	for folder_path in fake_video_paths:
		bs_path = os.path.join(folder_path, "bad_samples")
		os.makedirs(bs_path, exist_ok=True)
		fake_videos = [x for x in os.listdir(folder_path) if x not in 
			["metadata.json", "multiple_faces", "bad_samples", "bounding_boxes", "images"]]

		for video in tqdm(fake_videos, desc = folder_path):
			video_path = os.path.join(folder_path, video)
			video_handle = cv2.VideoCapture(video_path)
			video_length = video_handle.get(7)
			video_handle.release()
			if video_length < min_length:
				new_path = os.path.join(bs_path, video)
				os.replace(video_path, new_path)


# Sort videos tagged with multiple faces into seperate directories
def move_videos_with_multiple_faces(dataset):
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
		# Kaggle folders
		folder_paths = [os.path.join(kaggle_path, x) for x in os.listdir(kaggle_path)]
		folder_paths = sorted(folder_paths, key = lambda d: int(d.split('_')[-1]))
	else:
		raise Exception('Invalid dataset choice: ' + dataset)

	for folder_path in folder_paths:
		# Make a folder for videos with multiple faces if it doesn't exist
		multiple_faces_path = os.path.join(folder_path, 'multiple_faces')
		os.makedirs(multiple_faces_path, exist_ok=True)
		# Get videos in folder_path
		videos = os.listdir(folder_path)
		videos = [x for x in videos if x not in ['metadata.json', 'bounding_boxes', 'bad_samples', 'multiple_faces', 'images']]
		
		# Main movement loop
		for video in tqdm(videos, desc = folder_path):
			# Path to the video's bounding box file
			bb_path = os.path.join(folder_path, 'bounding_boxes', os.path.splitext(os.path.basename(video))[0]) + '.json'
			# Check whether the video is tagged for multiple faces
			multiple_faces = json.load(open(bb_path))['multiple_faces']

			if multiple_faces:
				file_path = os.path.join(folder_path, video)
				new_file_path = os.path.join(folder_path, 'multiple_faces', video)
				os.replace(file_path, new_file_path)


move_videos_with_multiple_faces('faceforensics')