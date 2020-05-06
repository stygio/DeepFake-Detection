import os
import shutil
from tqdm import tqdm

dataset_path = "D:/Kaggle_Dataset"


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


def short_videos(dataset_path):
	# Move videos which are too short to bad samples
	import cv2
	min_length = 32
	# Paths to fake videos (multiple folders of fakes)
	fake_video_paths = os.path.join(dataset_path, 'manipulated_sequences')
	fake_video_paths = [os.path.join(fake_video_paths, x) for x in os.listdir(fake_video_paths)]
	fake_video_paths = [os.path.join(x, 'c23', 'videos') for x in fake_video_paths]

	for folder_path in fake_video_paths:
		bs_path = os.path.join(folder_path, "bad_samples")
		os.makedirs(bs_path, exist_ok=True)
		fake_videos = [x for x in os.listdir(folder_path) if x not in 
			["metadata.json", "multiple_faces", "bad_samples", "bounding_boxes"]]

		for video in tqdm(fake_videos, desc = folder_path):
			video_path = os.path.join(folder_path, video)
			video_handle = cv2.VideoCapture(video_path)
			video_length = video_handle.get(7)
			video_handle.release()
			if video_length < min_length:
				new_path = os.path.join(bs_path, video)
				os.replace(video_path, new_path)