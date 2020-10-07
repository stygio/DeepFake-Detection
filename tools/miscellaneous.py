import os
import random
import json
from datetime import datetime

log_dir = "outputs/logs/"


# Return a string with current timestamp
def timestamp():
	now = datetime.now()
	timestamp_string = now.strftime("_%m%d%Y_%H%M%S")
	return timestamp_string


"""
Return an infinite generator of random files from provided list of directories
	directory_list - list of directories with samples
"""
def get_random_file_from_list(directory_list):
	while True:
		# Choose a random directory from the list of sample directories
		chosen_dir = random.choice(directory_list)
		dir_contents = os.listdir(chosen_dir)
		# Choose a random file from the directory
		chosen_file = random.choice(dir_contents)
		file_path = os.path.join(chosen_dir, chosen_file)
		if os.path.isfile(file_path):
			yield file_path


"""
Return an infinite generator of random folders from provided path to directory
	directory - path to directory containing requested folders
"""
def get_random_folder_from_path(directory):
	while True:
		# Choose randomly from items in the directory
		dir_contents = os.listdir(directory)
		random.shuffle(dir_contents)
		for folder in dir_contents:
			# Construct path to the chosen folder and verify that it is a folder
			folder_path = os.path.join(directory, folder)
			if os.path.isdir(folder_path):
				yield folder_path


"""
Return an infinite generator of random folders from provided list of directories
	directory_list - list of folders with samples
"""
def get_random_folder_from_list(directory_list):
	while True:
		random.shuffle(directory_list)
		for folder_path in directory_list:
			if os.path.isdir(folder_path):
				yield folder_path


"""
Put a file in a folder in its current directory, 
	file_path - current path to the file
	folder    - target folder for the file, which is created if it doesn't exist yet
"""
def put_file_in_folder(file_path, folder):
	split_path = os.path.split(file_path)
	new_path = os.path.join(split_path[0], folder, split_path[1])
	os.makedirs(os.path.dirname(new_path), exist_ok=True)
	try:
		print("DEBUG: Moving '{}' to '{}'".format(split_path[1], "{}/{}".format(folder, split_path[1])))
		os.replace(file_path, new_path)
	except FileNotFoundError:
		raise FileNotFoundError("File specified by the path {} doesn't exist.".format(file_path))


"""
Creates a log file and returns the path to it
	model_type    - model type name
	header_string - string which will be at the top of the log file
"""
def create_log(base_filename, header_string):
	filename = base_filename + timestamp() + ".csv"
	filename = os.path.join(log_dir, filename)
	f = open(filename, "w+")
	f.write(header_string)
	f.close()

	return filename


"""
Adds a string to the specified log
	log_file   - path to log file
	log_string - string to be appended to the log
"""
def add_to_log(log_file, log_string):
	f = open(log_file, "a")
	f.write(log_string)
	f.close()


"""
Get the absolute path to the boundingbox file for a video
	video_path - path to the video
"""
def get_boundingbox_path(video_path):
	return os.path.join(os.path.dirname(video_path), "bounding_boxes", os.path.splitext(os.path.basename(video_path))[0]) + ".json"


"""
Get the absolute path to the directory containing extracted faces from a video
	video_path - path to the video
"""
def get_images_path(video_path):
	return os.path.join(os.path.dirname(video_path), "images", os.path.splitext(os.path.basename(video_path))[0])


"""
Assemble a list of training samples
	dataset 	 - name of dataset {faceforensics, kaggle}
	dataset_path - absolute path to the dataset
"""
def get_training_samples(dataset, dataset_path):
	training_samples = []

	if dataset == 'faceforensics':
		# List of sorted folders in the faceforensics directory
		original_sequences = os.path.join(dataset_path, 'original_sequences')
		real_folder = os.path.join(original_sequences, 'c23', 'videos')
		manipulated_sequences = os.path.join(dataset_path, 'manipulated_sequences')
		fake_folders = [os.path.join(manipulated_sequences, x) for x in os.listdir(manipulated_sequences)]
		fake_folders = [os.path.join(x, 'c23', 'videos') for x in fake_folders]
		# Collect training samples
		for folder_path in fake_folders:
			videos = os.listdir(folder_path)
			videos = [x for x in videos if x not in ["metadata.json", "bounding_boxes", "bad_samples", "multiple_faces", "images"]]
			metadata = os.path.join(folder_path, "metadata.json")
			metadata = json.load(open(metadata))
			# Added tuples of fake and corresponding real videos to the training_samples
			for video in videos:
				# Check if video is in the train split
				if metadata[video]['split'] == 'train':
					fake_video_path = os.path.join(folder_path, video)
					real_video_path = os.path.join(real_folder, metadata[video]['original'])
					training_samples.append((fake_video_path, real_video_path))

	elif dataset == 'kaggle':
		# List of sorted folders in the kaggle directory
		kaggle_folders = [os.path.join(dataset_path, x) for x in os.listdir(dataset_path)]
		kaggle_folders = sorted(kaggle_folders, key = lambda d: int(d.split('_')[-1]))
		# Collect training samples
		for folder_path in kaggle_folders:
			videos = os.listdir(folder_path)
			videos = [x for x in videos if x not in ["metadata.json", "bounding_boxes", "bad_samples", "multiple_faces", "images"]]
			metadata = os.path.join(folder_path, "metadata.json")
			metadata = json.load(open(metadata))
			# Added tuples of fake and corresponding real videos to the training_samples
			for video in videos:
				# Check if video is labeled as fake
				if metadata[video]['label'] == 'FAKE' and metadata[video]['split'] == 'train':
					fake_video_path = os.path.join(folder_path, video)
					real_video_path = os.path.join(folder_path, metadata[video]['original'])
					# Check for multiple faces flag
					bb_dict = json.load(open(get_boundingbox_path(real_video_path)))
					if bb_dict['multiple_faces'] == False:
						training_samples.append((fake_video_path, real_video_path))

	return training_samples


"""
Assemble a list of evaluation samples
	dataset 	 - name of dataset {faceforensics, kaggle}
	dataset_path - absolute path to the dataset
	split 		 - dataset split {val, test}
"""
def get_evaluation_samples(dataset, dataset_path, split):
	evaluation_samples = []

	if dataset == 'faceforensics':
		# Folders in the faceforensics directory
		original_sequences = os.path.join(dataset_path, 'original_sequences')
		real_folder = os.path.join(original_sequences, 'c23', 'videos')
		manipulated_sequences = os.path.join(dataset_path, 'manipulated_sequences')
		fake_folders = [os.path.join(manipulated_sequences, x) for x in os.listdir(manipulated_sequences)]
		fake_folders = [os.path.join(x, 'c23', 'videos') for x in fake_folders]

		# Originals
		videos = os.listdir(real_folder)
		videos = [x for x in videos if x not in ["metadata.json", "bounding_boxes", "bad_samples", "multiple_faces", "images"]]
		metadata = os.path.join(real_folder, "metadata.json")
		metadata = json.load(open(metadata))
		for video in videos:
			# Check if video belongs to the correct dataset split
			if metadata[video]['split'] == split:
				real_video_path = os.path.join(real_folder, video)
				evaluation_samples.append((real_video_path, 'REAL'))
		# Fake videos
		for folder_path in fake_folders:
			videos = os.listdir(folder_path)
			videos = [x for x in videos if x not in ["metadata.json", "bounding_boxes", "bad_samples", "multiple_faces", "images"]]
			metadata = os.path.join(folder_path, "metadata.json")
			metadata = json.load(open(metadata))
			for video in videos:
				# Check if video belongs to the correct dataset split
				if metadata[video]['split'] == split:
					fake_video_path = os.path.join(folder_path, video)
					evaluation_samples.append((fake_video_path, 'FAKE'))

	elif dataset == 'kaggle':
		# Folders in the kaggle directory
		folder_paths = [os.path.join(dataset_path, x) for x in os.listdir(dataset_path)]

		# Iterate through folders and construct list of samples
		for folder_path in folder_paths:
			videos = os.listdir(folder_path)
			videos = [x for x in videos if x not in ["metadata.json", "bounding_boxes", "bad_samples", "multiple_faces", "images"]]
			metadata = os.path.join(folder_path, "metadata.json")
			metadata = json.load(open(metadata))
			for video in videos:
				# Check if video belongs to the correct dataset split
				if metadata[video]['split'] == split:
					video_path = os.path.join(folder_path, video)
					# Check for multiple faces flag
					bb_dict = json.load(open(get_boundingbox_path(video_path)))
					if bb_dict['multiple_faces'] == False:
						evaluation_samples.append((video_path, metadata[video]['label']))

	return evaluation_samples