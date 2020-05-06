import os
import json
from tqdm import tqdm

dataset_path = 'd:/faceforensics_dataset'

# Paths to real videos
real_folder = os.path.join(dataset_path, 'original_sequences', 'c23', 'videos')
# Paths to fake videos (multiple folders of fakes)
fake_folders = os.path.join(dataset_path, 'manipulated_sequences')
fake_folders = [os.path.join(fake_folders, x) for x in os.listdir(fake_folders)]
fake_folders = [os.path.join(x, 'c23', 'videos') for x in fake_folders]

# Splits for train, test, val
train = json.load(open('ff_splits/train.json'))
test = json.load(open('ff_splits/test.json'))
val = json.load(open('ff_splits/val.json'))
flatten = lambda l: [item for sublist in l for item in sublist]
train = flatten(train)
test = flatten(test)
val = flatten(val)


# Go through real videos
metadata_filename = os.path.join(real_folder, "metadata.json")
if not os.path.isfile(metadata_filename):
	real_videos = [x for x in os.listdir(real_folder) if x not in 
		["metadata.json", "multiple_faces", "bad_samples", "bounding_boxes"]]
	metadata = {}

	for video in tqdm(real_videos, desc = real_folder):
		v, _ = os.path.splitext(video)
		if v in train:
			split = 'train'
		elif v in test:
			split = 'test'
		elif v in val:
			split = 'val'
		else:
			split = 'train'
		metadata[video] = {}
		metadata[video]['split'] = split

	metadata_file = open(metadata_filename, "w+")
	json.dump(metadata, metadata_file)
	metadata_file.close()


# Go through the fake folders
for folder_path in fake_folders:
	metadata_filename = os.path.join(folder_path, "metadata.json")
	if not os.path.isfile(metadata_filename):
		fake_videos = [x for x in os.listdir(folder_path) if x not in 
			["metadata.json", "multiple_faces", "bad_samples", "bounding_boxes"]]
		metadata = {}
		
		# DeepFakeDetection folder has a different naming scheme
		if 'DeepFakeDetection' in folder_path:
			for video in tqdm(fake_videos, desc = folder_path):
				v, _ = os.path.splitext(video)
				original_video = v[:2] + v[5:-10] + '.mp4'
				metadata[video] = {}
				metadata[video]['split'] = 'train'
				metadata[video]['original'] = original_video
		
		# The other fake video folders
		else:
			for video in tqdm(fake_videos, desc = folder_path):
				v, _ = os.path.splitext(video)
				original_video = v.split('_')[0]
				if original_video in train:
					split = 'train'
				elif original_video in test:
					split = 'test'
				elif original_video in val:
					split = 'val'
				else:
					raise Exception('File not in ff_splits: ' + original_video)
				original_video += '.mp4'
				metadata[video] = {}
				metadata[video]['split'] = split
				metadata[video]['original'] = original_video

		metadata_file = open(metadata_filename, "w+")
		json.dump(metadata, metadata_file)
		metadata_file.close()