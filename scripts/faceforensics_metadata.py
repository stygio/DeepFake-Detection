import os
import json
from tqdm import tqdm
import random

dataset_path = 'd:/faceforensics_dataset'

# Paths to real videos
real_folder = os.path.join(dataset_path, 'original_sequences', 'c23', 'videos')
# Paths to fake videos (multiple folders of fakes)
fake_folders = os.path.join(dataset_path, 'manipulated_sequences')
fake_folders = [os.path.join(fake_folders, x) for x in os.listdir(fake_folders)]
fake_folders = [os.path.join(x, 'c23', 'videos') for x in fake_folders]

# # Fix misalignment in status as being multiple_faces
# for folder_path in fake_folders:
# 	real_bb_folder = os.path.join(real_folder, 'bounding_boxes')
# 	fake_bb_folder = os.path.join(folder_path, 'bounding_boxes')

# 	metadata_filename = os.path.join(folder_path, "metadata.json")
# 	metadata = json.load(open(metadata_filename))

# 	for k, v in metadata.items():
# 		fake_video = k
# 		fake_bb_path = os.path.join(fake_bb_folder, os.path.splitext(fake_video)[0]) + '.json'
# 		fake_bb = json.load(open(fake_bb_path))
# 		real_video = v['original']
# 		real_bb_path = os.path.join(real_bb_folder, os.path.splitext(real_video)[0]) + '.json'
# 		real_bb = json.load(open(real_bb_path))

# 		fake_bb['multiple_faces'] = real_bb['multiple_faces']
# 		os.remove(fake_bb_path)
# 		json.dump(fake_bb, open(fake_bb_path, 'w+'))


# Splits for train, test, val for youtube scraped part of faceforensics
train = json.load(open('scripts/ff_splits/train.json'))
test = json.load(open('scripts/ff_splits/test.json'))
val = json.load(open('scripts/ff_splits/val.json'))
flatten = lambda l: [item for sublist in l for item in sublist]
train = flatten(train)
test = flatten(test)
val = flatten(val)

# Randomly splitting google part of faceforensics
real_videos = [x for x in os.listdir(real_folder) if x not in 
		["metadata.json", "multiple_faces", "bad_samples", "bounding_boxes", "images"]]
google_dfdc_originals = [os.path.splitext(v)[0] for v in real_videos if os.path.splitext(v)[0] not in (train + test + val)]
random.shuffle(google_dfdc_originals)
split_1 = int(len(google_dfdc_originals) * 0.14)
split_2 = 2 * split_1
google_dfdc_originals_val = google_dfdc_originals[:split_1]
google_dfdc_originals_test = google_dfdc_originals[split_1:split_2]
google_dfdc_originals_train = google_dfdc_originals[split_2:]

# Go through real videos
real_metadata_filename = os.path.join(real_folder, "metadata.json")
if os.path.isfile(real_metadata_filename):
	os.remove(real_metadata_filename)

metadata = {}
for video in tqdm(real_videos, desc = real_folder):
	v, _ = os.path.splitext(video)
	if v in train:
		split = 'train'
	elif v in test:
		split = 'test'
	elif v in val:
		split = 'val'
	elif v in google_dfdc_originals:
		if v in google_dfdc_originals_train:
			split = 'train'
		elif v in google_dfdc_originals_test:
			split = 'test'
		elif v in google_dfdc_originals_val:
			split = 'val'
	else:
		split = 'train'
	metadata[video] = {}
	metadata[video]['split'] = split

metadata_file = open(real_metadata_filename, "w+")
json.dump(metadata, metadata_file)
metadata_file.close()

originals_metadata = metadata


# Go through the fake folders
for folder_path in fake_folders:
	metadata_filename = os.path.join(folder_path, "metadata.json")
	if os.path.isfile(metadata_filename):
		os.remove(metadata_filename)

	fake_videos = [x for x in os.listdir(folder_path) if x not in 
		["metadata.json", "multiple_faces", "bad_samples", "bounding_boxes", "images"]]
	metadata = {}
	
	# DeepFakeDetection folder has a different naming scheme
	if 'DeepFakeDetection' in folder_path:
		for video in tqdm(fake_videos, desc = folder_path):
			v, _ = os.path.splitext(video)
			original_video = v[:2] + v[5:-10] + '.mp4'
			metadata[video] = {}
			metadata[video]['split'] = originals_metadata[original_video]['split']
			metadata[video]['original'] = original_video
	
	# The other fake video folders
	else:
		for video in tqdm(fake_videos, desc = folder_path):
			v, _ = os.path.splitext(video)
			original_video = v.split('_')[0]
			# if original_video in train:
			# 	split = 'train'
			# elif original_video in test:
			# 	split = 'test'
			# elif original_video in val:
			# 	split = 'val'
			# else:
			# 	raise Exception('File not in ff_splits: ' + original_video)
			original_video += '.mp4'
			metadata[video] = {}
			# metadata[video]['split'] = split
			metadata[video]['split'] = originals_metadata[original_video]['split']
			metadata[video]['original'] = original_video

	metadata_file = open(metadata_filename, "w+")
	json.dump(metadata, metadata_file)
	metadata_file.close()