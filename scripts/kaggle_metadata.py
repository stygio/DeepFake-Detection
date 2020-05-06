import os
import json
from tqdm import tqdm

train_split = 0.8
val_split = 0.1
test_split = 0.1

assert train_split + val_split + test_split == 1, "Splits don't add up to 1."

kaggle_dataset_path = 'd:/kaggle_dataset'

kaggle_folders = [os.path.join(kaggle_dataset_path, x) for x in os.listdir(kaggle_dataset_path)]
kaggle_folders = sorted(kaggle_folders, key = lambda d: int(d.split('_')[-1]))

for folder in kaggle_folders:
	metadata_path = os.path.join(folder, 'metadata.json')
	metadata = json.load(open(metadata_path))

	# Get a list of the originals
	real_videos = []
	for k, v in metadata.items():
		if v['label'] == 'REAL':
			real_videos.append(k)

	# Calculate appropriate indexes for splits
	num_samples	= len(real_videos)
	train_end = int(train_split * num_samples)
	val_end = int((train_split + val_split) * num_samples)
	# Divide real samples according to split
	train_set	= real_videos[:train_end]
	val_set 	= real_videos[train_end:val_end]
	test_set 	= real_videos[val_end:]
	# Check whether the samples were split without intersection
	assert len(set(train_set).intersection(val_set)) == 0, "Matching elements in train_set & val_set."
	assert len(set(train_set).intersection(test_set)) == 0, "Matching elements in train_set & test_set."
	assert len(set(val_set).intersection(test_set)) == 0, "Matching elements in val_set & test_set."

	# Assign fakes to appropriate sets
	for k, v in metadata.items():
		if v['label'] == 'FAKE':
			if v['original'] in train_set:
				train_set.append(k)
			elif v['original'] in val_set:
				val_set.append(k)
			elif v['original'] in test_set:
				test_set.append(k)
			else:
				raise Exception('{} has original {} which is not in any of the splits.'.format(k, v['original']))

	# Write splits
	for sample in train_set:
		metadata[sample]['split'] = 'train'
	for sample in val_set:
		metadata[sample]['split'] = 'val'
	for sample in test_set:
		metadata[sample]['split'] = 'test'

	# Remove old file and write new one
	os.remove(metadata_path)
	json.dump(metadata, open(metadata_path, "w+"))