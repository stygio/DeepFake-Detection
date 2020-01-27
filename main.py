import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os

import tools.miscellaneous as misc
from tools.preprocessing import get_faces, faces_to_tensor, create_batch
from tools.opencv_helpers import loadFrameSequence
from networks.xception import xception


# real_img_dir = "C:\\Users\\Andrzej\\Pictures\\tempgarbage"
real_img_dir = "E:\\FaceForensics_Dataset\\original_sequences\\c23\\images"
fake_img_dir = "E:\\FaceForensics_Dataset\\manipulated_sequences\\DeepFakeDetection\\c23\\images"
real_img_dirs = misc.get_random_directory(real_img_dir)
fake_img_dirs = misc.get_random_directory(fake_img_dir)

video_path = "E:\\FaceForensics_Dataset\\original_sequences\\c23\\videos\\000.mp4"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test1():
	chosen_dir = next(real_img_dirs)
	faces = None
	for filename in os.listdir(chosen_dir):
		full_path = os.path.join(chosen_dir, filename)
		print("Processing file: {}".format(full_path))
		faces, face_positions = get_faces(img = full_path, isPath = True)
		print("Face positions (Y, X): {}".format(face_positions))
		break

	return faces
	

def test2():
	network = xception(pretrained = True).to(device)
	network.zero_grad()
	criterion = nn.BCELoss()
	optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
	data = faces_to_tensor(test1(), device)
	output = network(data)
	print(output)

	return output

def test3():
	batch = create_batch(video_path, device)
	return batch
