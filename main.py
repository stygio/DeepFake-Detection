import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os

import tools.miscellaneous as misc
from tools.preprocessing import get_faces, faces_to_tensor
from tools.opencv_helpers import loadFrameSequence
from networks.xception import xception


# real_img_dir = "C:\\Users\\Andrzej\\Pictures\\tempgarbage"
real_img_dir = "E:\\FaceForensics_Dataset\\original_sequences\\c23\\images"
fake_img_dir = "E:\\FaceForensics_Dataset\\manipulated_sequences\\DeepFakeDetection\\c23\\images"
real_img_dirs = misc.get_random_directory(real_img_dir)
fake_img_dirs = misc.get_random_directory(fake_img_dir)


def test1():
	chosen_dir = next(real_img_dirs)
	faces = None
	for filename in os.listdir(chosen_dir):
		full_path = os.path.join(chosen_dir, filename)
		print("Processing file: {}".format(full_path))
		faces = get_faces(img_path = full_path)
		break

	return faces
	

def test2():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	network = xception(pretrained = True).to(device)
	network.zero_grad()
	criterion = nn.BCELoss()
	optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
	data = faces_to_tensor(test1(), device)
	output = network(data)

	return output


test2()