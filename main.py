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
from tools.network import train_fc_layer
from models.xception import xception


# real_img_dir = "C:\\Users\\Andrzej\\Pictures\\tempgarbage"
real_img_dir = "E:\\FaceForensics_Dataset\\original_sequences\\c23\\images"
fake_img_dir = "E:\\FaceForensics_Dataset\\manipulated_sequences\\DeepFakeDetection\\c23\\images"
real_img_dirs = misc.get_random_file_path(real_img_dir)
fake_img_dirs = misc.get_random_file_path(fake_img_dir)

real_vid_dir = "E:\\FaceForensics_Dataset\\original_sequences\\c23\\videos"
fake_vid_dir = "E:\\FaceForensics_Dataset\\manipulated_sequences\\DeepFakeDetection\\c23\\videos"

real_vid_dirs = [	"E:\\FaceForensics_Dataset\\original_sequences\\c23\\videos"]
fake_vid_dirs = [	"E:\\FaceForensics_Dataset\\manipulated_sequences\\DeepFakeDetection\\c23\\videos",
					"E:\\FaceForensics_Dataset\\manipulated_sequences\\Deepfakes\\c23\\videos",
					"E:\\FaceForensics_Dataset\\manipulated_sequences\\Face2Face\\c23\\videos",
					"E:\\FaceForensics_Dataset\\manipulated_sequences\\FaceSwap\\c23\\videos",
					"E:\\FaceForensics_Dataset\\manipulated_sequences\\NeuralTextures\\c23\\videos",
]

one_face_vp = "E:\\FaceForensics_Dataset\\original_sequences\\c23\\videos\\000.mp4"
no_face_vp = "C:\\Users\\Andrzej\\Videos\\MazeEscape\\Maze1.mp4"
two_face_vp = "E:\\FaceForensics_Dataset\\original_sequences\\c23\\videos\\01__walking_and_outside_surprised.mp4"

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
	optimizer = optim.SGD(network.fc_binary.parameters(), lr=0.001, momentum=0.9)
	data = faces_to_tensor(test1(), device)
	output = network(data)
	print("Network output: {}".format(output))
	return network


def test3():
	train_fc_layer(real_vid_dirs, fake_vid_dirs, epochs = 10, batch_size = 20)

test3()