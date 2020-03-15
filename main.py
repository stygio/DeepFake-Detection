import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os

import tools.miscellaneous as misc
from tools.network import *


real_vid_dirs = [	"E:\\FaceForensics_Dataset\\original_sequences\\c23\\videos"]
fake_vid_dirs = [	"E:\\FaceForensics_Dataset\\manipulated_sequences\\DeepFakeDetection\\c23\\videos",
					"E:\\FaceForensics_Dataset\\manipulated_sequences\\Deepfakes\\c23\\videos",
					"E:\\FaceForensics_Dataset\\manipulated_sequences\\Face2Face\\c23\\videos",
					"E:\\FaceForensics_Dataset\\manipulated_sequences\\FaceSwap\\c23\\videos",
					"E:\\FaceForensics_Dataset\\manipulated_sequences\\NeuralTextures\\c23\\videos",
]

kaggle_path = "E:\\Kaggle_Dataset"

one_face_vp = "E:\\FaceForensics_Dataset\\original_sequences\\c23\\videos\\000.mp4"
no_face_vp = "C:\\Users\\Andrzej\\Videos\\MazeEscape\\Maze1.mp4"
two_face_vp = "E:\\FaceForensics_Dataset\\original_sequences\\c23\\videos\\01__walking_and_outside_surprised.mp4"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_training_ff():
	train_faceforensics(real_vid_dirs, fake_vid_dirs, batch_size = 32, epochs = 10, lr = 0.001, model = "inception_v3", only_fc_layer = True)

def test_training_kaggle():
	train_kaggle(kaggle_path, model_name = "xception", model_weights_path = "models/saved_models/xception_date$03122020_time$183409.pt", 
		batch_size = 32, epochs = 10, lr = 0.001, only_fc_layer = True)

test_training_kaggle()