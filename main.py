# import torch
# import torch.nn as nn
# import torch.nn.functional as functional
# import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
# from xception import xception
import os

from tools.preprocessing import preprocess_image
from tools.opencv_helpers import loadFrameSequence
import tools.miscellaneous as misc


# real_img_dir = "C:\\Users\\Andrzej\\Pictures\\tempgarbage"
real_img_dir = "E:\\FaceForensics_Dataset\\original_sequences\\c23\\images"
fake_img_dir = "E:\\FaceForensics_Dataset\\manipulated_sequences\\DeepFakeDetection\\c23\\images"
real_img_dirs = misc.get_random_directory(real_img_dir)
fake_img_dirs = misc.get_random_directory(fake_img_dir)

def test():
	chosen_dir = next(real_img_dirs)
	for filename in os.listdir(chosen_dir):
		full_path = os.path.join(chosen_dir, filename)
		print("Processing file: {}".format(full_path))
		faces = preprocess_image(img_path = full_path)
		# yield faces
		
test()