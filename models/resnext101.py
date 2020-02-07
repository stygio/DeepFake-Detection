# import torch
import torch.nn as nn
from torchvision import models

def resnext101(pretrained = True):
	# model = torch.hub.load('pytorch/vision', 'resnext101_32x8d', pretrained=True)
	model = models.resnext101_32x8d(pretrained = pretrained)
	num_features = model.fc.in_features
	model.fc = nn.Linear(num_features, 1)

	return model