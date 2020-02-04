from torchvision import models
import torch.nn as nn

def resnext101(pretrained = True):
	# model = models.resnext101_32x8d(pretrained = pretrained)
	model = torch.hub.load('pytorch/vision:v0.5.0', 'resnext101_32x8d', pretrained=True)
	num_features = model.fc.in_features
	model.fc = nn.Linear(num_features, 1)

	return model