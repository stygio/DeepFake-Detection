from torchvision import models
import torch.nn as nn

def resnet152(pretrained = True):
	model = models.resnet152(pretrained = pretrained)
	num_features = model.fc.in_features
	model.fc = nn.Linear(num_features, 1)

	return model