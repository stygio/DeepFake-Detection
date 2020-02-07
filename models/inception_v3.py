from torchvision import models
import torch.nn as nn

def inception_v3(pretrained = True):
	model = models.inception_v3(pretrained = pretrained)
	# model.aux_logit = False
	num_features = model.fc.in_features
	model.fc = nn.Linear(num_features, 1)

	return model