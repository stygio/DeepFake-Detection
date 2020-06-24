from torch import load

from models.xception import Binary_Xception
from models.inception_v3 import Binary_Inception
from models.resnet152 import Binary_ResNet152
from models.resnext101 import Binary_ResNeXt101
from models.efficientnet import Binary_EfficientNet

"""
Display network parameter counts in ratios of total parameters in the model
	model - pytorch model to be checked
"""
def count_parameters(model):
	total_parameters = sum(p.numel() for p in model.parameters())
	classifier_parameters = sum(p.numel() for p in model.classifier_parameters())
	higher_parameters = sum(p.numel() for p in model.higher_level_parameters())
	lower_parameters = sum(p.numel() for p in model.lower_level_parameters())

	print('Model parameter ratios: Classifier {:.2f}%, Higher {:.2f}%, Lower {:.2f}%'.format(
		classifier_parameters/total_parameters*100, higher_parameters/total_parameters*100, lower_parameters/total_parameters*100))

"""
Function for retrieving a chosen model
	model_name - which model to choose
	model_path - model path if a local version should be used
	pretrained - should it load a model with pretrained weights
"""
def get_model(model_name, model_path = None, pretrained = False):
	network = None

	if model_name == "xception":
		if model_path == None:
			network = Binary_Xception(pretrained = pretrained)
		else:
			network = Binary_Xception(pretrained = False)
			network.load_state_dict(load(model_path))
	
	elif model_name == "inception_v3":
		if model_path == None:
			network = Binary_Inception(pretrained = pretrained)
		else:
			network = Binary_Inception(pretrained = False)
			network.load_state_dict(load(model_path))
	
	elif model_name == "resnet152":
		if model_path == None:
			network = Binary_ResNet152(pretrained = pretrained)
		else:
			network = Binary_ResNet152(pretrained = False)
			network.load_state_dict(load(model_path))
	
	elif model_name == "resnext101":
		if model_path == None:
			network = Binary_ResNeXt101(pretrained = pretrained)
		else:
			network = Binary_ResNeXt101(pretrained = False)
			network.load_state_dict(load(model_path))
	
	elif model_name == "efficientnet-b5":
		if model_path:
			network = Binary_EfficientNet.from_name(model_name, {'num_classes': 1})
			network.load_state_dict(load(model_path))
		elif pretrained:
			network = Binary_EfficientNet.from_pretrained(model_name, num_classes = 1)
		else:
			network = Binary_EfficientNet.from_name(model_name, {'num_classes': 1})


	else:
		raise Exception("Invalid model chosen.")

	# Disable gradients
	for param in network.parameters():
		param.requires_grad = False

	return network