from torch import load

from models.xception import Binary_Xception
from models.inception_v3 import Binary_Inception
from models.resnet152 import Binary_ResNet152
from models.resnext101 import Binary_ResNeXt101



"""
Function for retrieving a chosen model
	model_name - which model to choose
	model_path - model path if a local version should be used
"""
def get_model(model_name, model_path = None):
	network = None

	if model_name == "xception":
		if model_path == None:
			network = Binary_Xception(pretrained = True)
		else:
			network = Binary_Xception(pretrained = False)
			network.load_state_dict(load(model_path))
	elif model_name == "inception_v3":
		if model_path == None:
			network = Binary_Inception(pretrained = True)
		else:
			network = Binary_Inception(pretrained = False)
			network.load_state_dict(load(model_path))
	elif model_name == "resnet152":
		if model_path == None:
			network = Binary_ResNet152(pretrained = True)
		else:
			network = Binary_ResNet152(pretrained = False)
			network.load_state_dict(load(model_path))
	elif model_name == "resnext101":
		if model_path == None:
			network = Binary_ResNeXt101(pretrained = True)
		else:
			network = Binary_ResNeXt101(pretrained = False)
			network.load_state_dict(load(model_path))
	else:
		raise Exception("Invalid model chosen.")

	# Disable gradients
	for param in network.parameters():
		param.requires_grad = False

	return network