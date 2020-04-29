from torch import load

from models.xception import xception
from models.inception_v3 import inception_v3
from models.resnet152 import resnet152
from models.resnext101 import resnext101



"""
Function for retrieving a chosen model
	model_name - which model to choose
	model_path - model path if a local version should be used
	training   - activates the dropout layer
"""
def get_model(model_name, training, model_path = None):
	network = None

	if model_name == "xception":
		if model_path == None:
			network = xception(pretrained = True)
		else:
			network = xception(pretrained = False)
			network.load_state_dict(load(model_path))
	elif model_name == "inception_v3":
		if model_path == None:
			network = inception_v3(pretrained = True)
		else:
			network = inception_v3(pretrained = False)
			network.load_state_dict(load(model_path))
	elif model_name == "resnet152":
		if model_path == None:
			network = resnet152(pretrained = True)
		else:
			network = resnet152(pretrained = False)
			network.load_state_dict(load(model_path))
	elif model_name == "resnext101":
		if model_path == None:
			network = resnext101(pretrained = True)
		else:
			network = resnext101(pretrained = False)
			network.load_state_dict(load(model_path))
	else:
		raise Exception("Invalid model chosen.")

	# Disable gradients
	for param in network.parameters():
		param.requires_grad = False

	# Set model mode (self.training True/False)
	if training:
		network = network.train()
	else:
		network = network.eval()

	return network