from torchvision.models.inception import Inception3, model_urls
from torchvision.models.utils import load_state_dict_from_url
import torch.nn as nn


class Binary_Inception(Inception3):
	"""
	My version of Inception with functions for unfreezing certain groups of layers.
	"""
	def __init__(self, pretrained = False, init_weights = True):
		# Constructor
		super(Binary_Inception, self).__init__(num_classes = 1, aux_logits = False, init_weights = False)

		if pretrained:
			model_dict = self.state_dict()
			pretrained_dict = load_state_dict_from_url(model_urls['inception_v3_google'], progress = False)
			# 1. filter out unnecessary keys
			pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'fc' not in k}
			# 2. overwrite entries in the existing state dict
			model_dict.update(pretrained_dict) 
			# 3. load the new state dict
			self.load_state_dict(model_dict)

		# Initialize new fc weights
		if init_weights:
			for m in self.modules():
				if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
					nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
				elif isinstance(m, nn.BatchNorm2d):
					nn.init.constant_(m.weight, 1)
					nn.init.constant_(m.bias, 0)
	
	def classifier_parameters(self):
		return self.fc.parameters()

	def higher_level_parameters(self):
		hl_parameters = []
		# hl_parameters += list(self.Mixed_7c.parameters())
		hl_parameters += list(self.Mixed_7b.parameters())
		hl_parameters += list(self.Mixed_7a.parameters())
		return hl_parameters

	def lower_level_parameters(self):
		ll_parameters = []
		ll_parameters += list(self.Mixed_7c.parameters())
		ll_parameters += list(self.Mixed_6e.parameters())
		ll_parameters += list(self.Mixed_6d.parameters())
		ll_parameters += list(self.Mixed_6c.parameters())
		ll_parameters += list(self.Mixed_6b.parameters())
		ll_parameters += list(self.Mixed_6a.parameters())   
		ll_parameters += list(self.Mixed_5d.parameters())
		ll_parameters += list(self.Mixed_5c.parameters())
		ll_parameters += list(self.Mixed_5b.parameters())     
		return ll_parameters
	
	def unfreeze_classifier(self):
		for param in self.classifier_parameters():
			param.requires_grad = True

	def unfreeze_higher_level(self):
		for param in self.higher_level_parameters():
			param.requires_grad = True

	def unfreeze_lower_level(self):
		for param in self.lower_level_parameters():
			param.requires_grad = True