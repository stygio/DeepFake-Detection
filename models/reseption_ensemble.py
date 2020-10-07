import torch
import torch.nn as nn
import torch.nn.functional as F

from .reseption_v1 import Reseption1
from .reseption_v2 import Reseption2

reseption_weights = {
	'reseption_v1': 'models/saved_models/Reseption_Fixed/reseption_v1_ep47.pt',
	'reseption_v2': 'models/saved_models/Reseption_Fixed/reseption_v2_ep48.pt',
}

class Reseption_Ensemble(nn.Module):

	def __init__(self, init_weights = True):
		super(Reseption_Ensemble, self).__init__()

		self.Reseption_V1 = Reseption1(num_classes = 1, init_weights = False)
		self.Reseption_V1.load_state_dict(torch.load(reseption_weights['reseption_v1']))
		self.Reseption_V1.fc = nn.Linear(1280, 512)
		
		self.Reseption_V2 = Reseption2(num_classes = 1, init_weights = False)
		self.Reseption_V2.load_state_dict(torch.load(reseption_weights['reseption_v2']))
		self.Reseption_V2.fc = nn.Linear(1568, 512)

		self.fc = nn.Linear(1024, 1)
		
		if init_weights:
			for m in self.modules():
				if isinstance(m, nn.Linear):
					nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('leaky_relu', 0.2))

	def forward(self, x):
		r1_out = self.Reseption_V1(x)
		r2_out = self.Reseption_V2(x)
		x = torch.cat([r1_out, r2_out], 1)
		x = F.dropout(x, p = 0.5, training = self.training)
		x = self.fc(x)
		return x

	def higher_level_parameters(self):
		return []

	def unfreeze_higher_level(self):
		pass

	def lower_level_parameters(self):
		return []
	
	def unfreeze_lower_level(self):
		pass

	def classifier_parameters(self):
		cl_parameters = []
		cl_parameters += list(self.fc.parameters())
		cl_parameters += list(self.Reseption_V1.fc.parameters())
		cl_parameters += list(self.Reseption_V2.fc.parameters())
		return cl_parameters
	
	def unfreeze_classifier(self):
		for param in self.classifier_parameters():
			param.requires_grad = True

