import thop
from torch import randn

from models.reseption_v1 import Reseption1
from models.reseption_v2 import Reseption2
from models.reseption_ensemble import Reseption_Ensemble
from models.inception_v3 import Binary_Inception
from models.resnext101 import Binary_ResNeXt101

models = {	'reseption_v1': Reseption1(),
			'reseption_v2': Reseption2(),
			'reseption_ensemble': Reseption_Ensemble(),
			'inception_v3': Binary_Inception(),
			'resnext101': Binary_ResNeXt101()
			}

input_tensor = randn(1, 3, 299, 299)


for model_name, model in models.items():
	macs, params = thop.profile(model, inputs = (input_tensor,))
	macs, params = thop.clever_format([macs, params], "%.3f")

	print('{} - MACs: {}, Params: {}'.format(model_name, macs, params))

