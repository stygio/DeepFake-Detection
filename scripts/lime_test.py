import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np
import os, json

import torch
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F

from models.reseption_v1 import Reseption1
from models.reseption_ensemble import Reseption_Ensemble
from models.model_helpers import get_model

model_locations = {	'reseption_v1': 		'models/saved_models/Reseption_Fixed/reseption_v1_ep47.pt',
					'reseption_v2': 		'models/saved_models/Reseption_Fixed/reseption_v2_ep48.pt',
					'reseption_ensemble':	'models/saved_models/Reseption_Fixed/reseption_ensemble_ep19.pt',
					'inception_v3':			'models/saved_models/SotA_VariousRAdam/inception_v3_ep30.pt',
					'resnext101':			'models/saved_models/SotA_VariousRAdam/resnext101_ep30.pt'
					}

model_name = 'resnext101'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_dir = os.path.join('tests', 'test_images')

model = get_model(model_name, model_locations[model_name]).to(device)
model.eval()

def get_image(path):
	with open(os.path.abspath(path), 'rb') as f:
		with Image.open(f) as img:
			return img.convert('RGB')

image_paths = [(x, os.path.join(image_dir, x)) for x in os.listdir(image_dir)]
images = [(image_name, get_image(image_path)) for image_name, image_path in image_paths]

pill_img_transform = transforms.Compose([
		transforms.Resize((299, 299))
	])

tensor_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
	])

def batch_predict(images):
	batch = torch.stack(tuple(tensor_transform(image) for image in images), dim=0).to(device)
	logits = model(batch)
	probs = torch.sigmoid(logits)
	return probs.detach().cpu().numpy()

def predict_image(image_name, image):
	logit = model(tensor_transform(pill_img_transform(image)).unsqueeze(0).to(device))
	pred = torch.sigmoid(logit)
	pred = pred.detach().cpu().numpy()[0][0]
	if pred < 0.5:
		print('{}: Image predicted to be FAKE with probability of {:.2f}%'.format(image_name, (1-pred)*100))
	else:
		print('{}: Image predicted to be REAL with probability of {:.2f}%'.format(image_name, (pred)*100))


# Part which uses lime
from lime import lime_image
from skimage.segmentation import mark_boundaries

explainer = lime_image.LimeImageExplainer()

for image_name, image in images:
	predict_image(image_name, image)

	explanation = explainer.explain_instance(np.array(pill_img_transform(image)), 
											 batch_predict, # classification function
											 top_labels=1, 
											 hide_color=0, 
											 num_samples=2000) # number of images that will be sent to classification function

	temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
	img_boundry1 = mark_boundaries(temp/255.0, mask)
	fig, ax = plt.subplots()
	plt.imshow(img_boundry1)
	plt.axis('off')
	# plt.show()
	fig_name = model_name + '_' + image_name
	plt.savefig(os.path.join('tests', 'test_results', fig_name), bbox_inches='tight')
