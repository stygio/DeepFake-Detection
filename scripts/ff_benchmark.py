import torch
import os
import json
import cv2
from tqdm import tqdm

from tools import preprocessing
from models import transform
from models.model_helpers import get_model

benchmark_path = 'D:\\faceforensics_benchmark_images'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'reseption_v1'
model_path = 'models/saved_models/Reseption_Fixed/reseption_v1_ep47.pt'

model = get_model(model_name, model_path).to(device)
model.eval()

preprocessing.initialize_mobilenet()

def model_transform(data):
	data = transform.to_PIL(data)
	data = transform.model_transforms[model_name](data)
	return data

def predict_image(image):
	faces, _ = preprocessing.get_faces(image)
	face = faces[0]
	# preprocessing.show_test_img(face)
	logit = model(model_transform(face).unsqueeze(0).to(device))
	pred = torch.sigmoid(logit)
	pred = pred.detach().cpu().numpy()[0][0]
	return pred

predictions_dict = {}
for image_name in tqdm(os.listdir(benchmark_path), desc = 'FF++ Benchmark'):
	image_path = os.path.join(benchmark_path, image_name)
	image = cv2.imread(image_path)

	pred = predict_image(image)
	predictions_dict[image_name] = "fake" if pred < 0.5 else "real"

predictions_filename = model_name + '_submission.json'
predictions_json = open(predictions_filename, "w+")
json.dump(predictions_dict, predictions_json)
predictions_json.close()