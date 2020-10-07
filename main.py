import argparse
from os.path import isdir, isfile

from tools.network import Network

kaggle_path = "D:\\Kaggle_Dataset"
ff_path = "D:\\FaceForensics_Dataset"


if __name__ == '__main__':

	p = argparse.ArgumentParser(
		description = "DeepFake-Detection by Andrzej Putyra: Detecting facial manipulations in video.",
		formatter_class = argparse.ArgumentDefaultsHelpFormatter)

	p.add_argument('--mode', 		'-m', 	type = str, 
		choices = ['train', 'val', 'test', 'detect'], required = True)
	p.add_argument('--model_name', 	'-mn', 	type = str,	
		choices = [	'reseption_v1', 'reseption_v2', 'reseption_ensemble', 
					'mini_inception', 'xception', 'inception_v3', 'resnet152', 'resnext101', 'efficientnet-b5'], required = True)
	p.add_argument('--model_path', 	'-mp', 	type = str, 
		help = "path to saved model", 	default = None)
	p.add_argument('--dataset', 	'-d', 	type = str, 
		choices = ['kaggle', 'faceforensics'], default = 'faceforensics')

	args = p.parse_args()

	mode = args.mode
	model_name = args.model_name
	model_path = args.model_path
	dataset = args.dataset
	if not (model_path == None or isfile(model_path)):
		raise Exception("Invalid model path '{}'".format(model_path))
	dataset_path = ff_path if dataset == 'faceforensics' else kaggle_path

	if mode == 'train':
		net = Network(model_name = model_name, model_weights_path = model_path, pretrained = False)
		
		# dataset_path 	= str(	input("Dataset path (absolute path): "))
		# if not isdir(dataset_path):
		# 	raise Exception("Invalid dataset path '{}'".format(dataset_path)) 

		# epochs 			= int(	input("Epochs: "))
		# batch_size 		= int(	input("Batch size (even): "))

		training_level = 'full'
		training_type = 'various'
		optim = 'radam'

		net.train(dataset, dataset_path, epochs = 50, batch_size = 24, lr = 0.001, 
				training_level = training_level, training_type = training_type, 
				optimizer_choice = optim, gradient_scaling = False)


	elif mode == 'val' or mode == 'test':
		net = Network(model_name = model_name, model_weights_path = model_path)
		net.evaluate(dataset, dataset_path, mode, batch_size = 7)


	elif mode == 'detect':
		print("To be implemeneted in a future release.")

