import argparse
from os.path import isdir, isfile

from tools.network import Network

# real_vid_dirs = [	"D:\\FaceForensics_Dataset\\original_sequences\\c23\\videos"]
# fake_vid_dirs = [	"D:\\FaceForensics_Dataset\\manipulated_sequences\\DeepFakeDetection\\c23\\videos",
# 					"D:\\FaceForensics_Dataset\\manipulated_sequences\\Deepfakes\\c23\\videos",
# 					"D:\\FaceForensics_Dataset\\manipulated_sequences\\Face2Face\\c23\\videos",
# 					"D:\\FaceForensics_Dataset\\manipulated_sequences\\FaceSwap\\c23\\videos",
# 					"D:\\FaceForensics_Dataset\\manipulated_sequences\\NeuralTextures\\c23\\videos",
# ]

# one_face_vp = "D:\\FaceForensics_Dataset\\original_sequences\\c23\\videos\\000.mp4"
# no_face_vp = "C:\\Users\\Andrzej\\Videos\\MazeEscape\\Maze1.mp4"
# two_face_vp = "D:\\FaceForensics_Dataset\\original_sequences\\c23\\videos\\01__walking_and_outside_surprised.mp4"

# kaggle_path = "D:\\Kaggle_Dataset"


if __name__ == '__main__':

	p = argparse.ArgumentParser(
		description = "DeepFake-Detection by Andrzej Putyra: Detecting facial manipulations in video.",
		formatter_class = argparse.ArgumentDefaultsHelpFormatter)

	p.add_argument('--mode', 		'-m', 	type = str, 
		choices = ['train', 'val', 'test', 'detect'], required = True)
	p.add_argument('--model_name', 	'-mn', 	type = str,	
		choices = ['xception', 'inception_v3', 'resnet152', 'resnext101'])
	p.add_argument('--model_path', 	'-mp', 	type = str, 
		help = "path to saved model", 	default = None)

	args = p.parse_args()

	mode = args.mode
	model_name = args.model_name
	model_path = args.model_path
	if not (model_path == None or isfile(model_path)):
			raise Exception("Invalid model path '{}'".format(model_path))


	if mode == 'train':
		dataset_name 	= str(	input("Dataset name {kaggle, face_forensics}: "))
		if dataset_name not in ['kaggle', 'face_forensics']:
			raise Exception("Invalid dataset name '{}'".format(dataset_name))
		dataset_path 	= str(	input("Dataset path (absolute path): "))
		if not isdir(dataset_path):
			raise Exception("Invalid dataset path '{}'".format(dataset_path)) 

		start_folder 	= int(	input("Start from folder: ")) if model_path else None
		epochs 			= int(	input("Epochs: "))
		batch_size 		= int(	input("Batch size (even): "))
		only_fc_layer	= str(	input("Only FC layer {True, False}: "))
		if only_fc_layer not in ['True', 'False']:
			raise Exception("Invalid choice for only_fc_layer '{}'".format(only_fc_layer))
		only_fc_layer = True if only_fc_layer == 'True' else False

		net = Network(model_name = model_name, model_weights_path = model_path)
		net.train_kaggle(dataset_path, epochs, batch_size, only_fc_layer = only_fc_layer, start_folder = start_folder)
	

	elif mode == 'val' or mode == 'test':
		net = Network(model_name = model_name, model_weights_path = model_path)
		net.evaluate_kaggle("D:/Kaggle_Dataset", mode = mode, batch_size = 50)	

	elif mode == 'detect':
		print("To be implemeneted in a future release.")

