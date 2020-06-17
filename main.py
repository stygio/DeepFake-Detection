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

kaggle_path = "D:\\Kaggle_Dataset"
ff_path = "D:\\FaceForensics_Dataset"


if __name__ == '__main__':

	p = argparse.ArgumentParser(
		description = "DeepFake-Detection by Andrzej Putyra: Detecting facial manipulations in video.",
		formatter_class = argparse.ArgumentDefaultsHelpFormatter)

	p.add_argument('--mode', 		'-m', 	type = str, 
		choices = ['train', 'val', 'test', 'detect'], required = True)
	p.add_argument('--model_name', 	'-mn', 	type = str,	
		choices = ['xception', 'inception_v3', 'resnet152', 'resnext101', 'efficientnet-b5'], required = True)
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
		net = Network(model_name = model_name, model_weights_path = model_path)
		
		# dataset_name 	= str(	input("Dataset name {kaggle, face_forensics}: "))
		# if dataset_name not in ['kaggle', 'face_forensics']:
		# 	raise Exception("Invalid dataset name '{}'".format(dataset_name))
		# dataset_path 	= str(	input("Dataset path (absolute path): "))
		# if not isdir(dataset_path):
		# 	raise Exception("Invalid dataset path '{}'".format(dataset_path)) 

		# epochs 			= int(	input("Epochs: "))
		# batch_size 		= int(	input("Batch size (even): "))

		training_level = 'lower'

		try:
			net.train(dataset, dataset_path, epochs = 10, batch_size = 24, lr = 0.0002, training_level = training_level)
		except KeyboardInterrupt:
			print("Execution ended by KeyboardInterrupt.")
			net.save_model(dataset + '_interrupted', training_level)


	elif mode == 'val' or mode == 'test':
		net = Network(model_name = model_name, model_weights_path = model_path)
		net.evaluate(dataset, dataset_path, mode, batch_size = 24)


	elif mode == 'detect':
		print("To be implemeneted in a future release.")

