import os
import random
from datetime import datetime
from torch import save

# # Return a generator of random real/fake image directories
# def get_random_directory(real_or_fake):
# 	if real_or_fake == "real":
# 		dir_list = os.listdir(real_img_dir)
# 		# random.shuffle(dir_list)
# 		for name in dir_list:
# 			yield os.path.join(real_img_dir, name)
# 	elif real_or_fake == "fake":
# 		dir_list = os.listdir(fake_img_dir)
# 		# random.shuffle(dir_list)
# 		for name in dir_list:
# 			yield os.path.join(fake_img_dir, name)
# 	else:
# 		raise Exception("Invalid value passed with <real_or_fake>")


# Return a string with current timestamp
def timestamp():
	now = datetime.now()
	timestamp_string = now.strftime("_date$%m%d%Y_time$%H%M%S")
	return timestamp_string


# Return an infinite generator of shuffled folders in <directory>
def get_random_file_path(directory):
	dir_list = os.listdir(directory)
	while True:
		random.shuffle(dir_list)
		for name in dir_list:
			chosen_file = os.path.join(directory, name)
			if os.path.isfile(chosen_file):
				yield chosen_file

"""
Put a file in a folder in its current directory, 
	file_path - current path to the file
	folder    - target folder for the file, which is created if it doesn't exist yet
"""
def put_file_in_folder(file_path, folder):
	split_path = os.path.split(file_path)
	new_path = os.path.join(split_path[0], folder, split_path[1])
	os.makedirs(os.path.dirname(new_path), exist_ok=True)
	try:
		print("DEBUG: Moving '{}' to '{}'".format(split_path[1], "{}/{}".format(folder, split_path[1])))
		os.replace(file_path, new_path)
	except FileNotFoundError:
		raise FileNotFoundError("File specified by the path {} doesn't exist.".format(file_path))


"""
Save a PyTorch network to a file in models/saved_models/
	network_state_dict - state dict of the network to be saved
	model_type         - model type name
"""
def save_network(network_state_dict, model_type):
	model_dir = "models/saved_models/"
	filename = model_type + timestamp() + ".pt"
	filename = os.path.join(model_dir, filename)
	print("Saving network as '{}'".format(filename))
	save(network_state_dict, filename)


"""
Creates a log file and returns the path to it
	model_type    - model type name
	header_string - string which will be at the top of the log file
"""
def create_log(model_type, header_string):
	log_dir = "outputs/logs/"
	filename = model_type + timestamp() + ".txt"
	filename = os.path.join(log_dir, filename)
	f = open(filename, "w+")
	f.write(header_string)
	f.close()

	return filename


"""
Adds a string to the specified log
	log_file   - path to log file
	log_string - string to be appended to the log
"""
def add_to_log(log_file, log_string):
	f = open(log_file, "a")
	f.write(log_string)
	f.close()
