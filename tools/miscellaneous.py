import os
import random

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


# Return a generator of shuffled folders in <directory>
def get_random_directory(directory):
	dir_list = os.listdir(directory)
	random.shuffle(dir_list)
	for name in dir_list:
		print("Chosen directory: {}".format(name))
		yield os.path.join(directory, name)