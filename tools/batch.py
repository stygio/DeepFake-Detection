import cv2
from PIL import Image
import torch
import time

from models import transform
from tools import preprocessing, opencv_helpers
import tools.miscellaneous as misc

class BatchGenerator:
	
	def __init__(self, model_type, device, batch_size):
		self.tensor_transform = transform.model_transforms[model_type]
		self.device = device
		self.batch_size = batch_size

	# Create a batch of face images from a point in the video
	def from_video_segment(self, video_path, start_frame = None):
		video_handle = cv2.VideoCapture(video_path)
		# Try..Except to handle the video_handle failure case
		try:
			assert video_handle.isOpened() == True, "VideoCapture() failed to open the video"
			video_length = video_handle.get(7)
			# If start_frame is not given choose random start_frame in the range of the video length in frames
			if start_frame == None:
				start_frame = random.randint(0, (video_length - 1) - self.batch_size)
			else:
				if not start_frame + self.batch_size <= video_length:
					raise IndexError("Requested segment of video is too long: last_frame {} > video length {}".format(
						start_frame + self.batch_size, video_length))

			# Grab a frame sequence
			start_time = time.time()
			frames = opencv_helpers.loadFrameSequence(video_handle, start_frame, sequence_length = self.batch_size)
			video_handle.release()
			print('DEBUG: <loadFrameSequence> elapsed time: {:.2f}'.format(time.time() - start_time))
			# Process the frames to retrieve only the faces, and construct the batch
			batch = []
			for frame in frames:
				# Retrieve detected faces and their positions. Throw an <AssertionError> in case of no detected faces.
				faces, face_positions = [], []
				try:
					# start_time = time.time()
					faces, face_positions = preprocessing.get_faces(frame)
					# print('DEBUG: <create_homogenous_batch> <preprocessing.get_faces> elapsed time: {:.2f}'.format(time.time() - start_time))
				except AssertionError:
					raise AttributeError("No faces detected in {}".format(video_path))
				# Check whether 1 face was detected. If more - throw a ValueError
				if len(face_positions) == 1:
					tensor_img = self.tensor_transform(Image.fromarray(faces[0]))
					batch.append(tensor_img)
				else:
					# ToDo: Multiple faces, choose closest one
					raise ValueError("Multiple faces detected in {}".format(video_path))
		except:
			# An error occured
			video_handle.release()
			raise

		# Stack list of tensors into a single tensor on device
		batch = torch.stack(batch).to(self.device)
		return batch


	# Create a batch of face images from a point in the video
	def from_frames(self, video_frames):
		# Process the frames to retrieve only the faces, and construct the batch
		batch = []
		for frame in video_frames:
			# Retrieve detected faces and their positions. Throw an <AssertionError> in case of no detected faces.
			faces, face_positions = [], []
			try:
				faces, face_positions = preprocessing.get_faces(frame)
			except AssertionError as Error:
				raise AttributeError("No faces detected.")
			# Check whether 1 face was detected. If more - throw a ValueError
			if len(face_positions) == 1:
				tensor_img = self.tensor_transform(Image.fromarray(faces[0]))
				batch.append(tensor_img)
			else:
				raise ValueError("Multiple faces detected.")

		# Stack list of tensors into a single tensor on device
		batch = torch.stack(batch).to(self.device)
		return batch


	# Create a batch of face images from various videos
	def from_various_videos(self, real_video_generator, fake_video_generator):
		# Process the frames to retrieve only the faces, and construct the batch
		batch = []
		labels = []
		while len(batch) < self.batch_size:
			video_path = None
			label = None
			if random.random() < 0.5:
				video_path = next(real_video_generator)
				label = 1
			else:
				video_path = next(fake_video_generator)
				label = 0

			# Grab a frame
			video_handle = cv2.VideoCapture(video_path)
			if (video_handle.isOpened() == True):
				# VideoCapture() succesfully opened the video
				video_length = video_handle.get(7)
				frame = opencv_helpers.getRandomFrame(video_handle)
				video_handle.release()
				cv2.destroyAllWindows()

				# Retrieve detected faces and their positions. Throw an <AssertionError> in case of no detected faces.
				faces, face_positions = [], []
				try:
					faces, face_positions = preprocessing.get_faces(frame)
				except AssertionError:
					print("No faces detected in {}".format(video_path))
				# Check whether 1 face was detected. If more - throw a ValueError
				if len(face_positions) == 1:
					tensor_img = self.tensor_transform(Image.fromarray(faces[0]))
					batch.append(tensor_img)
					labels.append(label)
				elif len(face_positions) >= 2:
					# ToDo: Multiple faces, choose closest one
					print("Multiple faces detected in {}".format(video_path))
					put_file_in_folder(file_path = video_path, folder = "multiple_faces")
			else:
				# VideoCapture() failed to open the video
				print("VideoCapture() failed to open {}".format(video_path))
				video_handle.release()
				cv2.destroyAllWindows()
				put_file_in_folder(file_path = video_path, folder = "bad_samples")

		# Stack list of tensors into a single tensor on device
		batch = torch.stack(batch).to(self.device)
		# Create label tensor
		labels = torch.tensor(labels, device = self.device, requires_grad = False, dtype = torch.float)
		labels = labels.view(-1,1)
		return batch, labels


	# """
	# Function to retrieve a homogenous batch in tensor form (FaceForensics dataset)
	# 	video_path_generator - generator object which returns paths to video samples
	# """
	# def get_ff_homogenous_batch(self, video_path_generator):
	# 	# While there is no batch, try to create one
	# 	batch, video_path = None, None
	# 	while not torch.is_tensor(batch):
	# 		try:
	# 			video_path = next(video_path_generator)
	# 			batch = preprocessing.create_homogenous_batch(video_path = video_path, 
	# 				model_type = model_type, device = device, batch_size = self.batch_size)
	# 		except AttributeError as Error:
	# 			# No faces error
	# 			print("DEBUG: {}".format(Error))
	# 		except ValueError as Error:
	# 			# Multiple faces error
	# 			print("DEBUG: {}".format(Error))
	# 			# Move the file to a special folder for videos with multiple faces
	# 			misc.put_file_in_folder(file_path = video_path, folder = "multiple_faces")
	# 		except (AssertionError, IndexError) as Error:
	# 			# Video access or video length error
	# 			print("DEBUG: {}".format(Error))
	# 			# Move the file to a special folder for corrupt/short videos
	# 			misc.put_file_in_folder(file_path = video_path, folder = "bad_samples")
	# 	return batch, video_path


	# """
	# Function to retrieve a disparate batch in tensor form (FaceForensics dataset)
	# 	real_video_generator - generator object which returns paths to real video samples
	# 	fake_video_generator - generator object which returns paths to fake video samples
	# """
	# def get_ff_disparate_batch(real_video_generator, fake_video_generator, model_type, device, batch_size):
	# 	batch, labels = preprocessing.create_disparate_batch(
	# 		real_video_generator = real_video_generator, fake_video_generator = fake_video_generator, model_type = model_type, device = device, batch_size = self.batch_size)
	# 	return batch, labels


	"""
	Function to retrieve a generator of batches in tensor form (Kaggle dataset)
	The batches contain sequences of consecutive frames from a single video
		video_path           - path to the video from which frames will be captured
	"""
	def single_video_batch(self, video_path):
		video_handle = cv2.VideoCapture(video_path)
		# Try..Except to handle the video_handle failure case
		try:
			assert video_handle.isOpened() == True, "VideoCapture() failed to open {}".format(video_path)
			frame_generator = opencv_helpers.yield_video_frames(video_handle, self.batch_size)

			# Iterate through the video yielding batches of <batch_size> frames
			error = False
			while not error:
				batch = None
				while not torch.is_tensor(batch):
					try:
						batch = self.from_frames(next(frame_generator))
						yield batch
					# Video is done (frame_generator_1 finished)
					except StopIteration:
						del frame_generator
						video_handle.release()
						error = True
						break
					except IndexError as Error:
						# Requested segment is invalid (goes out of bounds of video length)
						error = True
						break
					except AttributeError as Error:
						# No faces error
						print("DEBUG: {}".format(Error))
					except ValueError as Error:
						# Multiple faces error
						print("DEBUG: {}".format(Error))
						# Move the file to a special folder for videos with multiple faces
						misc.put_file_in_folder(file_path = video_path, folder = "multiple_faces")
						error = True
						break
					except AssertionError as Error:
						# Corrupt file error
						print("DEBUG: {}".format(Error))
						# Move the file to a special folder for corrupt videos
						misc.put_file_in_folder(file_path = video_path, folder = "bad_samples")
						error = True
						break
		
		except AssertionError:
			# Release the video file
			video_handle.release()
			# Move the file to a special folder for short/corrupt videos
			misc.put_file_in_folder(file_path = video_path, folder = "bad_samples")
			yield None


	"""
	Function to retrieve a generator of batches in tensor form (Kaggle dataset)
	The batches contain sequences of consecutive frames from a two videos (half of the batch from each one)
		video_path_1         - path to the 1st video from which frames will be captured
		video_path_2         - path to the 2nd video from which frames will be captured
	"""
	def dual_video_batch(self, video_path_1, video_path_2):
		# Open the two videos
		video_handle_1 = cv2.VideoCapture(video_path_1)
		video_handle_2 = cv2.VideoCapture(video_path_2)
		# Try..Except to handle the video_handle failure case
		try:
			# Check that the videos were opened successfully 
			assert video_handle_1.isOpened() == True, "VideoCapture() failed to open {}".format(video_path_1)
			assert video_handle_2.isOpened() == True, "VideoCapture() failed to open {}".format(video_path_2)
			# Get generators for sequences of frame from each video
			frame_generator_1 = opencv_helpers.yield_video_frames(video_handle_1, int(self.batch_size/2))
			frame_generator_2 = opencv_helpers.yield_video_frames(video_handle_2, int(self.batch_size/2))
			# Iterate through the video yielding batches of <batch_size> frames
			error = False
			while not error:
				batch = None
				while not torch.is_tensor(batch) and not error:
					batch1, batch2 = None, None
					while not torch.is_tensor(batch1):
						try:
							batch1 = self.from_frames(next(frame_generator_1))
						# Video is done (frame_generator_1 finished)
						except StopIteration:
							del frame_generator_1
							del frame_generator_2
							video_handle_1.release()
							video_handle_2.release()
							error = True
							break
						# No faces error
						except AttributeError as Error:
							print(">> DEBUG: {}".format(Error))
						# Multiple faces error
						except ValueError as Error:
							print(">> DEBUG: {}".format(Error))
							del frame_generator_1
							del frame_generator_2
							video_handle_1.release()
							video_handle_2.release()
							# Move the file to a special folder for videos with multiple faces
							misc.put_file_in_folder(file_path = video_path_1, folder = "multiple_faces")
							error = True
							break
					while not torch.is_tensor(batch2) and not error:
						try:
							batch2 = self.from_frames(next(frame_generator_2))
						# Video is done (frame_generator_2 finished)
						except StopIteration:
							del frame_generator_1
							del frame_generator_2
							video_handle_1.release()
							video_handle_2.release()
							error = True
							break
						# No faces error
						except AttributeError as Error:
							print(">> DEBUG: {}".format(Error))
						# Multiple faces error
						except ValueError as Error:
							print(">> DEBUG: {}".format(Error))
							del frame_generator_1
							del frame_generator_2
							video_handle_1.release()
							video_handle_2.release()
							# Move the file to a special folder for videos with multiple faces
							misc.put_file_in_folder(file_path = video_path_2, folder = "multiple_faces")
							error = True
							break
					if torch.is_tensor(batch1) and torch.is_tensor(batch2):
						batch = torch.cat((batch1, batch2), 0)
						yield batch
		
		except AssertionError:
			# Move the corrupt file to a special folder for short/corrupt videos
			if video_handle_1.isOpened() == False:
				video_handle_1.release()
				misc.put_file_in_folder(file_path = video_path_1, folder = "bad_samples")
			elif video_handle_2.isOpened() == False:
				video_handle_1.release()
				video_handle_2.release()
				misc.put_file_in_folder(file_path = video_path_2, folder = "bad_samples")
			yield None