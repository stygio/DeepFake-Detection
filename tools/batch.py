import cv2
from PIL import Image
import torch
import time

from models import transform
from tools import preprocessing, opencv_helpers
from tools.custom_errors import NoFaceDetected, MultipleFacesDetected, CorruptVideoError
import tools.miscellaneous as misc

class BatchGenerator:
	
	def __init__(self, model_type, device, batch_size):
		self.training_transform = transform.data_augmentation
		self.tensor_transform = transform.model_transforms[model_type]
		self.device = device
		self.batch_size = batch_size
		# Initializing mobilenet for face recognition
		preprocessing.initialize_mobilenet(0.4)

	# Create a batch of face images from a point in the video
	def from_video_segment(self, video_path, start_frame = None):
		video_handle = cv2.VideoCapture(video_path)
		# Try..Except to handle the video_handle failure case
		if video_handle.isOpened() == True:
			video_length = video_handle.get(7)
			# If start_frame is not given choose random start_frame in the range of the video length in frames
			if start_frame == None:
				start_frame = random.randint(0, (video_length - 1) - self.batch_size)
			else:
				if not start_frame + self.batch_size <= video_length:
					raise IndexError("Requested segment of video is too long: last_frame {} > video length {}".format(
						start_frame + self.batch_size, video_length))

			# Grab a frame sequence
			frames = opencv_helpers.loadFrameSequence(video_handle, start_frame, sequence_length = self.batch_size)
			video_handle.release()
			# Process the frames to retrieve only the faces, and construct the batch
			batch = []
			for frame in frames:
				# Retrieve detected faces and their positions. Throw an <AssertionError> in case of no detected faces.
				faces, face_positions = preprocessing.get_faces(frame)
				if len(face_positions) == 0:
					raise NoFaceDetected("No faces detected in {}".format(video_path))
				elif len(face_positions) == 1:
					tensor_img = self.tensor_transform(Image.fromarray(faces[0]))
					batch.append(tensor_img)
				else:
					raise MultipleFacesDetected("Multiple faces detected in {}".format(video_path))
		else:
			video_handle.release()
			# put_file_in_folder(file_path = video_path, folder = "bad_samples")
			raise CorruptVideoError("VideoCapture() failed to open {}".format(video_path))

		# Stack list of tensors into a single tensor on device
		batch = torch.stack(batch).to(self.device)
		return batch


	# Create a batch of face images from a point in the video
	def from_frames(self, video_frames):
		# Process the frames to retrieve only the faces, and construct the batch
		batch = []
		for frame in video_frames:
			# Retrieve detected faces and their positions.
			faces, face_positions = preprocessing.get_faces(frame)
			if len(face_positions) == 0:
					raise NoFaceDetected("No faces detected.")
			elif len(face_positions) == 1:
				tensor_img = self.tensor_transform(Image.fromarray(faces[0]))
				batch.append(tensor_img)
			else:
				raise MultipleFacesDetected("Multiple faces detected.")

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

				# Retrieve detected faces and their positions.
				faces, face_positions = preprocessing.get_faces(frame)
				if len(face_positions) == 0:
					raise NoFaceDetected("No faces detected in {}".format(video_path))
				elif len(face_positions) == 1:
					tensor_img = self.tensor_transform(Image.fromarray(faces[0]))
					batch.append(tensor_img)
					labels.append(label)
				else:
					print("Multiple faces detected in {}".format(video_path))
					put_file_in_folder(file_path = video_path, folder = "multiple_faces")
			else:
				print("VideoCapture() failed to open {}".format(video_path))
				video_handle.release()
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
		# Open the video
		video_handle = cv2.VideoCapture(video_path)
		try:
			# Check that video was opened correctly
			if video_handle.isOpened() == False:
				raise CorruptVideoError("cv2.VideoCapture() failed to open {}".format(video_path))

			# Generator for consecutive sequences of <batch_size> frames
			frame_generator = opencv_helpers.yield_video_frames(video_handle, int(self.batch_size))
			# Iterate through the video yielding batches of <batch_size> frames
			error = False
			while not error:
				batch = None
				while not torch.is_tensor(batch) and not error:
					try:
						batch = self.from_frames(next(frame_generator))
					# Video is done (frame_generator finished)
					except StopIteration:
						del frame_generator
						video_handle.release()
						error = True
						break
					except NoFaceDetected as Error:
						print("DEBUG: {}".format(Error))
					except MultipleFacesDetected as Error:
						print("DEBUG: {}".format(Error))
						video_handle.release()
						# Move the file to a special folder for videos with multiple faces
						misc.put_file_in_folder(file_path = video_path, folder = "multiple_faces")
						error = True
						break
				if torch.is_tensor(batch):
					yield batch
		
		except CorruptVideoError as Error:
			# print(">> DEBUG: {}".format(Error))
			# Release the video file
			video_handle.release()
			# Move the file to a folder for corrupt videos
			misc.put_file_in_folder(file_path = video_path, folder = "bad_samples")


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
		v1_err, v2_err = False, False
		try:
			# Check that the videos were opened successfully
			if video_handle_1.isOpened() == False:
				v1_err = True
				raise CorruptVideoError("cv2.VideoCapture() failed to open {}".format(video_path_1))
			if video_handle_2.isOpened() == False:
				v2_err = True
				raise CorruptVideoError("cv2.VideoCapture() failed to open {}".format(video_path_2))

			# Get generators for consecutive sequences of frames from each video
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
							video_handle_1.release()
							video_handle_2.release()
							error = True
							break
						except NoFaceDetected as Error:
							print(">> DEBUG: {}".format(Error))
						except MultipleFacesDetected as Error:
							print(">> DEBUG: {}".format(Error))
							video_handle_1.release()
							video_handle_2.release()
							# Move the file to a special folder for videos with multiple faces
							misc.put_file_in_folder(file_path = video_path_1, folder = "multiple_faces")
							error = True
							break
						except CorruptVideoError as Error:
							v1_err = True
							raise
					while not torch.is_tensor(batch2) and not error:
						try:
							batch2 = self.from_frames(next(frame_generator_2))
						# Video is done (frame_generator_2 finished)
						except StopIteration:
							video_handle_1.release()
							video_handle_2.release()
							error = True
							break
						# No faces error
						except NoFaceDetected as Error:
							print(">> DEBUG: {}".format(Error))
						# Multiple faces error
						except MultipleFacesDetected as Error:
							print(">> DEBUG: {}".format(Error))
							video_handle_1.release()
							video_handle_2.release()
							# Move the file to a special folder for videos with multiple faces
							misc.put_file_in_folder(file_path = video_path_2, folder = "multiple_faces")
							error = True
							break
						except CorruptVideoError as Error:
							v2_err = True
							raise
					if torch.is_tensor(batch1) and torch.is_tensor(batch2):
						batch = torch.cat((batch1, batch2), 0)
						yield batch
		
		except CorruptVideoError:
			# print(">> DEBUG: {}".format(Error))
			# Release video handles
			video_handle_1.release()
			video_handle_2.release()
			# Move the file to a folder for corrupt videos
			if v1_err:
				misc.put_file_in_folder(file_path = video_path_1, folder = "bad_samples")
			if v2_err:
				misc.put_file_in_folder(file_path = video_path_2, folder = "bad_samples")


	"""
	Function to retrieve a generator of training batches in tensor form (Kaggle dataset)
	The batches contain sequences of consecutive frames from a two videos (half of the batch from each one)
		fake_video_path	- path to the altered video
		real_video_path	- path to the original video the fake is based on
	"""
	def training_batch(self, fake_video_path, real_video_path, boxes, epoch, n = 10):
		frame_numbers = [epoch]
		for _ in range(int(self.batch_size/2)):
			frame_numbers.append(frame_numbers[-1] + n)
		# Open the two videos
		fake_video_handle = cv2.VideoCapture(fake_video_path)
		fake_video_length = fake_video_handle.get(7)
		assert frame_numbers[-1] < fake_video_length, "Fake video length too short for requested parameters."
		real_video_handle = cv2.VideoCapture(real_video_path)
		real_video_length = real_video_handle.get(7)
		assert frame_numbers[-1] < real_video_length, "Real video length too short for requested parameters."
		fake_err, real_err = False, False
		try:
			# Check that the videos were opened successfully
			if fake_video_handle.isOpened() == False:
				fake_err = True
				raise CorruptVideoError("cv2.VideoCapture() failed to open {}".format(fake_video_path))
			if real_video_handle.isOpened() == False:
				real_err = True
				raise CorruptVideoError("cv2.VideoCapture() failed to open {}".format(real_video_path))

			# Frames from the fake video
			fake_frames = opencv_helpers.specific_frames(fake_video_handle, frame_numbers)
			fake_video_handle.release()
			# Frames from the original video
			real_frames = opencv_helpers.specific_frames(real_video_handle, frame_numbers)
			real_video_handle.release()

			fake_faces, real_faces = [], []
			for i in range(int(self.batch_size/2)):
				top 	= boxes[str(frame_numbers[i])]['0']['top']
				bottom 	= boxes[str(frame_numbers[i])]['0']['bottom']
				left 	= boxes[str(frame_numbers[i])]['0']['left']
				right 	= boxes[str(frame_numbers[i])]['0']['right']
				fake_face = preprocessing.crop_image(fake_frames[i], (top, bottom, left, right))
				real_face = preprocessing.crop_image(real_frames[i], (top, bottom, left, right))
				fake_faces.append(self.tensor_transform(self.training_transform(Image.fromarray(fake_face))))
				real_faces.append(self.tensor_transform(self.training_transform(Image.fromarray(real_face))))

			batch = fake_faces + real_faces
			batch = torch.stack(batch).to(self.device)
			return batch
		
		except CorruptVideoError:
			# print(">> DEBUG: {}".format(Error))
			# Release video handles
			fake_video_handle.release()
			real_video_handle.release()
			# Move the file to a folder for corrupt videos
			if fake_err:
				misc.put_file_in_folder(file_path = fake_video_path, folder = "bad_samples")
			if real_err:
				misc.put_file_in_folder(file_path = real_video_path, folder = "bad_samples")


	"""
	Function to retrieve a generator of batches in tensor form (Kaggle dataset)
	The batches contain sequences of consecutive frames from a two videos (half of the batch from each one)
		video_path - path to the video
		boxes	   - dictionary of bounding boxes in the video	
	"""
	def evaluation_batch(self, video_path, boxes):
		# Open the video
		video_handle = cv2.VideoCapture(video_path)
		try:
			# Check that video was opened correctly
			if video_handle.isOpened() == False:
				raise CorruptVideoError("cv2.VideoCapture() failed to open {}".format(video_path))

			# Generator for consecutive sequences of <batch_size> frames
			frame_generator = opencv_helpers.yield_video_frames(video_handle, int(self.batch_size))
			# Iterate through the video yielding batches of <batch_size> frames
			error = False
			start_frame = 0
			while not error:
				try:
					batch = next(frame_generator)
					start_frame += int(self.batch_size)

					faces = []
					for i in range(int(self.batch_size)):
						top 	= boxes[str(start_frame + i)]['0']['top']
						bottom 	= boxes[str(start_frame + i)]['0']['bottom']
						left 	= boxes[str(start_frame + i)]['0']['left']
						right 	= boxes[str(start_frame + i)]['0']['right']
						face = preprocessing.crop_image(batch[i], (top, bottom, left, right))
						faces.append(self.tensor_transform(Image.fromarray(face)))

					batch = torch.stack(faces).to(self.device)
					yield batch

				# Video is done (frame_generator finished)
				except StopIteration:
					del frame_generator
					video_handle.release()
					error = True
					break
		
		except CorruptVideoError as Error:
			# Release the video file
			video_handle.release()
			# Move the file to a folder for corrupt videos
			misc.put_file_in_folder(file_path = video_path, folder = "bad_samples")