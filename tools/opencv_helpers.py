import cv2
import numpy as np
from random import randint

frame_rate = 30
segment_length = 1 * frame_rate		# Segment length is the number of seconds


def getFrame(video_handle):
	success_flag, frame = video_handle.read()
	if not success_flag:
		raise Exception("cv2.VideoCapture() returned (False, _)")
	return frame


def saveFrameCollection(filename):
	video_handle = cv2.VideoCapture(filename)
	frame_nr = 0
	success_flag, image = video_handle.read()
	image_path = filename.partition(".")[0] + "/"
	frame_collection = np.asarray(image)
	while (frame_nr < 60*frame_rate) and success_flag:
		frame_nr += 1
		success_flag, image = video_handle.read()
		if frame_nr % segment_length == 0:
			image_name = image_path + "segment{0}.png".format(int((frame_nr + 0)/segment_length))
			print(image_name)
			print(np.shape(frame_collection))
			cv2.imwrite(image_name, frame_collection)
			frame_collection = np.asarray(image)
		else:
			frame_collection = np.concatenate((frame_collection, image), axis=0)
	video_handle.release()


def loadFrameSequence(video_handle, start_frame, sequence_length, is_color = True):
	video_length = video_handle.get(7)
	try:
		assert start_frame + sequence_length <= video_length, "Not enough frames after <start_frame> to return sequence of requested <sequence_length>."
	except AssertionError:
		video_handle.release()
		raise

	current_frame = start_frame
	video_handle.set(1, current_frame)	#Set "CV_CAP_PROP_POS_FRAMES" to requested frame
	frame_sequence = []
	for _ in range(sequence_length):
		frame = getFrame(video_handle)
		if not is_color:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		# frame = np.expand_dims(frame, axis=0)
		# frame_sequence = np.concatenate((frame_sequence, frame), axis=0)
		frame_sequence.append(frame)
	frame_sequence = np.array(frame_sequence)
	
	return frame_sequence


def yield_video_frames(video_handle, sequence_length, is_color = True):
	video_length = video_handle.get(7)
	current_frame = 0

	while current_frame + sequence_length <= video_length:
		frame_sequence = []
		for _ in range(sequence_length):
			frame = getFrame(video_handle)
			if not is_color:
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			frame_sequence.append(frame)
		frame_sequence = np.array(frame_sequence)
		current_frame += sequence_length
		
		yield frame_sequence


def getRandomFrame(video_handle, is_color = True):
	video_length = video_handle.get(7)
	try:
		assert video_handle.get(7) >= 1, "Video doesn't have a single frame."
	except AssertionError:
		video_handle.release()
		cv2.destroyAllWindows()
		raise

	random_frame = randint(0, video_length - 1)
	video_handle.set(1, random_frame)	#Set "CV_CAP_PROP_POS_FRAMES" to requested frame
	frame = getFrame(video_handle)
	if not is_color:
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	return frame


def videoFromFrameSequence(filename, frame_sequence, fps, is_color):
	FourCC = cv2.VideoWriter_fourcc(*'XVID')
	frame_size = (np.shape(frame_sequence)[2], np.shape(frame_sequence)[1])
	video_writer = cv2.VideoWriter(filename, FourCC, float(fps), frame_size, is_color)
	for frame in frame_sequence:
		video_writer.write(frame)
	video_writer.release()
	cv2.destroyAllWindows()


