import cv2
import numpy as np
from random import randint
import os
import json

from tools.custom_errors import CorruptVideoError
from tools import preprocessing


def getFrame(video_handle):
	success_flag, frame = video_handle.read()
	if not success_flag:
		raise Exception("cv2.VideoCapture() returned (False, _)")
	return frame


def save_frames_from_video(filename):
	video_handle = cv2.VideoCapture(filename)
	assert video_handle.isOpened(), "Unable to open " + filename
	video_length = video_handle.get(7)
	image_path = filename.partition(".")[0] + "/"
	os.makedirs(image_path, exist_ok = True)

	for frame_nr in range(int(video_length)):
		frame = getFrame(video_handle)
		image_name = image_path + "frame_{0}.png".format(frame_nr)
		cv2.imwrite(image_name, frame)


def save_faces_from_video(filename, boxes):
	video_handle = cv2.VideoCapture(filename)
	assert video_handle.isOpened(), "Unable to open " + filename
	video_length = video_handle.get(7)
	image_path = filename.partition(".")[0] + "/"
	os.makedirs(image_path, exist_ok = True)
	boxes = json.load(open(boxes))

	for frame_nr in range(int(video_length)):
		top 	= boxes[str(frame_nr)]['0']['top']
		bottom 	= boxes[str(frame_nr)]['0']['bottom']
		left 	= boxes[str(frame_nr)]['0']['left']
		right 	= boxes[str(frame_nr)]['0']['right']
		frame = getFrame(video_handle)
		face = preprocessing.crop_image(frame, (top, bottom, left, right))
		image_name = image_path + "frame_{0}.png".format(frame_nr)
		cv2.imwrite(image_name, face)


def getRandomFrame(video_handle, is_color = True):
	video_length = video_handle.get(7)
	video_handle.set(1, start_frame)	#Set "CV_CAP_PROP_POS_FRAMES" to requested frame
	
	try:
		assert video_handle.get(7) >= 1, "Video doesn't have a single frame."
	except AssertionError:
		video_handle.release()
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


def yield_video_frames(video_handle, segment_length, is_color = True):
	video_length = video_handle.get(7)
	current_frame = 0

	while current_frame + segment_length <= video_length:
		frame_sequence = []
		for _ in range(int(segment_length)):
			try:
				frame = getFrame(video_handle)
			except Exception:
				if video_handle.get(1) + 1 <= video_length:
					raise CorruptVideoError("getFrame raised Exception() despite handled video having more frames")
				else:
					raise

			if not is_color:
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			frame_sequence.append(frame)
		frame_sequence = np.array(frame_sequence)
		current_frame += segment_length
		
		yield frame_sequence


def load_video_segment(video_handle, start_frame, segment_length, is_color = True):
	video_length = video_handle.get(7)
	try:
		assert start_frame + segment_length <= video_length, "Not enough frames after <start_frame> to return sequence of requested <segment_length>."
	except AssertionError:
		video_handle.release()
		raise

	video_handle.set(1, start_frame)	#Set "CV_CAP_PROP_POS_FRAMES" to requested frame
	frame_sequence = []
	for _ in range(int(segment_length)):
		try:
			frame = getFrame(video_handle)
		except Exception:
			if video_handle.get(1) + 1 <= video_length:
				raise CorruptVideoError("getFrame raised Exception() despite handled video having more frames")
			else:
				raise
				
		if not is_color:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		frame_sequence.append(frame)
	
	return np.array(frame_sequence)


def specific_frames(video_handle, frame_numbers, is_color = True):
	frames = []
	for frame_number in frame_numbers:
		video_handle.set(1, frame_number)		#Set "CV_CAP_PROP_POS_FRAMES" to requested frame
		try:
			frame = getFrame(video_handle)
		except Exception:
			if video_handle.get(1) + 1 <= video_handle.get(7):
				raise CorruptVideoError("getFrame raised Exception() despite handled video having more frames")
			else:
				raise
		if not is_color:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		frames.append(frame)
		
	return np.array(frames)


"""
Extracts faces from the requested video
	video_path 		- path to video to extract faces from
	bb_path    	  	- path to bounding boxes
	target_folder 	- where to dump the images
"""
def extract_faces_from_video(video_path, bb_path, target_folder):
	# Open the video & determine its length
	video_handle = cv2.VideoCapture(video_path)
	assert video_handle.isOpened(), "Unable to open " + video_name
	video_length = video_handle.get(7)

	# Get boundingbox information
	boxes = json.load(open(bb_path))
	# Create an empty file named multiple_faces in the video is tagged as such
	if boxes['multiple_faces']:
		open(os.path.join(target_folder, 'multiple_faces'), 'w+')
	
	# Main face extraction loop
	for frame_nr in range(int(video_length)):
		frame = getFrame(video_handle)

		for face_nr, box in boxes[str(frame_nr)].items():
			face_folder = os.path.join(target_folder, face_nr)
			os.makedirs(face_folder, exist_ok = True)
			box = (box['top'], box['bottom'], box['left'], box['right'])
			face = preprocessing.crop_image(frame, box)
			image_filename = os.path.join(face_folder, "{0}.png".format(frame_nr))
			cv2.imwrite(image_filename, face)