# Unit tests for preprocessing functions

import sys, os
sys.path.append(os.path.dirname(sys.path[0]))
import unittest
import torch

from tools import preprocessing


class TestPreprocessing(unittest.TestCase):

	def setUp(self):
		# Relative paths to test videos
		self.one_face_video = "media_resources/one_face.mp4"
		self.no_face_video = "media_resources/no_face.mp4"
		self.multiple_face_video = "media_resources/multiple_faces.mp4"
		# Setting pytorch device
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# def tearDown(self):
	# 	print('tearDown')

	# Check	whether preprocessing.create_batch() creates a valid <torch.tensor> object for a video with one detectable face
	def test_create_batch_oneFace(self):
		self.assertIsInstance(preprocessing.create_batch(self.one_face_video, self.device), type(torch.tensor([0.0])))

	# Check	whether preprocessing.create_batch() raises an <AttributeError> for a video with no detectable faces
	def test_create_batch_noFace(self):
		self.assertRaises(AttributeError, preprocessing.create_batch, self.no_face_video, self.device)
	
	# Check	whether preprocessing.create_batch() raises a <ValueError> for a video with multiple detectable faces 
	def test_create_batch_multipleFaces(self):
		self.assertRaises(ValueError, preprocessing.create_batch, self.multiple_face_video, self.device)
		
		
if __name__ == '__main__':
	unittest.main()