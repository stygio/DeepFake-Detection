# Defining some custom errors

class NoFaceDetected(Exception):
	pass

class MultipleFacesDetected(Exception):
	pass

class CorruptVideoError(Exception):
	pass
