from tools import preprocessing

kaggle_path = "D:/Kaggle_Dataset"
face_forensics_path = "D:/FaceForensics_Dataset"

# preprocessing.generate_boundingboxes_kaggle(kaggle_path)
preprocessing.generate_boundingboxes_ff(face_forensics_path)