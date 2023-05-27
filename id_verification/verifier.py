from typing import List
from keras.models import Model
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from sklearn.preprocessing import Normalizer
from scipy.spatial.distance import cosine
from utils import getROI

class IDVerifier:
  def __init__(self, ref_image_path: str, class_image_paths: List, segmentation_model: Model, detector: MTCNN, encoder: Model):
    self._reference_path = ref_image_path
    self._class_image_paths = class_image_paths
    self.face_detector = detector
    self.face_encoder = encoder
    self.segmentation_model = segmentation_model
    self._required_size = (160, 160)

  def normalize(self, img):
      mean, std = img.mean(), img.std()
      return (img - mean) / std

  def get_encoding(self, face):
      face = self.normalize(face)
      face = cv2.resize(face, self._required_size)
      encoding = self.face_encoder.predict(np.expand_dims(face, axis=0))[0]
      return encoding

  def get_face(self, image, image_path):
    roi = self.face_detector.detect_faces(image)
    if len(roi) < 1:
      raise Exception(f"No face found in image. Path: {image_path}")
    x1, y1, width, height = roi[0]['box']
    x1, y1 = abs(x1) , abs(y1)
    x2, y2 = x1+width , y1+height
    face = image[y1:y2 , x1:x2]
    return face 

  def get_reference_encodings(self):
    img = getROI(self._reference_path, self.segmentation_model)
    ref_face = self.get_face(img, self._reference_path)
    image_encoding = self.get_encoding(ref_face) 
    return image_encoding

  def encode_faces(self):
    l2_normalizer = Normalizer()
    face_encodings = []
    for path in self._class_image_paths:
      face_img = cv2.imread(path)
      face = self.get_face(face_img, path)
      encoding = self.get_encoding(face)
      face_encodings.append(encoding)
    
    face_encodings = np.sum(face_encodings, axis=0 )
    face_encodings = l2_normalizer.transform(np.expand_dims(face_encodings, axis=0))[0]
    return face_encodings
  
  def get_similarity_score(self):
    ref_encoding = self.get_reference_encodings()
    encodings = self.encode_faces()
    return 1 - cosine(ref_encoding, encodings)