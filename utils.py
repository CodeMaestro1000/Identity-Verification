"""A set of helper functions for performing image processing"""

import cv2, logging
import numpy as np
from seg_model.inference import predict_one

logging.getLogger('utils').setLevel(logging.INFO)

def order_points(pts):
	"""
  Inputs: a list of co-ordinates

  Orders the coordinates  such that the first entry in the list is the top-left,
	the second entry is the top-right, the third is the bottom-right, and the fourth is the bottom-left
  
  Returns the points in the order above
  """
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	"""
  Inputs: image (ndarray), a set of ordered points

  Transforms the image to give a birds-eye view perspective

  Output:
  The transformed image (ndarray)
  """
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped

def getROI(image_path, model, size=(1024, 1024), color=1):
  """
  Takes an image, performs segmentation and crops out the ROI from the image

  Inputs:
  Path to an image
  semantic segmentation model
  color flag - 1 (leave default colour configuration), otherwise change to RGB
  size - size of the output image

  Output:
  Cropped image (ndarray)
  """
  img = cv2.imread(image_path)
  img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
  if color != 1:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  ratio = img.shape[0]/256 

  mask = predict_one(image_path, model)
  mask = (mask * 255.0).astype('uint8')
  ret, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
  contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  if not contours:
    logging.info("E-01: ID Card not found in image")
    return
  if len(contours) > 1:
    largest_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
  else:
    largest_contour = contours[0]

  contour_perimeter = cv2.arcLength(largest_contour, True)
  perimeter_approx = cv2.approxPolyDP(largest_contour, 0.02 * contour_perimeter, True)
  if len(perimeter_approx) != 4:
    raise Exception("E-02: ID Card not found")
    return

  warped = four_point_transform(img, perimeter_approx.reshape(4, 2) * ratio)
  return warped