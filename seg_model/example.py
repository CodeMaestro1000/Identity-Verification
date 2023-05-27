"""
Example code showing a complete pipeline usage including the semantic segmentation model
"""
from metrics import BinaryMeanIoU
import tensorflow as tf
import cv2, argparse, logging
from utils import getROI

logging.getLogger().setLevel(logging.INFO)

logging.info("Example code using semantic segmentation to help crop an ID card from an input image.")

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True, help="path to input image to be segmented")
parser.add_argument("-o", "--output", required=True, help="output filepath to store the segmented image")
args = vars(parser.parse_args())

logging.info("Loading model...")
model = tf.keras.models.load_model('best_model.h5',  custom_objects={'BinaryMeanIoU': BinaryMeanIoU})
logging.info("Done.")

logging.info("Processing image....")
cropped_img = getROI(args['image'], model)
cv2.imwrite(args['output'], cropped_img)
logging.info(f"Done. \nOuput file saved to {args['output']}")
