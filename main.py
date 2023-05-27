"""
Example code showing a complete pipeline usage including the semantic segmentation model
"""
import time
import tensorflow as tf
import argparse, logging, mtcnn
from seg_model.metrics import BinaryMeanIoU
from id_verification.model import InceptionResNetV2
from id_verification.verifier import IDVerifier


logging.getLogger().setLevel(logging.INFO)

start = time.time()
logging.info("Entrypoint for ID Verification Code...")

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--ref-image", required=True, help="path to the input identity document to be verified")
parser.add_argument("-c", "--class-images", required=True, nargs='+', help="path to images collected from the supposed owner of the ID")
parser.add_argument("-t", "--threshold", required=False, help="threshold value for an identity document to be considered verified", default=0.5)
args = vars(parser.parse_args())

logging.info("Loading U-Net model...")
seg_model = tf.keras.models.load_model('seg_model/best_model.h5',  custom_objects={'BinaryMeanIoU': BinaryMeanIoU})
logging.info("Done.")

logging.info("Loading Facenet model...")
face_encoder = InceptionResNetV2()
path = "id_verification/facenet_keras.h5"
face_encoder.load_weights(path)
face_detector = mtcnn.MTCNN()
logging.info("Done.")

logging.info("Processing....")
verifier = IDVerifier(args["ref_image"], args["class_images"], seg_model, face_detector, face_encoder)
similarity_score = verifier.get_similarity_score()
if similarity_score < float(args["threshold"]):
    logging.info(f"Verification failed...\nScore: {similarity_score}, Threshold: {args['threshold']}")
else:
    logging.info(f"Verification Successful...\nScore: {similarity_score}, Threshold: {args['threshold']}")

end = time.time()
logging.info(f"Done. Execution time: {end - start} seconds")
