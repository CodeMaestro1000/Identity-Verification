"""A set of functions to perform inference with the u-net model"""

import tensorflow as tf
import numpy as np
import logging

logger = logging.getLogger('dev')
logger.setLevel(logging.INFO)

#model = tf.keras.models.load_model('best_model.h5',  custom_objects={'BinaryMeanIoU': BinaryMeanIoU})

def predict_one(path, model, height=256, width=256):
    """Input:
    Path to an image, model, image height and width

    Output:
    A single prediction
    """
    img = tf.keras.utils.load_img(path, target_size=(height, width), color_mode="grayscale")
    img_array = tf.keras.utils.img_to_array(img)

    image_array = tf.expand_dims(img_array, 0)
    pred = model.predict(image_array)

    mask = pred[0]
    return mask

def predict_batch(filenames, model, height=256, width=256):
    """Inputs: 
    List of filenames, model, image height and width

    Output: 
    Batch of predictions
    """
    test_images = []
    for image in filenames:
        img = tf.keras.utils.load_img(image, target_size=(height, width), color_mode="grayscale")
        img_array = tf.keras.utils.img_to_array(img)
        test_images.append(img_array/255.0)
    test_images = np.array(test_images)
    predictions = model.predict(test_images)
    return predictions

def run_test(filenames, masknames, model, height=256, width=256):
    """
    Performs inference on a test set to get performance metrics
    Inputs:
    List of filenames and a corresponding list of ground truth masknames, model, height and width of images

    Outputs:
    Returns the mean IoU over the test set
    """
    test_images = []
    test_masks = []
    ious = []
    n_classes = 2
    IoU = tf.keras.metrics.MeanIoU(num_classes=n_classes)

    for image, mask in zip(filenames, masknames):
        img = tf.keras.utils.load_img(image, target_size=(height, width), color_mode="grayscale")
        img_array = tf.keras.utils.img_to_array(img)
        m = tf.keras.utils.load_img(mask, target_size=(height, width), color_mode="grayscale")
        mask_array = tf.keras.utils.img_to_array(m)
        test_images.append(img_array/255.0)
        test_masks.append(mask_array/255.0)
    test_images = np.array(test_images)
    test_masks = np.array(test_masks)
    predictions = model.predict(test_images)
    
    for i in range(len(predictions)):
        y_pred_thresholded = (predictions[i] > 0.5).astype('float32')
        IoU.update_state(test_masks[i], y_pred_thresholded)
        res = IoU.result().numpy()
        logger.info(f"Mean IoU = {res}")
        ious.append(res)
    
    ious = np.array(ious)
    return np.mean(ious)

