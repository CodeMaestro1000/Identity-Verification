"""
Implementation of custom metric for model evaluation
"""
import tensorflow as tf


class BinaryMeanIoU(tf.keras.metrics.MeanIoU):
    """
    Compute the MeanIoU for a binary segmentation task
    y_pred returns a value between 0 and 1, threshold is used to clip predictions to either 0 or 1
    then the mean IoU is computed between the new_y_pred and y_true

    see: https://www.tensorflow.org/api_docs/python/tf/keras/metrics/MeanIoU for more information on Mean IoU
    """
    def __init__(self, num_classes=2, threshold=0.5, name="mean_iou",**kwargs):
      super().__init__(num_classes, name=name, **kwargs)
      self.num_classes=num_classes
      self.threshold = threshold

    def update_state(self, y_true, y_pred, sample_weight=None):
      new_y_pred = tf.cast(y_pred > self.threshold, tf.int32)
      return super().update_state(y_true, new_y_pred, sample_weight)

    def get_config(self):
      """This function saves threshold and num_classses as part of the saved model"""
      base_config = super().get_config()
      base_config.update({"threshold": self.threshold, "num_classes": self.num_classes})
      return base_config