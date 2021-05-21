import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import MeanIoU


class OneHotMeanIoU(MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)


def w_categorical_crossentropy(y_true, y_pred, weights):
    """
    Keras-style categorical crossentropy loss function, with weighting for each class.
    """
    y_true_max = K.argmax(y_true, axis=-1)
    weighted_true = K.gather(weights, y_true_max)
    loss = K.categorical_crossentropy(y_pred, y_true) * weighted_true
    return loss
