import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Conv2D, Concatenate, BatchNormalization, \
    LeakyReLU, Conv2DTranspose, Activation, Reshape, Dense, Lambda, Permute, Add
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

from sensei import layers


def build_cloudFCN(inputs,batch_norm=True, num_channels=12, num_classes=1):

    # ------LOCAL INFORMATION GATHERING
    inputs = Conv2D(num_channels, (1, 1), strides=1, padding='SAME', activation='linear',
                kernel_initializer='glorot_uniform')(inputs)
    inputs = LeakyReLU(alpha=0.01)(inputs)


    inputs1 = Conv2D(num_channels, (1, 1), strides=1, padding='SAME', activation='linear',
                kernel_initializer='glorot_uniform')(inputs)
    inputs1 = LeakyReLU(alpha=0.01)(inputs1)
    inputs = Add()([inputs,inputs1])

    inputs1 = Conv2D(num_channels, (1, 1), strides=1, padding='SAME', activation='linear',
                kernel_initializer='glorot_uniform')(inputs)
    inputs1 = LeakyReLU(alpha=0.01)(inputs1)
    inputs = Add()([inputs,inputs1])

    inputs1 = Conv2D(num_channels, (1, 1), strides=1, padding='SAME', activation='linear',
                kernel_initializer='glorot_uniform')(inputs)
    inputs1 = LeakyReLU(alpha=0.01)(inputs1)
    inputs = Add()([inputs,inputs1])

    x0 = Conv2D(num_channels, (3, 3), strides=1, dilation_rate = 3, padding='SAME', activation='tanh',
                kernel_initializer='glorot_uniform')(inputs)

    x1 = Conv2D(num_channels, (3, 3), strides=1, dilation_rate = 6, padding='SAME', activation='tanh',
                kernel_initializer='glorot_uniform')(inputs)


    x_in = Concatenate()([inputs, x0, x1])
    x_in = Conv2D(num_channels, (1, 1), strides=1, padding='VALID', activation='tanh',
                     kernel_initializer='glorot_uniform')(x_in)
    if batch_norm:
        x_in = BatchNormalization(axis=-1, momentum=0.9)(x_in)

    x_RES_1 = Conv2D(8, (1, 1), strides=1, padding='VALID', activation='tanh',
                     kernel_initializer='glorot_uniform')(x_in)
    if batch_norm:
        x_RES_1 = BatchNormalization(axis=-1, momentum=0.9)(x_RES_1)

    # =================================

    x = Conv2D(64, (3, 3), strides=2, padding='SAME', activation='linear',
               kernel_initializer='glorot_uniform')(x_RES_1)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.9)(x)

    x = Conv2D(96, (3, 3), strides=2, padding='SAME', activation='linear',
               kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.9)(x)

    x_RES_2 = Conv2D(64, (1, 1), strides=1, padding='VALID', activation='linear',
                     kernel_initializer='glorot_uniform')(x)
    x_RES_2 = LeakyReLU(alpha=0.01)(x_RES_2)
    if batch_norm:
        x_RES_2 = BatchNormalization(
            axis=-1, momentum=0.9)(x_RES_2)

    x = Conv2D(192, (3, 3), strides=2, padding='SAME', activation='linear',
               kernel_initializer='glorot_uniform')(x_RES_2)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.9)(x)

    x = Conv2D(96, (1, 1), strides=1, padding='VALID', activation='linear',
               kernel_initializer='glorot_uniform')(x)
    x_RES_3 = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x_RES_3 = BatchNormalization(
            axis=-1, momentum=0.9)(x_RES_3)

    x = Conv2D(256, (3, 3), strides=2, padding='SAME', activation='linear',
               kernel_initializer='glorot_uniform')(x_RES_3)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.9)(x)

    # -----------------------CODE LAYER

    x = Conv2DTranspose(128, (3, 3), strides=2, padding='SAME', activation='linear',
                        kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.9)(x)

    x = Concatenate()([x, x_RES_3])
    x = Conv2D(128, (1, 1), strides=1, padding='VALID', activation='linear',
               kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.9)(x)

    x = Conv2DTranspose(128, (3, 3), strides=2, padding='SAME', activation='linear',
                        kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.9)(x)

    x = Concatenate()([x, x_RES_2])
    x = Conv2DTranspose(128, (3, 3), strides=2, padding='SAME', activation='linear',
                        kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.9)(x)

    x = Conv2D(64, (1, 1), strides=1, padding='VALID', activation='linear',
               kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.9)(x)

    x = Conv2D(48, (1, 1), strides=1, padding='VALID', activation='linear',
               kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.9)(x)

    x = Conv2DTranspose(48, (3, 3), strides=2, padding='SAME', activation='linear',
                        kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.9)(x)

    x = Concatenate()([x, x_RES_1, inputs])

    # =============FCL-type convolutions at each pixel...

    x = Conv2DTranspose(48, (1, 1), strides=1, padding='SAME', activation='linear',
                        kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.9)(x)

    x = Conv2DTranspose(32, (1, 1), strides=1, padding='SAME', activation='linear',
                        kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.9)(x)

    x = Conv2DTranspose(12, (1, 1), strides=1, padding='SAME', activation='linear',
                        kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.9)(x)

    x = Conv2DTranspose(4, (1, 1), strides=1, padding='SAME', activation='linear',
                        kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.9)(x)

    X0 = Conv2DTranspose(3, (3, 3), strides=1, padding='SAME', activation='linear',
                         kernel_initializer='glorot_uniform')(x)
    X0 = LeakyReLU(alpha=0.01)(X0)

    x = Concatenate()([x, X0, inputs])  # even more local info
    x = Conv2DTranspose(5, (1, 1), strides=1, padding='SAME', activation='linear',
                        kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)

    x = Conv2DTranspose(num_classes, (1, 1), strides=1, padding='SAME', activation='linear',
                        kernel_initializer='glorot_uniform')(x)
    if num_classes > 1:
        outputs = Activation('softmax')(x)
    # return Model(inputs=inputs, outputs=outputs)
    return outputs
