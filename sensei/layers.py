from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer
import tensorflow as tf
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import array_ops
import numpy as np


class PermuteDescriptors(Layer):

    def __init__(self, **kwargs):
        super(PermuteDescriptors, self).__init__(**kwargs)

    def build(self, input_shape):
        super(PermuteDescriptors, self).build(input_shape)

    def call(self, x):
        spectral_pairing_1 = tf.tile(K.expand_dims(x,1),[1,K.shape(x)[1],1,1])
        spectral_pairing_2 = tf.tile(K.expand_dims(x,2),[1,1,K.shape(x)[1],1])
        return K.concatenate([spectral_pairing_1,spectral_pairing_2],axis=-1)

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        return tf.TensorShape([*input_shape[:2],input_shape[1],2*input_shape[-1]])

    def get_config(self):
        config = super(PermuteDescriptors, self).get_config()
        return config

class CombinePairedDescriptors(Layer):

    def __init__(self, output_channels,**kwargs):
        self.output_channels = output_channels
        super(CombinePairedDescriptors, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(1,1,input_shape.as_list()[-1],self.output_channels),
                                      initializer='random_uniform',
                                      trainable=True)
        super(CombinePairedDescriptors, self).build(input_shape)

    def call(self, x):
        return K.conv2d(x,self.kernel)

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        return tf.TensorShape([*input_shape[:-1],self.output_channels])

    def get_config(self):
        config = super(CombinePairedDescriptors, self).get_config()
        config.update({"output_channels": self.output_channels})
        return config

class PermuteBands(Layer):

    def __init__(self, **kwargs):
        super(PermuteBands, self).__init__(**kwargs)

    def build(self, input_shape):
        super(PermuteBands, self).build(input_shape)

    def call(self, x):
        band_pairing_1 = tf.tile(K.expand_dims(x,1),[1,K.shape(x)[1],1,1,1,1])
        band_pairing_2 = tf.tile(K.expand_dims(x,2),[1,1,K.shape(x)[1],1,1,1])
        return K.concatenate([band_pairing_1,band_pairing_2],axis=-1)

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        return tf.TensorShape([*input_shape[:2],input_shape[1],*input_shape[-3:-1],2*input_shape[-1]])

    def get_config(self):
        config = super(PermuteBands, self).get_config()
        return config


class PermuteDifferenceBands(Layer):

    def __init__(self, **kwargs):
        super(PermuteDifferenceBands, self).__init__(**kwargs)

    def build(self, input_shape):
        super(PermuteDifferenceBands, self).build(input_shape)

    def call(self, x):
        band_pairing_1 = tf.tile(K.expand_dims(x,1),[1,K.shape(x)[1],1,1,1,1])
        band_pairing_2 = tf.tile(K.expand_dims(x,2),[1,1,K.shape(x)[1],1,1,1])
        return band_pairing_1-band_pairing_2

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        return tf.TensorShape([*input_shape[:2],input_shape[1],*input_shape[-3:-1],input_shape[-1]])

    def get_config(self):
        config = super(PermuteDifferenceBands, self).get_config()
        return config


class Tile_bands_to_descriptor_count(Layer):

    def __init__(self,**kwargs):
        super(Tile_bands_to_descriptor_count, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Tile_bands_to_descriptor_count, self).build(input_shape)

    def call(self, x):
        bands = x[0]
        spectra = x[1]

        #ADD SPATIAL DIMS to spectra
        tilings = K.concatenate([tf.constant([1],dtype='int32'),[K.shape(spectra)[1]],tf.constant([1],dtype='int32'),tf.constant([1],dtype='int32'),tf.constant([1],dtype='int32')])
        tiled_bands = tf.tile(K.expand_dims(bands,1),tilings)

        return tiled_bands

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0][0],input_shape[1][1],*input_shape[0][1:]])

    def get_config(self):
        config = super(Tile_bands_to_descriptor_count, self).get_config()
        return config

class Concatenate_bands_with_descriptors(Layer):

    def __init__(self,rank=6,**kwargs):
        self.rank = rank
        super(Concatenate_bands_with_descriptors, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Concatenate_bands_with_descriptors, self).build(input_shape)

    def call(self, x):
        bands = x[0]
        spectra = x[1]

        #ADD SPATIAL DIMS to spectra
        spectra = K.expand_dims(spectra,axis=-2)
        spectra = K.expand_dims(spectra,axis=-2)
        tilings = K.concatenate([tf.constant([1]*(self.rank-3),dtype='int32'),K.shape(bands)[-3:-1],tf.constant([1],dtype='int32')])
        spectra = tf.tile(spectra,tilings)

        return K.concatenate((bands,spectra),axis=-1)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape[0][:-1]+[input_shape[0][-1]+input_shape[1][-1]])

    def get_config(self):
        config = super(Concatenate_bands_with_descriptors, self).get_config()
        config.update({"rank": self.rank})
        return config

class Matmul_bandcombination_with_spectralfeatures(Layer):

    def __init__(self,**kwargs):
        super(Matmul_bandcombination_with_spectralfeatures, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Matmul_bandcombination_with_spectralfeatures, self).build(input_shape)

    def call(self, x):
        bands = x[0]
        spectra = x[1]

        #ADD SPATIAL DIMS to spectra
        spectra = K.expand_dims(spectra,axis=3)
        spectra = K.expand_dims(spectra,axis=4)


        new_channels = tf.divide(K.shape(spectra)[-1:],K.shape(bands)[-1:])
        new_shape = K.concatenate((K.shape(spectra)[:-1],K.shape(bands)[-1:],K.cast(new_channels,'int32')))
        spectra = K.reshape(spectra,new_shape)
        outs = K.batch_dot(bands,spectra)
        return outs

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        return tf.TensorShape(input_shape[0][:-1] + [int(input_shape[1][-1]/input_shape[0][-1])])

    def get_config(self):
        config = super(Matmul_bandcombination_with_spectralfeatures, self).get_config()
        return config

class Flatten_2D_Op(Layer):
    def __init__(self,layer_2D,**kwargs):
        self.layer_2D = layer_2D
        super(Flatten_2D_Op,self).__init__(**kwargs)

    def build(self,input_shape):
        shape_4d = K.variable(np.concatenate((
                            [None],
                            input_shape[-3:]
                            )))
        super(Flatten_2D_Op,self).build(shape_4d)

    def call(self,x):
        shape_4d = K.concatenate((
                            tf.constant([-1],dtype='int32'),
                            tf.shape(x)[-3:]
                            ))
        inputs_4d = K.reshape(x,shape_4d)
        outputs_4d = self.layer_2D.__call__(inputs_4d)
        shape_Nd = K.concatenate((tf.shape(x)[:-3],tf.shape(outputs_4d)[-3:]))
        outputs_Nd = K.reshape(outputs_4d,shape_Nd)

        return outputs_Nd

    def compute_output_shape(self, input_shape):
        shape_4d = K.concatenate((
                            tf.constant([-1],dtype='int32'),
                            input_shape[-3:]
                            ))
        output_shape_4d = self.layer_2D.compute_output_shape(shape_4d)
        output_shape_Nd = K.concatenate((
                            input_shape[:-3],
                            output_shape_4d[-3:]
                            ))
        return output_shape_Nd

    def get_config(self):
        config = super(Flatten_2D_Op, self).get_config()
        config.update({"layer_2D": self.layer_2D})
        return config


if __name__ == '__main__':
    from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, MaxPooling2D
    from tensorflow.keras.models import Model
    import numpy as np

    inp2d = tf.ones(shape=(10,20,20,40))
    conv2d = Conv2D(16,(3,3),strides=2,name='Conv2D')

    out2d = conv2d(inp2d)
    print(out2d.shape)

    maxpool2d = MaxPooling2D()
    inp4d = tf.ones(shape=(10,5,7,20,20,40))
    convNd = Flatten_2D_Op(conv2d)
    maxpoolNd = Flatten_2D_Op(maxpool2d)
    out4d = maxpoolNd(inp4d)
    print(np.min(out4d),np.max(out4d))

    inps = Input(shape=(20,19,16))
    lay1 = Conv2D(20,(3,3),strides=2)(inps)
    lay2 = MaxPooling2D()(lay1)
    mod2d = Model(inputs=inps,outputs=lay2)
    modNd = Flatten_2D_Op(mod2d)

    out2d = mod2d(tf.ones((10,20,19,16)))
    outNd = modNd(tf.ones((10,5,8,20,19,16)))

    print(out2d.shape)
    print(outNd.shape)
