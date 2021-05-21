from tensorflow.keras.layers import Input, Conv2D, Concatenate, BatchNormalization, \
    LeakyReLU, Reshape, Dense, Lambda, Permute, Multiply, Add, Activation, Flatten
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import tensorflow as tf
import tensorflow_addons as tfa
import os

from sensei import layers


def permuted_descriptor_block(descriptors,config):
    """
    Combines each possible paired permutation of N descriptors, puts each through
    a neural network, as defined in config. Then, sums the pairs and returns
    N feature vectors.
    """
    block_name = config['block_name']
    layer_sizes = config['layer_sizes']
    final_size = config['final_size']
    skips = config['skips']
    lasso_regularization = config['lasso_regularization']
    concat_input_output = config['concat_input_output']

    pairs = layers.PermuteDescriptors(name='PermuteDescriptors')(descriptors)
    skip_pairs = pairs # prepare in case skip connections used
    if layer_sizes is not None:
        if isinstance(layer_sizes,int):
            layer_sizes = [layer_sizes]
        for i,n in enumerate(layer_sizes):
            pairs = layers.Flatten_2D_Op(Conv2D(n,(1,1),kernel_initializer='glorot_uniform',strides=(1,1)),name=block_name+'/Conv-{}'.format(i))(pairs)
            pairs = LeakyReLU(alpha=0.01,name=block_name+'/LeakyReLU-{}'.format(i))(pairs)
            pairs = tfa.layers.GroupNormalization(groups=2, axis=-1,name=block_name+'/GroupNorm-{}'.format(i))(pairs)
            # pairs = BatchNormalization(axis=-1,momentum=0.99,name=block_name+'/BatchNorm-{}'.format(i))(pairs)
            if skips=='add' and pairs.shape[-1]==skip_pairs.shape[-1]:
                pairs = Add(name=block_name+'/SkipAdd-{}'.format(i))([pairs,skip_pairs])
            elif skips=='concat':
                pairs = Concatenate(axis=-1,name=block_name+'/SkipConcat-{}'.format(i))([pairs,skip_pairs])
            skip_pairs = pairs


    if lasso_regularization:
        pairs = layers.Flatten_2D_Op(Conv2D(final_size,(1,1),
                                kernel_initializer='glorot_uniform',strides=(1,1),
                                kernel_regularizer=regularizers.l1(1e-5)),
                                name=block_name+'/Conv-last'
                                )(pairs)
    else:
        pairs = layers.Flatten_2D_Op(Conv2D(final_size,(1,1),
                                kernel_initializer='glorot_uniform',strides=(1,1)),
                                name=block_name+'/Conv-last'
                                )(pairs)
    if config.get('pooling') == 'mean':
        summed = Lambda(lambda x: K.mean(x,axis=1),name=block_name+'/MeanPairs')(pairs)
    elif config.get('pooling') == 'sum':
        summed = Lambda(lambda x: K.sum(x,axis=1),name=block_name+'/SummedPairs')(pairs)
    elif config.get('pooling') == 'max':
        summed = Lambda(lambda x: K.max(x,axis=1),name=block_name+'/MaxedPairs')(pairs)
    else:
        raise ValueError('Pooling must be one of mean/sum/max. {} not a recognised method'.format(config.get('pooling')))
    if concat_input_output:
        summed = Concatenate(axis=-1,name=block_name+'/ConcatenateInputOutput')([summed,descriptors])
    return summed


def descriptor_block(descriptors,config):
    """
    Puts each descriptor vector through a neural network, as defined in config.
    """
    block_name = config['block_name']
    layer_sizes = config['layer_sizes']
    final_size = config['final_size']
    skips = config['skips']
    lasso_regularization = config['lasso_regularization']

    feats = descriptors
    skip_feats = feats
    if layer_sizes is not None:
        if isinstance(layer_sizes,int):
            layer_sizes = [layer_sizes]
        for i,n in enumerate(layer_sizes):
            feats = layers.Flatten_2D_Op(Conv2D(n,(1,1),kernel_initializer='glorot_uniform',strides=(1,1)),name=block_name+'/Conv-{}'.format(i))(feats)
            feats = LeakyReLU(alpha=0.01,name=block_name+'/LeakyReLU-{}'.format(i))(feats)
            feats = tfa.layers.GroupNormalization(groups=2, axis=-1,name=block_name+'/GroupNorm-{}'.format(i))(feats)

            # feats = BatchNormalization(axis=-1,momentum=0.99,name=block_name+'/BatchNorm-{}'.format(i))(feats)
            if skips=='add' and feats.shape[-1]==skip_feats.shape[-1]:
                feats = Add(name=block_name+'/SkipAdd-{}'.format(i))([feats,skip_feats])
            elif skips=='concat':
                feats = Concatenate(axis=-1,name=block_name+'/SkipConcat-{}'.format(i))([feats,skip_feats])
            skip_feats = feats

    if lasso_regularization:
        feats = layers.Flatten_2D_Op(Conv2D(final_size,(1,1),
                                kernel_initializer='glorot_uniform',strides=(1,1),
                                kernel_regularizer=regularizers.l1(1e-5)),
                                name=block_name+'/Conv-last'
                                )(feats)
    else:
        feats = layers.Flatten_2D_Op(Conv2D(final_size,(1,1),
                                kernel_initializer='glorot_uniform',strides=(1,1)),
                                name=block_name+'/Conv-last'
                                )(feats)
    return feats


def band_integration_block(bands,descriptors,config):
    """
    First, concatenates the band with the descriptors (by tiling the descriptors
    to the same spatial extent as bands). Then, performs stride-1 convolutions
    on resulting feature maps, and sums them, leading to a single feature map
    output.
    """
    block_name = config['block_name']
    layer_sizes = config['layer_sizes']
    final_size = config['final_size']
    skips = config['skips']
    lasso_regularization = config['lasso_regularization']
    feats = layers.Concatenate_bands_with_descriptors(rank=5,name=block_name+'/ConcatBandsAndDescriptors')((bands,descriptors))
    skip_feats = feats
    if layer_sizes is not None:
        if isinstance(layer_sizes,int):
            layer_sizes = [layer_sizes]
        for i,n in enumerate(layer_sizes):
            feats = layers.Flatten_2D_Op(Conv2D(n,(1,1),kernel_initializer='glorot_uniform',strides=(1,1)),name=block_name+'/Conv-{}'.format(i))(feats)
            feats = LeakyReLU(alpha=0.01,name=block_name+'/LeakyReLU-{}'.format(i))(feats)
            feats = tfa.layers.GroupNormalization(groups=2, axis=-1,name=block_name+'/GroupNorm-{}'.format(i))(feats)

            # feats = BatchNormalization(axis=-1,momentum=0.99,name=block_name+'/BatchNorm-{}'.format(i))(feats)
            if skips=='add' and feats.shape[-1]==skip_feats.shape[-1]:
                feats = Add(name=block_name+'/SkipAdd-{}'.format(i))([feats,skip_feats])
            elif skips=='concat':
                feats = Concatenate(axis=-1,name=block_name+'/SkipConcat-{}'.format(i))([feats,skip_feats])
            skip_feats = feats
    if lasso_regularization:
        feats = layers.Flatten_2D_Op(Conv2D(final_size,(1,1),
                                kernel_initializer='glorot_uniform',strides=(1,1),
                                kernel_regularizer=regularizers.l1(1e-5)),
                                name=block_name+'/Conv-last'
                                )(feats)
    else:
        feats = layers.Flatten_2D_Op(Conv2D(final_size,(1,1),
                                kernel_initializer='glorot_uniform',strides=(1,1)),
                                name=block_name+'/Conv-last'
                                )(feats)

    if config.get('sum_outputs'):
        if config.get('pooling') == 'mean':
            feats = Lambda(lambda x: K.mean(x,axis=1),name=block_name+'/MeanBands')(feats)
        elif config.get('pooling') == 'sum':
            feats = Lambda(lambda x: K.sum(x,axis=1),name=block_name+'/SummedBands')(feats)
        elif config.get('pooling') == 'max':
            feats = Lambda(lambda x: K.max(x,axis=1),name=block_name+'/MaxedBands')(feats)
        else:
            raise ValueError('Pooling must be one of mean/sum/max. {} not a recognised method'.format(config.get('pooling')))
    return feats

def band_multiplication_block(bands,descriptors,config):
    """
    Multiply bands by descriptors. Then sum them, leading to a single feature map
    output.
    """
    block_name = config['block_name']
    if config.get('offset'):
        bands += tf.constant(config['offset'])
    descriptors_expanded = Lambda(lambda x: K.expand_dims(K.expand_dims(x,2),2))(descriptors)

    feat_maps = Multiply(name=block_name+'/Multiply')([bands,descriptors_expanded])

    if config.get('sum_outputs'):
        if config.get('pooling') == 'mean':
            feat_map = Lambda(lambda x: K.mean(x,axis=1),name=block_name+'/MeanBands')(feat_maps)
        elif config.get('pooling') == 'sum':
            feat_map = Lambda(lambda x: K.sum(x,axis=1),name=block_name+'/SummedBands')(feat_maps)
        elif config.get('pooling') == 'max':
            feat_map = Lambda(lambda x: K.max(x,axis=1),name=block_name+'/MaxedBands')(feat_maps)
        else:
            raise ValueError('Pooling must be one of mean/sum/max. {} not a recognised method'.format(config.get('pooling')))
    return feat_map



def summed_band_block(bands,config):
    """
    A set of stride 1 convolutions to be used on summed feature maps.
    """

    block_name = config['block_name']
    layer_sizes = config['layer_sizes']
    final_size = config['final_size']
    skips = config['skips']
    lasso_regularization = config['lasso_regularization']

    feats = bands
    skip_feats = feats
    if layer_sizes is not None:
        if isinstance(layer_sizes,int):
            layer_sizes = [layer_sizes]
        for i,n in enumerate(layer_sizes):
            feats = layers.Flatten_2D_Op(Conv2D(n,(1,1),kernel_initializer='glorot_uniform',strides=(1,1)),name=block_name+'/Conv-{}'.format(i))(feats)
            feats = LeakyReLU(alpha=0.01,name=block_name+'/LeakyReLU-{}'.format(i))(feats)
            feats = tfa.layers.GroupNormalization(groups=2, axis=-1,name=block_name+'/GroupNorm-{}'.format(i))(feats)
            # feats = BatchNormalization(axis=-1,momentum=0.99,name=block_name+'/BatchNorm-{}'.format(i))(feats)
            if skips=='add' and feats.shape[-1]==skip_feats.shape[-1]:
                feats = Add(name=block_name+'/SkipAdd-{}'.format(i))([feats,skip_feats])
            elif skips=='concat':
                feats = Concatenate(axis=-1,name=block_name+'/SkipConcat-{}'.format(i))([feats,skip_feats])
            skip_feats = feats

    if lasso_regularization:
        feats = layers.Flatten_2D_Op(Conv2D(final_size,(1,1),
                                kernel_initializer='glorot_uniform',strides=(1,1),
                                kernel_regularizer=regularizers.l1(1e-5)),
                                name=block_name+'/Conv-last'
                                )(feats)
    else:
        feats = layers.Flatten_2D_Op(Conv2D(final_size,(1,1),
                                kernel_initializer='glorot_uniform',strides=(1,1)),
                                name=block_name+'/Conv-last'
                                )(feats)

    return feats


def SEnSeI(config):
    bandsin = Input((None,None,None,1),name='Input-bands')
    descriptorsin = Input((None,config['descriptor_size']),name='Input-descriptors')

    #DESCRIPTOR_BLOCK
    if config.get('DESCRIPTOR_BLOCK'):
        descriptors = descriptor_block(descriptorsin,config['DESCRIPTOR_BLOCK'])
    else:
        descriptors = descriptorsin

    #PERMUTED_DESCRIPTOR_BLOCK
    if config.get('PERMUTED_DESCRIPTOR_BLOCK'):
        descriptors = permuted_descriptor_block(descriptors,config['PERMUTED_DESCRIPTOR_BLOCK'])

    #COMBINED_DESCRIPTOR_BLOCK
    if config.get('COMBINED_DESCRIPTOR_BLOCK'):
        descriptors = descriptor_block(descriptors,config['COMBINED_DESCRIPTOR_BLOCK'])

    #BAND INTEGRATION BLOCK
    if config.get('BAND_INTEGRATION_BLOCK'):
        feature_map = band_integration_block(bandsin,descriptors,config['BAND_INTEGRATION_BLOCK'])
    elif config.get('BAND_MULTIPLICATION_BLOCK'):
        feature_map = band_multiplication_block(bandsin,descriptors,config['BAND_MULTIPLICATION_BLOCK'])
    else:
        raise Error('SEnSeI requires either a "band integration" or "band multiplication" block, config had neither.')

    #SUMMED_BAND_BLOCK
    if config.get('SUMMED_BAND_BLOCK'):
        feature_map = summed_band_block(feature_map,config['SUMMED_BAND_BLOCK'])

    return Model(inputs=(bandsin,descriptorsin),outputs=feature_map,name='SEnSeI')


def SEnSeIRecoveryModule(descriptor_size,feature_size):
    featuresin = Input((None,None,feature_size),name='recovery/Input-features')
    real_descriptorsin = Input((None,descriptor_size),name='recovery/Input-real-descriptors')
    candidate_descriptorsin = Input((None,descriptor_size),name='recovery/Input-candidate-descriptors')

    # candidate_descriptors = candidate_descriptorsin
    candidate_descriptors = descriptor_block(candidate_descriptorsin,{
                                          'block_name': 'RECOVERY/CANDIDATE_DESCRIPTOR_BLOCK',
                                          'layer_sizes':[8,12,16,24,32],
                                          'final_size': 64,
                                          'skips': 'add',
                                          'lasso_regularization': True
                                          })
    # print(candidate_descriptors.shape)

    tiled_features1 = layers.Tile_bands_to_descriptor_count(name='RECOVERY/Tile-features-1')((featuresin,candidate_descriptors))
    candidate_preds = band_integration_block(tiled_features1, candidate_descriptors,{
                                          'block_name': 'RECOVERY/CANDIDATE_PREDICTION_BLOCK',
                                          'layer_sizes':[128,64,32],
                                          'final_size': 16,
                                          'skips': None,
                                          'lasso_regularization': False
                                          })

    candidate_preds = layers.Flatten_2D_Op(Conv2D(1,(1,1),
                            kernel_initializer='glorot_uniform',strides=(1,1)),
                            name='RECOVERY/CANDIDATE_PREDICTION_BLOCK/Conv-final'
                            )(candidate_preds)
    candidate_preds = Flatten(name='candidates')(candidate_preds)
    candidate_preds = Activation('sigmoid',name='RECOVERY/CANDIDATE_PREDICTION_BLOCK/outs')(candidate_preds) # [batch_size,N_candidates]


    real_descriptors = descriptor_block(real_descriptorsin,{
                                          'block_name': 'RECOVERY/REAL_DESCRIPTOR_BLOCK',
                                          'layer_sizes':[8,12,16,24],
                                          'final_size': 64,
                                          'skips': None,
                                          'lasso_regularization': True
                                          })

    tiled_features2 = layers.Tile_bands_to_descriptor_count(name='RECOVERY/Tile-features-2')((featuresin,real_descriptors))
    band_value_estimates = band_integration_block(tiled_features2, real_descriptors,{
                                          'block_name': 'RECOVERY/BAND_ESTIMATION_BLOCK',
                                          'layer_sizes':[128,64,32],
                                          'final_size': 16,
                                          'skips': None,
                                          'lasso_regularization': False
                                          })
    #
    band_value_estimates = layers.Flatten_2D_Op(Conv2D(1,(1,1),
                            kernel_initializer='glorot_uniform',strides=(1,1)),
                            name='RECOVERY/BAND_ESTIMATION_BLOCK/Conv-final'
                            )(band_value_estimates)

    band_value_estimates = Flatten(name='bands')(band_value_estimates)
    # band_value_estimates = Activation('sigmoid',name='RECOVERY/BAND_ESTIMATION_BLOCK/outs')(band_value_estimates) # [batch_size,N_candidates]

    recovery_module = Model(inputs=(featuresin,real_descriptorsin,candidate_descriptorsin),outputs=(candidate_preds,band_value_estimates),name='SEnSeI-Recovery-Module')
    return recovery_module

if __name__=='__main__':
    import numpy as np
    import yaml
    import time
    from tensorflow.keras.mixed_precision import experimental as mixed_precision

    tf.config.experimental.set_memory_growth(
        tf.config.list_physical_devices('GPU')[0], True
        )


    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)


    with open('/home/ali/Projects/clouds/SENSEI/models/sensei/config.yaml','r') as f:
        config = yaml.load(f)
    print(config)

    recovery = SEnSeIRecoveryModule(3,64)
    recovery.summary(line_length=240)

    features = np.ones((8,1,1,64))
    descriptors = np.ones((8,7,3))
    candidates = np.ones((8,14,3))

    outs = recovery.predict((features,descriptors,candidates))
    print(outs[0].shape)
    print(outs[1].shape)
