import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.layers import Input
from tensorflow.keras.metrics import AUC, MeanIoU
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import SGD
from tensorflow_addons.layers import GroupNormalization
import yaml

from sensei.utils import OneHotMeanIoU
from sensei.data.loader import Dataloader, CommonBandsDataloader
from sensei.data.utils import SYNTHETIC_DICT
from sensei.data import transformations as trf
from sensei.callbacks import LearningRateLogger, ImageCallback
from sensei import models
from sensei.layers import Flatten_2D_Op, PermuteDescriptors, Tile_bands_to_descriptor_count, Concatenate_bands_with_descriptors
from sensei.deeplabv3p import Deeplabv3



#Unknown issue means memory growth must be True, otherwise breaks. May be hardware-specific?
tf.config.experimental.set_memory_growth(
    tf.config.list_physical_devices('GPU')[0], True
)

#Mixed precision training, allows ~doubling of batch size
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)


def main(config):
    model = build_model(config)
    loaders = get_loaders(config)
    train_model(model,*loaders,config)

def build_model(config):

    if config['SENSEI']:
        BANDS = (3,14)  # samples random set of bands numbering between 3-14
        bandsin = Input(shape=(None, None, None, 1))
        descriptorsin = Input(shape=(None, 3))
        sensei_pretrained = load_model(
                    config['SENSEI_PATH'],
                    custom_objects={        #annoying requirement when using custom layers
                        'GroupNormalization':GroupNormalization,
                        'Flatten_2D_Op':Flatten_2D_Op,
                        'PermuteDescriptors':PermuteDescriptors,
                        'Tile_bands_to_descriptor_count':Tile_bands_to_descriptor_count,
                        'Concatenate_bands_with_descriptors':Concatenate_bands_with_descriptors
                        })
        sensei = sensei_pretrained.get_layer('SEnSeI')
        feature_map = sensei((bandsin,descriptorsin))
        NUM_CHANNELS = sensei.output_shape[-1]
    else:
        if config['S2_L8_COMMON']:
            NUM_CHANNELS=8
        else:
            NUM_CHANNELS=13

    # ---CloudFCN---
    if config['MODEL_TYPE']=='CloudFCN':
        mainin = Input(shape=(None, None, NUM_CHANNELS))
        outs = models.build_cloudFCN(mainin, num_channels=NUM_CHANNELS, num_classes=config['CLASSES'])
        main_model = Model(inputs=mainin,outputs=outs,name='CloudFCN')
    # ---DeepLabv3---
    elif config['MODEL_TYPE']=='DeepLabv3':
        main_model = Deeplabv3(
                        input_shape=(config['PATCH_SIZE'], config['PATCH_SIZE'], NUM_CHANNELS),
                        classes=config['CLASSES'],
                        backbone='mobilenetv2'
                        )
    else:
        print('MODEL_TYPE not recognised')
        sys.exit()

    if config['SENSEI']:
        outs = main_model(feature_map)
        model = Model(inputs=(bandsin,descriptorsin), outputs=outs)
        return model
    else:
        return main_model

def train_model(model, train_loader, valid_loader, display_loader, config):
    logdir = os.path.join('./logs',config['NAME'])
    modeldir = logdir.replace('logs','models')
    os.makedirs(modeldir,exist_ok=True)

    # If this model name already exists, ask if old weights should be used
    if os.path.exists(os.path.join(modeldir,'latest.h5')):
        choice = input('saved weights located for this model. Would you like to use these weights? (y or n): ')
        if choice.lower()=='y':
            # model.load_weights(os.path.join(modeldir,'latest.h5'))
            model = load_model(
                        os.path.join(modeldir,'latest.h5'),
                        custom_objects={
                            'GroupNormalization':GroupNormalization,
                            'Flatten_2D_Op':Flatten_2D_Op,
                            'PermuteDescriptors':PermuteDescriptors,
                            'Tile_bands_to_descriptor_count':Tile_bands_to_descriptor_count,
                            'Concatenate_bands_with_descriptors':Concatenate_bands_with_descriptors
                            }
                        )

            # LR = float(input('Select new initial learning rate: '))
            initial_epoch = int(input(
                                'Select new initial epoch (you could check '
                                'tensorboard for the last step): '
                                ))
    else:
        initial_epoch = 0

    # Random validation images for tensorboard visualisation
    np.random.seed(13)
    disp_idxs = np.array(np.random.choice(np.arange(len(display_loader)),25))
    file_writer_images = tf.summary.create_file_writer(logdir + '/images')
    image_callback = ImageCallback(display_loader,file_writer_images,idxs = disp_idxs)

    lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.25,
        patience=3,
        min_lr=2e-5
        )

    if config['CLASSES']==2:
        metrics=['categorical_accuracy', AUC()]

    if config['CLASSES']==3:
        metrics=['categorical_accuracy', OneHotMeanIoU(config['CLASSES'])]

    optimizer = SGD(lr=config['LR'], momentum=config['MOMENTUM'], nesterov=True)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=metrics
        )
    if config['SENSEI']:
        model.get_layer('SEnSeI').summary(line_length=160)
        model.get_layer('deeplabv3p').summary(line_length=160)
    model.summary(line_length=160)

    callback_list = [
        TensorBoard(log_dir=logdir, update_freq='epoch'),
        ModelCheckpoint(
            modeldir+'/{epoch:02d}-{val_loss:.2f}.h5',
            save_weights_only=False,
            save_best_only=True,
            monitor='val_loss',
            mode='min'
            ),
        ModelCheckpoint(
            modeldir+'/latest.h5',
            save_weights_only=False,
            save_best_only=False
            ),
        lr_schedule,
        LearningRateLogger(),
        image_callback
        ]

    model.fit(
        train_loader,
        validation_data=valid_loader,
        validation_steps=len(valid_loader),
        steps_per_epoch=config['STEPS_PER_EPOCH'],
        epochs=config['EPOCHS'],
        initial_epoch = initial_epoch,
        callbacks=callback_list,
        shuffle=True
        )



def get_loaders(config,band_selection='all'):
    if config['CLASSES']==2:
        class_dict = {
            'clear':'CLEAR',
            'cloud':'CLOUD',
            'shadow':'CLEAR',
            'SHADOW OVER WATER':'CLEAR',
            'WATER':'CLEAR',
            'ICE/SNOW':'CLEAR',
            'FLOODED':'CLEAR',
            'LAND':'CLEAR',
            'CLOUD':'CLOUD',
            'FILL': 'CLEAR',
            'SHADOW': 'CLEAR',
            'CLEAR':'CLEAR',
            'THIN': 'CLOUD',
            'THICK': 'CLOUD'
            }
    elif config['CLASSES']==3:
        class_dict = {
            'clear':'CLEAR',
            'cloud':'CLOUD',
            'shadow':'SHADOW',
            'SHADOW OVER WATER':'SHADOW',
            'WATER':'CLEAR',
            'ICE/SNOW':'CLEAR',
            'FLOODED':'CLEAR',
            'LAND':'CLEAR',
            'CLOUD':'CLOUD',
            'FILL': 'CLEAR',
            'SHADOW': 'SHADOW',
            'CLEAR':'CLEAR',
            'THIN': 'CLOUD',
            'THICK': 'CLOUD'
            }
    else:
        print(
            'CLASSES not recognised, should either be 2 (clear vs. cloud), '
            'or 3 (clear vs. cloud vs. shadow)'
            )


    train_transformations = [
        trf.Base(config['PATCH_SIZE']),
        trf.Class_merge(class_dict),
        trf.Sometimes(0.5,trf.Chromatic_scale(factor_min=0.95, factor_max=1.05)),
        trf.Sometimes(0.5,trf.Bandwise_salt_and_pepper(0.001,0.001,pepp_value=0,salt_value=1.1)),
        trf.Sometimes(0.1,trf.Salt_and_pepper(0.001,0.001,pepp_value=0,salt_value=1.1)),
        trf.Sometimes(0.05,trf.Quantize(30,min_value=-1,max_value=2)),
        trf.Sometimes(0.05,trf.Quantize(40,min_value=-1,max_value=2)),
        trf.Sometimes(0.5,trf.Chromatic_shift(shift_min=-0.05,shift_max=0.05)),
        trf.Sometimes(0.5,trf.White_noise(sigma=0.02)),
        trf.Sometimes(0.5,trf.Descriptor_scale(factor_min=0.99,factor_max=1.01))
        ]
    if config['SYNTHETIC_BANDS']:
        train_transformations.append(trf.Sometimes(0.5,trf.Synthetic_bands(SYNTHETIC_DICT,N=3,p=0.5)))

    valid_transformations = [
        trf.Base(config['PATCH_SIZE'], fixed=True),
        trf.Class_merge(class_dict)
        ]

    if config['SENSEI']:
        convert_shape = True
        output_descriptors = True
        descriptor_style=config['DESCRIPTOR_STYLE']
        BANDS = (3,14)
    else:
        convert_shape = False
        output_descriptors = False
        descriptor_style=config['DESCRIPTOR_STYLE']
        BANDS = 'all'

    if config['S2_L8_COMMON']: #Use CommonBandsDataloader
        train_loader = CommonBandsDataloader(config['TRAIN_DIRS'], config['BATCH_SIZE'], config['PATCH_SIZE'], shuffle=True,
                                  transformations=train_transformations,
                                  band_selection=BANDS,
                                  convert_shape = convert_shape,
                                  output_descriptors = output_descriptors,
                                  descriptor_style = descriptor_style,
                                  repeated=2 # makes sure there are enough samples for 1000 steps at batch_size = 8
                                  )

        valid_loader = CommonBandsDataloader(config['VALID_DIRS'], config['BATCH_SIZE'], config['PATCH_SIZE'],
                                  transformations=valid_transformations,
                                  band_selection='all',
                                  convert_shape = convert_shape,
                                  output_descriptors = output_descriptors,
                                  descriptor_style = descriptor_style
                                  )

        display_loader = CommonBandsDataloader(config['VALID_DIRS'], 1, config['PATCH_SIZE'],
                                  transformations=valid_transformations,
                                  band_selection='all',
                                  convert_shape = convert_shape,
                                  output_descriptors = output_descriptors,
                                  descriptor_style = descriptor_style
                                  )

    else:
        train_loader = Dataloader(config['TRAIN_DIRS'], config['BATCH_SIZE'], config['PATCH_SIZE'], shuffle=True,
                                  transformations=train_transformations,
                                  band_selection=BANDS,
                                  convert_shape = convert_shape,
                                  output_descriptors = output_descriptors,
                                  descriptor_style = descriptor_style,
                                  repeated=2 # makes sure there are enough samples for 1000 steps at batch_size = 8
                                  )

        valid_loader = Dataloader(config['VALID_DIRS'], config['BATCH_SIZE'], config['PATCH_SIZE'],
                                  transformations=valid_transformations,
                                  band_selection='all',
                                  convert_shape = convert_shape,
                                  output_descriptors = output_descriptors,
                                  descriptor_style = descriptor_style
                                  )

        display_loader = Dataloader(config['VALID_DIRS'], 1, config['PATCH_SIZE'],
                                  transformations=valid_transformations,
                                  band_selection='all',
                                  convert_shape = convert_shape,
                                  output_descriptors = output_descriptors,
                                  descriptor_style = descriptor_style
                                  )

    return (train_loader, valid_loader, display_loader)

if __name__=='__main__':
    with open(sys.argv[1],'r') as f:
        config = yaml.load(f)
    main(config)
