import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import AUC, MeanIoU
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.optimizers import SGD
from tensorflow_addons.layers import GroupNormalization
from skimage import exposure

from sensei import models, spectral_encoders as encs
from sensei.layers import Flatten_2D_Op, PermuteDescriptors, Tile_bands_to_descriptor_count, Concatenate_bands_with_descriptors
from sensei.data.loader import SlidingWindow
from sensei.deeplabv3p import Deeplabv3
from sensei.utils import OneHotMeanIoU


# ---TO BE CONFIGURED BY USER---
SCENEDIR = 'directory/of/scenes/as/numpy/arrays' # arrays in normalised reflectance, X-by-Y-by-channels
TRUTHDIR = 'directory/of/groundtruths/as/onehot/numpy/arrays' # Optional, set to False if no groundtruth available
SATELLITE = 'Sentinel2' # See DESCRIPTORS dict in sensei/data/utils.py for options, or to add others
OUTDIR = False # Optional, set to False if you don't want to save predictions
VISUALISE = True
STRIDE = 256
MODELS = [
      {'name':'EXAMPLE',
       'path':'models/example/SEnSeI-DLv3_S2L8.h5'},
     ]


if __name__=='__main__':

    tf.config.experimental.set_memory_growth(
        tf.config.list_physical_devices('GPU')[0], True
    )

    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)

    for model in MODELS:
        print('\n'+model['name']+'\n')
        name = model['name']

        if OUTDIR:
            os.makedirs(OUTDIR,exist_ok=True)

        model = load_model(
                    model['path'],
                    custom_objects={
                         'Flatten_2D_Op':Flatten_2D_Op,
                         'PermuteDescriptors':PermuteDescriptors,
                         'Tile_bands_to_descriptor_count':Tile_bands_to_descriptor_count,
                         'Concatenate_bands_with_descriptors':Concatenate_bands_with_descriptors,
                         'GroupNormalization':GroupNormalization,
                         'OneHotMeanIoU':OneHotMeanIoU
                         })

        scenes = os.listdir(SCENEDIR)
        for i,s in enumerate(scenes):
            print(i,s.replace('.npy',''),end='\r')
            scene = np.load(os.path.join(SCENEDIR,s))

            swind = SlidingWindow(scene,model,satellite=SATELLITE,batch_size=8,patch_size=257,bands='all',stride=STRIDE)
            mask = swind.predict()
            if OUTDIR:
                np.save(os.path.join(OUTDIR,s),mask)

            if VISUALISE:
                mask = mask[...,-1]
                fig,ax = plt.subplots(figsize=(16,16))
                norm_scene = (scene[...,3:0:-1]-scene[...,3:0:-1].min())/(scene[...,3:0:-1].max()-scene[...,3:0:-1].min())
                ax.imshow(exposure.equalize_adapthist(norm_scene,clip_limit=0.01))
                disp_mask = np.zeros((*mask.shape,4))
                if TRUTHDIR:
                    truth = np.load(os.path.join(TRUTHDIR,s))
                    truth = np.argmax(truth,axis=-1)
                    disp_mask[(mask>0.5)*(truth==1)]=[0,1.0,0,0.1]
                    disp_mask[(mask<0.5)*(truth==1)]=[1.0,0,0,0.1]
                    disp_mask[(mask>0.5)*(truth==0)]=[1.0,1.0,0,0.1]
                else:
                    disp_mask[mask>0.5]=[0.9,0,1.0,0.1]

                ax.imshow(disp_mask)
                ax.set_title(s.replace('.npy',''),fontsize=16)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.tight_layout()
                plt.show()
