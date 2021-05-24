from concurrent.futures import ThreadPoolExecutor
from imageio import imread
import json
import numpy as np
import os
from pprint import pprint
import random
from scipy import misc
from tensorflow.keras.utils import Sequence
import tensorflow as tf
import warnings

from sensei.data import transformations as trf
from sensei.data import utils
from sensei.data.utils import DESCRIPTORS

class Dataloader(Sequence):
    """
    Customisable generator for use with keras' model.fit() method. Does not hold
    dataset in memory, making it ideal for large quantities of data, but it is
    recommended to use as fast a drive as possible (preferably an m.2 SSD).

    Dataset should be saved on disk using format from https://github.com/ESA-PhiLab/eo4ai

    Attributes
    ----------
    dirs : str
        Path to root directory of dataset. Dataloader will recursively search for valid
        subdirectories (see self.parse_dirs() for information on the criteria used).
    batch_size : int
        Number of examples per batch
    patch_size : int
        Size of returned images and masks
    transformations : list, optional
        Functions like those defined in sensei.data.transformations, used to
          augment data. Recommended to at least use sensei.data.transformations.Base
    shuffle : bool, optional
        If True, selects examples for batch randomly, instead of in order
    num_threads : int, optional
        Each example's preparation can be done in parallel, this defines the number of
          threads to use. Values above self.batch_size won't give additional speed.
          IMPORTANT: This is not the same as keras' model.fit() workers=... argument,
          and may cause issues if set above 1.
    band_selection : str / int / tuple / list, optional
        How to select the spectral bands of the data:
          - str: Either "all" or "RGB" or "notRGB", fixed selection of those bands.
          - int: Random selection of that number of bands
          - tuple: Random selection of bands, with some number between values in tuple.
              IMPORTANT-tuple MUST BE LENGTH 2.
          - list: Fixed selection of those bands at indices defined in list, more flexible
              than using the str format as its totally customisable.
    convert_shape : bool, optional
        Used to convert shape from default (B,X,Y,C) to (B,C,X,Y,1) which is used
          for input into SEnSeI
    output_metadata : bool, optional
        Useful for checking dataset integrity/debugging, allows output of each example's
          metadata in the batch
    output_descriptors : bool, optional
        Required for use with SEnSeI, adds descriptors to batch
    descriptor_style : str, optional
        Type of descriptor encoding used, can either be "log" or "bandpass". We use "log"
          for our experiments.
    rotatereflect : bool, optional
        Quick way to get roto-reflection augmentation without defining functions
          from sensei.data.transformations
    repeated : int, optional
        Can be used to simply repeat the dataset by a given number of times, which
          gets around keras' model.fit() running out of data.
    """

    def __init__(self, dirs, batch_size, patch_size,
                    transformations=None, shuffle=False, num_threads=1,
                    band_selection=None, convert_shape = True, output_metadata=False, output_filename=False,
                    output_descriptors=True,descriptor_style='log',rotatereflect=False,repeated=False):
        self.dirs = dirs
        self.paths = self.parse_dirs()  # returns full paths for annotation folders
        self.repeated = repeated
        if self.repeated:
            self.paths = self.paths*self.repeated
        self.N_samples = len(self.paths)
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.transformations = transformations
        self.shuffle = shuffle
        self.num_threads = num_threads
        self.band_selection = band_selection
        self.convert_shape = convert_shape
        self.output_metadata = output_metadata
        self.output_filename = output_filename
        self.output_descriptors = output_descriptors
        self.descriptor_style = descriptor_style
        self.rotatereflect = rotatereflect

        if self.rotatereflect:
            if isinstance(self.transformations[0],trf.Base):
                if self.transformations[0].fixed==False:
                    warnings.warn('Setting rotatereflect to True samples all 8 roto-reflection symmetries. With random rotation and reflection already applied in self.transformations, this will not be effective.')

        print('Dataset created with {} samples'.format(self.N_samples))

    def __len__(self):
        if self.rotatereflect:
            return 8*int(np.ceil(self.N_samples / float(self.batch_size)))
        else:
            return int(np.ceil(self.N_samples / float(self.batch_size)))

    def __getitem__(self, idx):
        if not self.shuffle:
            idxs = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        elif self.N_samples >= self.batch_size:
            idxs = np.random.choice(
                self.N_samples, self.batch_size, replace=False)
        else:
            idxs = np.random.choice(len(self.dataset), self.batch_size, replace=True)

        self.band_policy = self._get_band_policy()
        paths = self._get_paths(idxs)

        #Use multi-threading
        if self.num_threads>1:
            with ThreadPoolExecutor(self.num_threads) as pool:
                batch = list(pool.map(self._read_and_transform, paths))
        #Use single threading
        else:
            batch = [self._read_and_transform(p) for p in paths]

        ims, descriptors, masks, metadata = zip(*batch)
        ims = np.array(ims)
        descriptors = np.array(descriptors)
        masks = np.array(masks)

        if self.rotatereflect:
            rota = np.floor(8*idx/len(self)).astype('int')
            flip = np.floor(2*idx/len(self)).astype('int')
            if rota:
                ims = np.rot90(ims, k=rota,axes=(1,2))
                masks = np.rot90(masks, k=rota,axes=(1,2))
            if flip:
                ims = np.flip(ims,axis=1)
                masks = np.flip(masks,axis=1)

        if self.convert_shape:
            ims = np.moveaxis(ims,-1,1)[...,np.newaxis]

        if self.output_descriptors:
            return_tuple = [(ims,descriptors), masks]
        else:
            return_tuple = [ims,masks]
        if self.output_metadata:
            return_tuple.append(metadata)

        if self.output_filename:
            return_tuple.append(paths)
        if len(return_tuple) == 2:
            return return_tuple[0], return_tuple[1]
        elif len(return_tuple) == 3:
            return return_tuple[0], return_tuple[1], return_tuple[2]
        elif len(return_tuple) == 4:
            return return_tuple[0], return_tuple[1], return_tuple[2], return_tuple[3]
        elif len(return_tuple) == 5:
            return return_tuple[0], return_tuple[1], return_tuple[2], return_tuple[3], return_tuple[4]

    def _get_paths(self, indices):
        if isinstance(indices, slice):
            if indices.step is None:
                indices = range(indices.start, indices.stop)
            else:
                indices = range(indices.start, indices.stop, indices.step)
        return [self.paths[idx%self.N_samples] for idx in indices]


    def _read_and_transform(self, paths):
        im,descriptors,mask,metadata = self._read(paths)
        for transform in self.transformations:
            im,descriptors,mask,metadata =  transform(im,descriptors,mask,metadata)

        # Normalise descriptors:
        descriptors = self._encode_descriptors(descriptors)

        im,descriptors,metadata = self._select_bands(im,descriptors,metadata)
        return im, descriptors, mask, metadata

    def parse_dirs(self):
        """
        Look for all valid subdirectories in self.dirs.

        Returns
        -------
        valid_subdirs : list
            Paths to all subdirectories containing:
              - image.npy
              - descriptors.npy
              - mask.npy
              - metadata.json
        """
        valid_subdirs = []
        if isinstance(self.dirs, str):
            self.dirs = [self.dirs]
        for dir in self.dirs:
            for root, dirs, paths in os.walk(dir):
                valid_subdirs += [os.path.join(root, dir) for dir in dirs
                                  if  os.path.isfile(os.path.join(root, dir, 'image.npy'))
                                  and os.path.isfile(os.path.join(root, dir, 'descriptors.npy'))
                                  and os.path.isfile(os.path.join(root, dir, 'mask.npy'))
                                  and os.path.isfile(os.path.join(root, dir, 'metadata.json'))]
        return valid_subdirs

    def _read(self,path):
        filenames = ['image.npy','descriptors.npy','mask.npy','metadata.json']

        image_file, descriptor_file, mask_file, metadata_file = [os.path.join(path,fname) for fname in filenames]
        im = np.load(image_file,mmap_mode='r').astype(np.float32)
        descriptors = np.load(descriptor_file)

        mask = np.load(mask_file,mmap_mode='r')
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        return im, descriptors, mask, metadata

    def _encode_descriptors(self,descriptors):
        if self.descriptor_style == 'bandpass':
            centres = descriptors[:,1]
            widths = 1.5*np.log((descriptors[:,2]-descriptors[:,0])/centres)/np.log(15)+1.5
            logcentres = centres

            RGB_min,RGB_max = 400,680
            RGB_centre = (RGB_max+RGB_min)/2
            NIR_min,NIR_max = 680,1100
            NIR_centre = (NIR_max+NIR_min)/2
            cirr_min,cirr_max = 1100,1600
            cirr_centre = (cirr_max+cirr_min)/2
            SWIR_min,SWIR_max = 1100,3000
            SWIR_centre = (SWIR_max+SWIR_min)/2
            TIR_min,TIR_max = 9000, 13000
            TIR_centre = (TIR_max+TIR_min)/2


            RGB_pass = 0.5+0.5*np.tanh(1*(logcentres-RGB_centre)/(RGB_max-RGB_centre))
            NIR_pass = 0.5+0.5*np.tanh(1.5*(logcentres-NIR_centre)/(NIR_max-NIR_centre))
            cirr_pass = 0.5+0.5*np.tanh(2*(logcentres-cirr_centre)/(cirr_max-cirr_centre))
            SWIR_pass = 0.5+0.5*np.tanh(2*(logcentres-SWIR_centre)/(SWIR_max-SWIR_centre))
            TIR_pass = 0.5+0.5*np.tanh(2*(logcentres-TIR_centre)/(TIR_max-TIR_centre))

            return np.stack([RGB_pass, NIR_pass, cirr_pass, SWIR_pass, TIR_pass, widths], axis=1)
        elif self.descriptor_style == 'log':
            return np.log10(descriptors-300) - 2

    def _select_bands(self,im,descriptors,metadata):
        padding = 0

        if self.band_policy is None or self.band_policy == 'all':
            return im,descriptors,metadata
        elif self.band_policy == 'RGB':
            try:
                band_indices = [metadata['named_bands']['Red'], metadata['named_bands']['Green'], metadata['named_bands']['Blue']]
            except:
                raise ValueError(
                    'RGB selected, however data does not appear to contain "Red", "Green" and "Blue" bands.'
                )
        elif self.band_policy == 'notRGB':
            rgb_indices = [metadata.get('Red'), metadata.get('Green'), metadata.get('Blue')]
            band_indices = [i for i in range(im.shape[-1]) if i not in rgb_indices]
        elif isinstance(self.band_policy,list):
            band_indices = self.band_policy
        elif isinstance(self.band_policy,int):
            if self.band_policy <= len(metadata['bands']):
                band_indices = random.sample(list(range(len(metadata['bands']))), self.band_policy)

            else:
                band_indices = list(range(len(metadata['bands'])))
                padding = self.band_policy - len(metadata['bands'])
        else:
            print('Band policy:  {}  not recognised.'.format(self.band_policy))

        im = im[...,band_indices]
        descriptors = descriptors[band_indices,...]
        metadata = self._remap_metadata_in_band_select(metadata,band_indices)

        if padding:
            im = np.concatenate([im, -0.5*np.ones((*im.shape[:-1], padding))], axis=-1)
            descriptors = np.concatenate([descriptors, -np.ones((padding, *descriptors.shape[1:]))], axis=0)
            metadata['bands'] += [None] * padding
            metadata['band_widths'] += [None] * padding
            metadata['band_centres'] += [None] * padding
            metadata['band_types'] += [None] * padding

        return im,descriptors,metadata

    def _get_band_policy(self):
        if isinstance(self.band_selection,tuple):
            return random.randint(*self.band_selection)
        elif isinstance(self.band_selection,str):
            return self.band_selection
        elif isinstance(self.band_selection,list):
            return self.band_selection
        else:
            return None

    def _remap_metadata_in_band_select(self,metadata,band_indices):
        for name,idx in metadata['named_bands'].items():
            if idx is not None:
                if idx in band_indices:
                    metadata['named_bands'][name] = band_indices.index(idx)
                else:
                    metadata['named_bands'][name] = None

        metadata['bands'] = [metadata['bands'][b] for b in band_indices]
        metadata['band_widths'] =  [metadata['band_widths'][b] for b in band_indices]
        metadata['band_centres'] =  [metadata['band_centres'][b] for b in band_indices]
        metadata['band_types'] =  [metadata['band_types'][b] for b in band_indices]

        return metadata

class CommonBandsDataloader(Dataloader):

    def _read(self,path):
        filenames = ['image.npy','descriptors.npy','mask.npy','metadata.json']

        image_file, descriptor_file, mask_file, metadata_file = [os.path.join(path,fname) for fname in filenames]
        im = np.load(image_file,mmap_mode='r').astype(np.float32)
        descriptors = np.load(descriptor_file)


        mask = np.load(mask_file,mmap_mode='r')
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        band_policy_save = self.band_policy
        if metadata['spacecraft_id']=='Sentinel2':
            self.band_policy = [0,1,2,3,7,10,11,12]
            im,descriptors,metadata = self._select_bands(im,descriptors,metadata)
        elif metadata['spacecraft_id']=='Landsat8':
            self.band_policy = [0,1,2,3,4,8,5,6]
            im,descriptors,metadata = self._select_bands(im,descriptors,metadata)
        else:
            Error('metadata of data doees not contain recognised satellite for SharedBandsDataloader')
        self.band_policy = band_policy_save
        return im, descriptors, mask, metadata

class SEnSeITrainer(Sequence):

    def __init__(self,descriptor_size,batch_size,num_channels=(3,15),test_mode=False):
        self.descriptor_size = descriptor_size
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.test_mode = test_mode

        if self.test_mode:
            self.descriptor_bank = self.generate_test_descriptor_bank()

    def __len__(self):
        return 10000

    def __getitem__(self,idx):
        if isinstance(self.num_channels,tuple):
            C = np.random.randint(*self.num_channels)
        else:
            C = self.num_channels

        if self.test_mode:
            possible_descriptors = np.array([self.descriptor_bank[np.random.choice(self.descriptor_bank.shape[0],2*C,replace=False),:] for i in range(self.batch_size)])
            descriptors = possible_descriptors[:,:C,...]
            band_values = 1.7*np.random.uniform(size=(self.batch_size,C,1,1,1))

        else:
            descriptors = self.generate_descriptors(C)
            band_values = 1.7*np.random.uniform(size=(self.batch_size,C,1,1,1))
            possible_descriptors = np.concatenate([descriptors,self.generate_descriptors(C)],axis=1)

        outputs = np.concatenate([np.ones((self.batch_size,C)),np.zeros((self.batch_size,C))],axis=1)

        return (band_values,descriptors,possible_descriptors),{'candidates':outputs,'band_values':np.squeeze(band_values)}

    def generate_descriptors(self,C):

        lefts = 2.5*np.random.uniform(size=(self.batch_size,C,1))
        middles = lefts+0.02+0.05*np.abs(np.random.normal(size=(self.batch_size,C,1)))
        rights = middles+0.02+0.05*np.abs(np.random.normal(size=(self.batch_size,C,1)))
        return np.concatenate([lefts,middles,rights],axis=2)

    def generate_test_descriptor_bank(self):
        centres = np.arange(0.15,2.5,0.06)
        widelefts = centres-0.15
        widerights = centres+0.15

        narrowlefts = centres-0.02
        narrowrights = centres+0.02

        wides = np.stack([widelefts,centres,widerights])
        narrows = np.stack([narrowlefts,centres,narrowrights])
        return np.concatenate([narrows,wides],axis=1).T


class SlidingWindow(Dataloader):

    def __init__(self,scene,model,satellite=None,stride=256,patch_size=256,batch_size=1,bands='all',ensemble=False):
        self.scene = scene
        if self.scene.ndim==3:
            self.scene = self.scene[np.newaxis,...]
        self.model = model

        # Default to Sentinel-2
        if satellite is None:
            self.satellite = 'Sentinel2'
        else:
            self.satellite = satellite

        self.bands = bands

        if len(self.model.inputs)==2:
            self.add_descriptors = True
            self.descriptors = self.get_descriptors()
        else:
            self.add_descriptors = False
            self.descriptors = None

        self.stride = stride
        self.patch_size = patch_size
        self.batch_size = batch_size

        self.ensemble = ensemble

        ys = np.arange(0,self.scene.shape[1]-self.patch_size+self.stride,self.stride)
        xs = np.arange(0,self.scene.shape[2]-self.patch_size+self.stride,self.stride)

        y_shift = int((ys[-1]+self.patch_size-self.scene.shape[1])/2)
        x_shift = int((xs[-1]+self.patch_size-self.scene.shape[2])/2)

        self.yy,self.xx = np.meshgrid(ys-y_shift,xs-x_shift)
        self.yy = self.yy.ravel()
        self.xx = self.xx.ravel()


    def __len__(self):
        return int(np.ceil(len(self.yy) / float(self.batch_size)))

    def __getitem__(self,idx):
        idxs = self._get_slice(idx)
        patches = self.make_patches(idxs)
        if self.add_descriptors:
            descriptors = np.concatenate([self.descriptors[np.newaxis,...]]*patches.shape[0],axis=0)
            patches = np.moveaxis(patches,-1,1)[...,np.newaxis]
            return patches, descriptors
        else:
            return patches

    def _get_slice(self,idx):
        idxs = np.arange(idx * self.batch_size, min((idx + 1) * self.batch_size,len(self.yy)))
        return idxs

    def predict(self):

        total_mask = np.zeros((self.scene.shape[1],self.scene.shape[2],self.model.output.shape[-1]))
        count_mask = np.zeros((self.scene.shape[1],self.scene.shape[2],self.model.output.shape[-1]))

        masks = np.zeros((len(self.yy),self.patch_size,self.patch_size,self.model.output.shape[-1]))
        for i in range(len(self)):
            idxs = self._get_slice(i)
            if self.add_descriptors:
                ims,descriptors = self[i]
                masks[idxs,...] = self.model.predict((ims,descriptors))
            else:
                ims = self[i]
                masks[idxs,...] = self.model.predict(ims)
            if self.ensemble:
                for j in range(self.ensemble):
                    band_indices = random.sample(list(range(descriptors.shape[1])), random.randint(3,ims.shape[1]))
                    ensemble_ims = ims[:,band_indices,...]
                    ensemble_descriptors = descriptors[:,band_indices,...]
                    masks[idxs,...] += self.model.predict((ensemble_ims,ensemble_descriptors))
                masks[idxs,...] = masks[idxs,...]/(self.ensemble+1)

        for ymin,xmin,m in zip(self.yy.ravel(),self.xx.ravel(),masks):
            if ymin+self.patch_size>total_mask.shape[0]:
                m = m[:-(ymin+self.patch_size-total_mask.shape[0]),:,:]
                ymax = total_mask.shape[0]
            else:
                ymax = ymin+self.patch_size

            if xmin+self.patch_size>total_mask.shape[1]:
                m = m[:,:-(xmin+self.patch_size-total_mask.shape[1]),:]
                xmax = total_mask.shape[1]
            else:
                xmax = xmin+self.patch_size


            if ymin<0:
                m = m[-ymin:,:,:]
                ymin = 0

            if xmin<0:
                m = m[:,-xmin:,:]
                xmin = 0

            total_mask[ymin:ymax,xmin:xmax,:]+=m
            count_mask[ymin:ymax,xmin:xmax,:]+=1

        if np.sum(count_mask==0)!=0:
            warnings.warn('Some areas of mask not covered, make sure stride < patch_size')

        total_mask = np.divide(total_mask,np.clip(count_mask,1,np.float('inf')))

        return total_mask


    def make_patches(self,idxs):

        yy,xx = self.yy[idxs],self.xx[idxs]

        if self.bands=='all':
            patches = np.zeros((len(yy),self.patch_size,self.patch_size,self.scene.shape[-1]))
        else:
            patches = np.zeros((len(yy),self.patch_size,self.patch_size,len(self.bands)))
        for i, (y, x) in enumerate(zip(yy,xx)):
            ymax, xmax = y+self.patch_size, x+self.patch_size
            # calculate relative bounding box (if it goes over the edge of the scene, correct the relative position in the patches)
            rel_y, rel_x = max(0,y)-y, max(0,x)-x
            rel_ymax = self.patch_size-(ymax-min(self.scene.shape[1],ymax))
            rel_xmax = self.patch_size-(xmax-min(self.scene.shape[2],xmax))
            if self.bands=='all':
                patch = self.scene[0,max(0,y):min(ymax,self.scene.shape[1]),max(0,x):min(xmax,self.scene.shape[2]),:]
            else:
                patch = np.moveaxis(self.scene[0,max(0,y):min(ymax,self.scene.shape[1]),max(0,x):min(xmax,self.scene.shape[2]),self.bands],0,-1)
            for j in range(patch.shape[-1]):
                patch[...,j][patch[...,j]==0] = np.mean(patch[...,j][patch[...,j]!=0])
            patches[i,rel_y:rel_ymax,rel_x:rel_xmax,:] = patch
        return patches


    def get_descriptors(self):
        descriptors = DESCRIPTORS[self.satellite]
        if not self.bands=='all':
            descriptors = descriptors[self.bands,...]
        descriptors = self._encode_descriptors(descriptors)
        return descriptors

    def _encode_descriptors(self,descriptors):
        return np.log10(descriptors-300) - 2
