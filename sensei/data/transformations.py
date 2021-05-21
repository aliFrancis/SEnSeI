import numpy as np
import os
from scipy import misc
import time
import random
import copy

class Base:
    """
    Makes transformation for image/mask pair that is a randomly cropped, rotated
    and flipped portion of the original.

    Parameters
    ----------
    patch_size : int
        Spatial dimension of output image/mask pair (assumes Width==Height).
    fixed : bool, optional
        If True, always take patch from top-left of scene, with no rotation or
        flipping. This is useful for validation and reproducability.

    Returns
    -------
    apply_transform : func
        Transformation for image/mask pairs.
    """
    def __init__(self, patch_size, fixed = False):
        self.patch_size = patch_size
        self.fixed = fixed

    def __call__(self,img, descriptors, mask, metadata):
        if self.fixed:
            left = 0
            top = 0
            crop_size = int(
                min(self.patch_size, img.shape[0], img.shape[1]))
            img = img[top:top + crop_size, left:left + crop_size, ...]
            mask = mask[top:top + crop_size, left:left + crop_size, ...]

        else:
            if not self.patch_size == img.shape[0]:
                crop_size = int(
                    min(self.patch_size, img.shape[0] - 1, img.shape[1] - 1))

                left = int(
                    random.randint(
                        0,
                        img.shape[1] -
                        crop_size))
                top = int(
                    random.randint(
                        0,
                        img.shape[0] -
                        crop_size))

                img = img[top:top + crop_size, left:left + crop_size, ...]
                mask = mask[top:top + crop_size, left:left + crop_size, ...]

        rota = random.choice([0, 1, 2, 3])
        flip = random.choice([True, False])
        if rota and not self.fixed:
            img = np.rot90(img, k=rota)
            mask = np.rot90(mask, k=rota)
        if flip and not self.fixed:
            img = np.fliplr(img)
            mask = np.fliplr(mask)
        return img, descriptors, mask, metadata


class Class_merge:
    """
    Create image/mask pairs where classes in mask have been merged (reduces final mask
    dimension by 1).

    Parameters
    ----------
    class_map : dict
        Relationships between input classes and desired outputs. Each possible input
        class should be used as a key, and the desired class it will be put into as
        the correponding value.


    Returns
    -------
    apply_transform : func
        Transformation for image/mask pairs.
    """
    def __init__(self,class_map):
        self.class_map = class_map
        self.out_classes = sorted(list(set([out_class for out_class in class_map.values()])))

    def __call__(self,img, descriptors, mask, metadata):
        out_mask = np.zeros((*mask.shape[:-1],len(self.out_classes)),dtype='bool')

        for inp_class,out_class in self.class_map.items():
            if inp_class in metadata['classes']:
                inp_idx = metadata['classes'].index(inp_class)
                out_idx = self.out_classes.index(out_class)
                out_mask[...,out_idx]+= mask[...,inp_idx]
        metadata['classes'] = self.out_classes
        return img, descriptors, out_mask, metadata

class Sometimes:
    """
    Wrapper function which randomly applies the transform with probability p.

    Parameters
    ----------
    p : float
        Probability of transform being applied
    transform : func
        Function which transforms image/mask pairs.

    Returns
    -------
    apply_transform : func
        Transformation for image/mask pairs.
    """

    def __init__(self,p, transform):
        self.p = p
        self.transform = transform

    def __call__(self,img, descriptors, mask, metadata):
        random_apply = random.random() < self.p
        if random_apply:
            return self.transform(img, descriptors, mask, metadata)
        else:
            return img, descriptors, mask, metadata

class Chromatic_shift:
    """
    Adds a different random amount to each spectral band in image.

    Parameters
    ----------
    shift_min : float, optional
        Lower bound for random shift.
    shift_max : float, optional
        Upper bound for random shift.

    Returns
    -------
    apply_transform : func
        Transformation for image/mask pairs.
    """
    def __init__(self,shift_min=-0.10, shift_max=0.10):
        self.shift_min=shift_min
        self.shift_max=shift_max

    def __call__(self,img, descriptors, mask, metadata):
        img = img + np.random.uniform(low=self.shift_min,
                                      high=self.shift_max, size=[1, 1, img.shape[-1]]).astype(np.float32)
        return img, descriptors, mask, metadata

class Chromatic_scale:
    """
    Multiplies each spectral band in batch by a different random factor.

    Parameters
    ----------
    factor_min : float, optional
        Lower bound for random factor.
    factor_max : float, optional
        Upper bound for random factor.

    Returns
    -------
    apply_transform : func
        Transformation for image/mask pairs.
    """
    def __init__(self,factor_min=0.90, factor_max=1.10):
        self.factor_min=factor_min
        self.factor_max=factor_max

    def __call__(self,img, descriptors, mask, metadata):
        img = img * np.random.uniform(low=self.factor_min,
                                      high=self.factor_max, size=[1, 1, img.shape[-1]]).astype(np.float32)
        return img, descriptors, mask, metadata

class Intensity_shift:
    """
    Adds single random amount to all spectral bands.

    Parameters
    ----------
    shift_min : float, optional
        Lower bound for random shift.
    shift_max : float, optional
        Upper bound for random shift.

    Returns
    -------
    apply_transform : func
        Transformation for image/mask pairs.
    """
    def __init__(self,shift_min=-0.10, shift_max=0.10):
        self.shift_min=shift_min
        self.shift_max=shift_max
    def __call__(self,img, descriptors, mask, metadata):
        img = img + (self.shift_max-self.shift_min)*random.random()+self.shift_min
        return img, descriptors, mask, metadata

class Intensity_scale:
    """
    Multiplies all spectral bands by a single random factor.

    Parameters
    ----------
    factor_min : float, optional
        Lower bound for random factor.
    factor_max : float, optional
        Upper bound for random factor.

    Returns
    -------
    apply_transform : func
        Transformation for image/mask pairs.
    """

    def __init__(self,factor_min=0.95, factor_max=1.05):
        self.factor_min=factor_min
        self.factor_max=factor_max
    def __call__(self,img, descriptors, mask, metadata):
        img = img * random.uniform(self.factor_min, self.factor_max)
        return img, descriptors, mask, metadata

class White_noise:
    """
    Adds white noise to image.

    Parameters
    ----------
    sigma : float, optional
        Standard deviation of white noise

    Returns
    -------
    apply_transform : func
        Transformation for image/mask pairs.
    """
    def __init__(self,sigma=0.1):
        self.sigma=sigma
    def __call__(self,img, descriptors, mask, metadata):
        noise = (np.random.randn(*img.shape) * self.sigma).astype(np.float32)
        return img + noise, descriptors, mask, metadata

class Bandwise_salt_and_pepper:
    """
    Adds salt and pepper (light and dark) noise to image,  treating each band independently.

    Parameters
    ----------
    salt_rate : float
        Percentage of pixels that are set to salt_value.
    pepp_rate : float
        Percentage of pixels that are set to pepp_value.
    pepp_value : float, optional
        Value that pepper pixels are set to.
    salt_value : float, optional
        Value that salt pixels are set to.

    Returns
    -------
    apply_transform : func
        Transformation for image/mask pairs.
    """
    def __init__(self,salt_rate, pepp_rate, pepp_value=0, salt_value=255):
        self.salt_rate  = salt_rate
        self.pepp_rate  = pepp_rate
        self.pepp_value = pepp_value
        self.salt_value = salt_value
    def __call__(self,img, descriptors, mask, metadata):
        salt_mask = np.random.choice([False, True], size=img.shape, p=[
                                     1 - self.salt_rate, self.salt_rate])
        pepp_mask = np.random.choice([False, True], size=img.shape, p=[
                                     1 - self.pepp_rate, self.pepp_rate])

        img[salt_mask] = self.salt_value
        img[pepp_mask] = self.pepp_value

        return img, descriptors, mask, metadata

class Salt_and_pepper:
    """
    Adds salt and pepper (light and dark) noise to image, to all bands in a pixel.

    Parameters
    ----------
    salt_rate : float
        Percentage of pixels that are set to salt_value.
    pepp_rate : float
        Percentage of pixels that are set to pepp_value.
    pepp_value : float, optional
        Value that pepper pixels are set to.
    salt_value : float, optional
        Value that salt pixels are set to.

    Returns
    -------
    apply_transform : func
        Transformation for image/mask pairs.
    """
    def __init__(self,salt_rate, pepp_rate, pepp_value=0, salt_value=255):
        self.salt_rate  = salt_rate
        self.pepp_rate  = pepp_rate
        self.pepp_value = pepp_value
        self.salt_value = salt_value
    def __call__(self,img, descriptors, mask, metadata):
        salt_mask = np.random.choice(
            [False, True], size=img.shape[:-1], p=[1 - self.salt_rate, self.salt_rate])
        pepp_mask = np.random.choice(
            [False, True], size=img.shape[:-1], p=[1 - self.pepp_rate, self.pepp_rate])

        img[salt_mask] = [self.salt_value for i in range(img.shape[-1])]
        img[pepp_mask] = [self.pepp_value for i in range(img.shape[-1])]

        return img, descriptors, mask, metadata

class Quantize:
    """
    Quantizes an image based on a given number of steps by rounding values to closest
    value.

    Parameters
    ----------
    number_steps : int
        Number of values to round to
    min_value : float
        Lower bound of quantization
    max_value : float
        Upper bound of quantization
    clip : bool
        True if values outside of [min_value:max_value] are clipped. False otherwise.

    Returns
    -------
    apply_transform : func
        Transformation for image/mask pairs.
    """
    def __init__(self,number_steps, min_value=0, max_value=255, clip=False):
        self.number_steps = number_steps
        self.min_value = min_value
        self.max_value = max_value
        self.clip = clip
        self.stepsize = (self.max_value-self.min_value)/self.number_steps

    def __call__(self,img, descriptors, mask, metadata):
        img = (img//self.stepsize)*self.stepsize
        if self.clip:
            img = np.clip(img, self.min_value, self.max_value)
        return img, descriptors, mask, metadata

class Descriptor_scale:
    """
    Multiplies all values in descriptors by a random factor.

    Parameters
    ----------
    factor_min : float, optional
        Lower bound for random factor.
    factor_max : float, optional
        Upper bound for random factor.


    Returns
    -------
    apply_transform : func
        Transformation for image/mask pairs.
    """
    def __init__(self,factor_min=0.95, factor_max=1.05):
        self.factor_min=factor_min
        self.factor_max=factor_max

    def __call__(self,img, descriptors, mask, metadata):
        factors = np.random.uniform(size=descriptors.shape, low=self.factor_min, high=self.factor_max)
        descriptors = descriptors * factors
        return img, descriptors, mask, metadata

class Descriptor_shift:
    """
    Adds a different random amount to each value in descriptors.

    Parameters
    ----------
    shift_min : float, optional
        Lower bound for random shift.
    shift_max : float, optional
        Upper bound for random shift.

    Returns
    -------
    apply_transform : func
        Transformation for image/mask pairs.
    """
    def __init__(self,shift_min=-0.05, shift_max=0.05):
        self.shift_min=shift_min
        self.shift_max=shift_max

    def __call__(self,img, descriptors, mask, metadata):
        shifts = np.random.uniform(size=descriptors.shape, low=self.shift_min, high=self.shift_max)
        descriptors = descriptors + shifts
        return img, descriptors, mask, metadata


class Synthetic_bands:
    """
    Synthesises new bands by combining pairs of existing bands.

    Parameters
    ----------
    combos : dict
        All allowed combinations of input bands for synthetic band creation. An entry for each satellite.
    N : int, optional
        Max number of combinations used
    p : float, optional
        Probability of successive combinations being applied (p=1 means N combinations will always be used). First combination always made.
    remove : bool, optional
        If True, original bands used in combination are removed. If False, both synthetic and original bands remain.
    Returns
    -------
    apply_transform : func
        Transformation for image/mask pairs.
    """
    def __init__(self,combos,N=1,p=1,remove=True):
        self.combos = combos
        self.N = N
        self.p = p
        self.remove = remove

    def __call__(self, img, descriptors, mask, metadata):

        combo_list = copy.deepcopy(self.combos[metadata['spacecraft_id']])

        #find number of combinations to use
        n = 1
        carry_over = True
        while carry_over:
            if random.random()<self.p:
                n+=1
                if n==self.N or n==len(combo_list):
                    carry_over = False
            else:
                carry_over = False
        original_bands = metadata['bands']
        to_remove = []
        for i in range(n):
            combo = random.choice(combo_list)
            combo_list.remove(combo)
            original_band_idxs = [metadata['bands'].index(b) for b in combo['inputs']]
            to_remove += original_band_idxs
            synthetic_band = np.mean(img[...,original_band_idxs],axis=-1,keepdims=True)
            synthetic_descriptor = combo['descriptor']
            synthetic_name = combo['name']
            synthetic_type = combo['type']
            img = np.concatenate([img,synthetic_band],axis=-1)
            descriptors = np.concatenate([descriptors,np.expand_dims(synthetic_descriptor,0)],axis=0)
            metadata['bands'].append(synthetic_name)
            metadata['band_types'].append(synthetic_type)
            metadata['band_centres'].append(synthetic_descriptor[1])
            metadata['band_widths'].append(synthetic_descriptor[2]-synthetic_descriptor[0])
            metadata['named_bands'][synthetic_name] = img.shape[-1]-1

        to_remove = list(set(to_remove))
        if self.remove:
            img = np.delete(img,to_remove,axis=-1)
            descriptors = np.delete(descriptors,to_remove,axis=0)
            metadata['bands'] = [e for i, e in enumerate(metadata['bands']) if i not in to_remove]
            metadata['band_types'] = [e for i, e in enumerate(metadata['band_types']) if i not in to_remove]
            metadata['band_centres'] = [e for i, e in enumerate(metadata['band_centres']) if i not in to_remove]
            metadata['band_widths'] = [e for i, e in enumerate(metadata['band_widths']) if i not in to_remove]
            for b,v in metadata['named_bands'].items():
                if v in to_remove:
                    metadata['named_bands'][b] = None
                elif v <= len(original_bands):
                    metadata['named_bands'][b] = metadata['bands'].index(original_bands[v])
                else:
                    metadata['named_bands'][b] = metadata['bands'].index(b)
        return img, descriptors, mask, metadata
