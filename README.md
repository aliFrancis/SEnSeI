# SEnSeI: Spectral Encoder for Sensor Independence

A python 3 package for developing sensor independent deep learning models for cloud masking in satellite imagery.


## Installation

Download the source code, and install as a python package:

```
git clone https://github.com/aliFrancis/SEnSeI
cd SEnSeI
python setup.py install
```

## Basic usage

Models can be trained using the sensei/fitting/fit_model.py script, with a .yaml config file. Some fields in the .yaml files included in sensei/fitting/config/ need to be set by the user, such as the paths to the datasets. Before training, the user must follow the instructions below in the *Datasets* section to get their training data ready. To train a model:

```
python sensei/fitting/fit_model.py sensei/fitting/config/<some-yaml-file>.yaml
```

This will fit the model, save the weights, and keep a tensorboard log.

To run a model you have trained (or the example model included in this repository) on new data, you can use the _inference.py_ script. Unlike in training, the data for inference does not need to be processed in the specific style described in the Datasets section below. Images can be in X-by-Y-by-channels arrays (with normalized reflectances). If using a model including SEnSeI, the satellite that the data comes from must be specified (see the options at the top of _inference.py_). For satellites we have not yet defined, you can append the _DESCRIPTORS_ held in _sensei/data/utils.py_, and pass that new satellite's name.

```
python sensei/inference.py
```

To customise the architecture and unsupervised training of SEnSeI, you can use the _sensei/fit_sensei.py_ script, and amend the config file _models/sensei/sensei.yaml_.

## Datasets

Because SEnSeI requires both the reflectance values of the satellite _and_ the wavelengths of each channel, the data must be preprocessed in a specific way. This package does not deal with dataset preprocessing. Instead, all datasets we use in our experiments are prepared using our other package, [_eo4ai_](https://github.com/ESA-PhiLab/eo4ai), which will output datasets in the format required for SEnSeI.

For example, our 513 scene Sentinel-2 dataset can be downloaded from [its Zenodo page](https://zenodo.org/record/4172871):

```
wget https://zenodo.org/record/4172871/files/subscenes.zip -O ~/Downloads/Sentinel-2-subscenes.zip
wget https://zenodo.org/record/4172871/files/masks.zip -O ~/Downloads/Sentinel-2-masks.zip
```

And then uncompressed:

```
cd <path/to/eo4ai/project>
python setup.py install
mkdir -p datasets/downloaded/Sentinel-2
unzip -d datasets/downloaded/Sentinel-2 ~/Downloads/Sentinel-2-subscenes.zip
unzip -d datasets/downloaded/Sentinel-2 ~/Downloads/Sentinel-2-masks.zip
```

Finally, prepared using the _prepare_dataset.py_ script included in _eo4ai_, with a resolution of 20m, in patches of size 263 pixels across, with a stride of 253 pixels (these values can of course be changed to suit your needs):

```
python eo4ai/prepare_dataset.py -g -r 20 -p 263 -s 253 S2IRIS513 datasets/downloaded/Sentinel-2 datasets/processed/Sentinel-2
```

A similar procedure is possible for the other datasets supported by _eo4ai_, and should produce data compatible with the models in this repository. Note that the [Landsat 8 CCA dataset](https://landsat.usgs.gov/landsat-8-cloud-cover-assessment-validation-data) (also known as the Biome dataset) contains the panchromatic band, whilst the [SPARCS dataset](https://www.usgs.gov/core-science-systems/nli/landsat/spatial-procedures-automated-removal-cloud-and-shadow-sparcs) does not. If a non-sensor independent model is to be used on both, eo4ai's *-b* switch can be used to exclude the panchromatic band from the processed Biome dataset, making them compatible.

For our experiments, we split the data into training, validation and test sets in a 40:10:50 ratio. To reproduce our experiments precisely, the scenes used in each can be found in _Sentinel-2-splits/_.
