[![PyPI version](https://badge.fury.io/py/csbdeep.svg)](https://pypi.org/project/csbdeep)
[![Linux build status](https://travis-ci.com/CSBDeep/CSBDeep.svg?branch=master)](https://travis-ci.com/CSBDeep/CSBDeep)
[![Windows build status](https://ci.appveyor.com/api/projects/status/xbl32vudixshj990/branch/master?svg=true)](https://ci.appveyor.com/project/UweSchmidt/csbdeep-c2jtk)

# CSBDeep – a toolbox for CARE

This is the CSBDeep Python package, which provides a toolbox for content-aware restoration 
of fluorescence microscopy images (CARE), based on deep learning via Keras and TensorFlow.

Please see the documentation at http://csbdeep.bioimagecomputing.com/doc/.


## Fork info

This CARE fork explores multibody U-Net opportunities for 
denoising multiplane data from self (e.g. for z-stacked brightfield microscopy images). 

Contributions:
* `care_multiplane.py` presents MultiplaneCARE class which reconstructs N image planes with N U-Net bodies 
in leave-one-out way;
* `nets.py` implements `uxnet()` which returns multibody U-Net model; 
* `blocks.py` has convolutional blocks reimplemented as Keras Sequential models
for weight sharing.