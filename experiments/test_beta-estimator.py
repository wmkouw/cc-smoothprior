#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Function to test inference of the smoothing parameter of a hidden Potts-MRF.

Author: W.M.Kouw
Date: 18-09-2018
"""
import numpy as np
import scipy.optimize as opt

from tomopy.misc import phantom as ph
from hPottsMRF import hiddenPottsMarkovRandomField

# Generate checkerboard
Z = ph.checkerboard(size=32, dtype='uint8')[0, :, :]

# Normalize board
Z = np.round(Z / 255.).astype('uint8')

# Create array with 3 unique values
Z += Z + 1
Z = np.pad(Z, [8, 8], mode='constant', constant_values=0)

# Call instance of hPottsMRF
model = hiddenPottsMarkovRandomField()

# Print gradient for current beta
grad = model.mean_field_Potts_grad(1.0, Z)
print(grad)

# # Estimate beta on checkerboard image
beta = model.maximum_likelihood_beta(Z)
print(beta)
