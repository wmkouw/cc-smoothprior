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

from hPottsMRF import hiddenPotts

# Generate checkerboard
Z = ph.checkerboard(size=16, dtype='uint8')[0, :, :]

# Normalize board
Z = np.round(Z / 255.).astype('uint8')

# Create array with 3 unique values
Z += Z + 1
Z = np.pad(Z, [4, 4], mode='constant', constant_values=0)

# Call instance of hPottsMRF
model = hiddenPotts(num_classes=3, tissue_specific=True)

# Map Z to one-hot data
Z = model.one_hot(Z)

# Initial value
beta0 = np.ones((3, ))

# Print gradient for current beta
cost = model.mean_field_Potts(beta0, Z)
print(cost)

# Print gradient for current beta
grad = model.mean_field_Potts_grad(beta0, Z)
print(grad)

# Check gradient
diff = opt.check_grad(model.mean_field_Potts,
                      model.mean_field_Potts_grad, beta0, Z)
print(diff)

# # Estimate beta on checkerboard image
beta = model.maximum_likelihood_beta(Z,
                                     verbose=True,
                                     ub=[None, None, None],
                                     max_iter=10)

print(beta)
