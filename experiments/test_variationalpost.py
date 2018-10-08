#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Function to test inference of the smoothing parameter of a hidden Potts-MRF.

Author: W.M.Kouw
Date: 18-09-2018
"""
import numpy as np
import numpy.random as rnd
import scipy.optimize as opt

import matplotlib.pyplot as plt

from tomopy.misc import phantom as ph
from hPottsMRF import hiddenPottsMarkovRandomField

# Generate checkerboard
L = ph.checkerboard(size=32, dtype='uint8')[0, :, :]

# Normalize board
L = np.round(L / 255.).astype('uint8')

# Create array with 3 unique values
L += L + 1
L = np.pad(L, [8, 8], mode='constant', constant_values=0)

# Generate observation matrix from label image
R = rnd.randn(*L.shape)*.1 + 1

# Observed image
Y = L + R

# Call instance of hPottsMRF
model = hiddenPottsMarkovRandomField()

# Segment observed image
nu, theta = model.segment(Y, K=3, Q=L, output_params=True)
print(theta)

Z = np.argmax(nu, axis=2)

# Plot segmentation
fg, ax = plt.subplots(nrows=2, ncols=3, figsize=(12,6))

plot00 = ax[0, 0].imshow(Y)
# plt.colorbar(plot00)
ax[0, 0].set_title('Observed image')

ax[0, 1].imshow(Z)
# ax[0, 1].colorbar()
ax[0, 1].set_title('Predicted segmentation')

ax[0, 2].imshow(np.abs(L - Z))
# ax[0, 2].colorbar()
ax[0, 2].set_title('Absolute prediction error')

for k in range(3):
    ax[1, k].imshow(nu[:, :, k])
    # ax[1, k].colorbar()
    ax[1, k].set_title('Predictions for class ' + str(k))

plt.show()
