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
R = rnd.randn(*L.shape)*.2 + 1

# Observed image
Y = L + R

# Shape
H, W = Y.shape

# Corners
patch_indices = [(0, 0), (0, 10), (0, W-1),
                 (10, W-1), (H-1, W-1), (W-1, 10),
                 (H-1, 0), (10, 0), (10, 10), 
                 (5, 5), (15, 15), ((12, 12))]

# Call instance of hPottsMRF
model = hiddenPottsMarkovRandomField(neighbourhood_size=(3, 3), num_iter=5)

# Plot segmentation
fg, ax = plt.subplots(nrows=3, ncols=4, figsize=(12, 6))

# Extract neighbourhoods
cnt = 0
for row in range(3):
    for col in range(4):

        # Extract neighbourhood
        b = model.neighbourhood(Y, patch_indices[cnt])

        cnt += 1

        # Show image
        ax[row, col].imshow(b)

plt.show()
