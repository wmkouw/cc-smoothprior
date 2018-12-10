#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test variational GMM implementation on checkerboard image.

Author: W.M.Kouw
Date: 05-11-2018
"""
import pandas as pd
import numpy as np
import numpy.random as rnd
import scipy.optimize as opt
import scipy.stats as st
import matplotlib.pyplot as plt

from vGMM import variationalGaussianMixture
from util import generate_checkerboard
from vis import plot_segmentations

'''Experimental parameters'''

# Visualize predictions
vis = True

# Number of repetitions
nR = 1

# Number of iterations
nI = 500

# Number of classes
nK = 2

# Gaussian noise parameters
mu = [0, 0]
si2 = [0.2, 0.2]

# Shape of image
shape = 20

'''Repeat experiment'''

la_h = np.zeros((nR, nK))
be_h = np.zeros((nR, nK))
ce_h = np.zeros((nR, nK))
em_h = np.zeros((nR, nK))
ve_h = np.zeros((nR, nK))

for r in np.arange(nR):
    # Report progress
    print('At repetition ' + str(r) + '/' + str(nR))

    # Generate image according to set parameters
    Y = generate_checkerboard(shape=shape)

    # Add independent Gaussian noise
    X = np.zeros(Y.shape)
    for k in range(nK):
        X[Y == k] += rnd.normal(mu[k], np.sqrt(si2[k]), np.sum(Y == k))

    # Initialize model
    model = variationalGaussianMixture(num_components=2)

    # Segment image
    Y_hat, nu, theta = model.segment(X, max_iter=nI)

    # Store estimated parameters
    la_h[r, :], be_h[r, :], ce_h[r, :], em_h[r, :], ve_h[r, :] = theta

    # Plot images, plus error image
    if vis:
        plot_segmentations(Y, X, Y_hat)

'''Posteriors for hyperparameters.'''

la_ = np.mean(em_h, axis=0)
be_ = np.mean(be_h, axis=0)
ce_ = np.mean(ce_h, axis=0)
em_ = np.mean(em_h, axis=0)
ve_ = np.mean(ve_h, axis=0)

# All classes are deemed independent
for k in range(nK):

    # Expected posterior precision
    si2_h = 1 / (ce_[k] * be_[k])

    # Expected posterior mean
    mu_h = em_[k]

    # Check whether posterior distributions center around noise distributions
    print("mu_h_" + str(k) + " = " + str(mu_h) + " (" + str(mu[k] + k) + ")")
    print("si_h_" + str(k) + " = " + str(si2_h) + " (" + str(si2[k]) + ")")
