#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to reproduce results from McGrory, Titterington, Reeves & Pettitt.

Author: W.M.Kouw
Date: 18-09-2018
"""
import pandas as pd
import numpy as np
import numpy.random as rnd
import scipy.optimize as opt
import scipy.stats as st

import matplotlib.pyplot as plt

from hPottsMRF import hiddenPotts
from util import generate_Potts
from vis import plot_segmentations

'''Experimental parameters'''

# Visualize predictions
vis = True

# Number of repetitions
nR = 1

# Number of classes
nK = 3

# Gaussian noise parameters
mu = [0, 0, 0]
si2 = [0.1, 0.1, 0.1]

# Smoothing parameter
beta = 2.0

# Shape of image
shape = (20, 20)

'''Repeat experiment'''

beta_hat = np.zeros((nR, nK))
em_hat = np.zeros((nR, nK))
la_hat = np.zeros((nR, nK))
ga_hat = np.zeros((nR, nK))
ks_hat = np.zeros((nR, nK))

for r in np.arange(nR):
    # Report progress
    print('At repetition ' + str(r) + '/' + str(nR))

    # Generate image according to set parameters
    Y, energy = generate_Potts(shape=shape, ncolors=nK, beta=beta)

    # Add independent Gaussian noise
    X = np.copy(Y).astype('float64')
    for k in range(nK):
        X[Y == k] += rnd.normal(mu[k], np.sqrt(si2[k]), np.sum(Y == k))

    # Initialize model
    model = hiddenPotts(num_classes=nK, tissue_specific=True)

    # Map label image to one-hot
    Y1 = model.one_hot(Y)

    # Estimate smoothing parameter
    beta_hat[r, :] = model.maximum_likelihood_beta(Y1, max_iter=10)

    # Segment image
    Y_hat, nu, theta = model.segment(X, beta=beta_hat[r, :], num_iter=10)

    # Store estimated parameters
    em_hat[r, :], la_hat[r, :], ga_hat[r, :], ks_hat[r, :] = theta

    # Plot images, plus error image
    if vis:
        plot_segmentations(Y, X, Y_hat, show=True)

# Report results
print('Mean estimated beta = ' + str(np.mean(beta_hat, axis=0)))

'''Posteriors for hyperparameters.'''

em_h = np.mean(em_hat, axis=0)
la_h = np.mean(la_hat, axis=0)
ga_h = np.mean(ga_hat, axis=0)
ks_h = np.mean(ks_hat, axis=0)

# All classes are deemed independent
for k in range(nK):

    # Check if mode can be computed
    if la_h[k] >= 1:

        # Modal posterior precision
        tau_hat_k = (la_h[k]/2 - 1) / (ks_h[k]/2)

    else:
        # Expected posterior precision
        tau_hat_k = la_h[k] / ks_h[k]

    # Compute estimated sigma
    si2_h_k = 1./tau_hat_k

    # Expected posterior mean
    mu_h_k = em_h[k]

    # Check whether posterior distributions center around noise distributions
    print("mu_h_" + str(k) + " = " + str(mu_h_k) + " (" + str(mu[k] + k) + ")")
    print("si_h_" + str(k) + " = " + str(si2_h_k) + " (" + str(si2[k]) + ")")
