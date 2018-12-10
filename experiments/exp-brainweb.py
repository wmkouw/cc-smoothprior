#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to test VB-EM Potts on Brainweb data.

Author: W.M.Kouw
Date: 30-10-2018
"""
import numpy as np
import numpy.random as rnd
import scipy.ndimage as nd

from sklearn.mixture import BayesianGaussianMixture

from hPotts import VariationalHiddenPotts
from vGMI import UnsupervisedGaussianMixture, SemisupervisedGaussianMixture
from vis import plot_segmentations
from util import set_classes, imfilter_Sobel

import matplotlib.pyplot as plt


'''Experimental parameters'''

# Number of repetitions
nR = 1

# Visualize predictions
vis = True

# Maximum number of iterations
max_iter = 10

# Number of patients
nP = 4

# Number of classes
K = 4

# Number of channels
D = 1

# Shape of image
imsize = (256, 256)

# Preallocate result array
err = np.zeros((4, nP, nR))

'''Repeat experiment'''

beta_hat = np.zeros((nP,))
at = np.zeros((K,))
bt = np.zeros((K,))
nt = np.zeros((K,))
mt = np.zeros((K, D))
Wt = np.zeros((D, D, K))

for r in range(nR):
    # Report progress
    print('\n At repetition ' + str(r+1) + '/' + str(nR))

    for n in np.arange(nP):
        # Report progress
        print('At patient ' + str(n+1) + '/' + str(nP) + '\n')

        # Filename current patient
        fn = '../data/Brainweb/subject' + str(n+1).zfill(2) + '_256'

        # Load scan
        X = np.fromfile(fn + '_GE2D_1.5T_RSS.raw',
                        count=np.prod(imsize),
                        dtype='uint8')

        # Reshape binary list to image
        X = nd.rotate(X.reshape(imsize), 90)

        # Normalize observations
        X[X < 0] = 0
        X[X > 255] = 255
        X = X / 255.

        # Load segmentation
        Y = np.fromfile(fn + '.raw', count=np.prod(imsize), dtype='uint8')

        # Reshape binary list to image
        Y = nd.rotate(Y.reshape(imsize), 90)

        # Restrict segmentation
        for k in np.setdiff1d(np.unique(Y), np.arange(K)):
            Y[Y == k] = 0

        # Strip skull
        X[Y == 0] = 0

        # Given labels
        y = np.zeros((*imsize, K), dtype='bool')
        z = np.zeros((K, 3), dtype='uint8')
        for k in range(K):

            # Indices of labels
            Yk = np.argwhere(Y == k)

            # Choose labels
            ix = rnd.choice(np.arange(Yk.shape[0]), size=1)

            # One-hot label image
            y[Yk[ix, 0], Yk[ix, 1], k] = True
            z[k, :] = np.array([Yk[ix, 0], Yk[ix, 1], k])

        # TODO: U-net first layer filters
        # X_ = nd.gaussian_filter(X, sigma=1.0)

        # # Normalize filter response
        # X_ /= np.amax(X_)

        # # Add activations as channels
        # X = np.dstack((X, X_))
        X = np.atleast_3d(X)

        # Shape of X
        H, W, D = X.shape

        '''Scikit's VB GMM'''

        # Initialize model
        model = BayesianGaussianMixture(n_components=K, verbose=3)

        # Fit model
        model.fit(X.reshape((H*W, D)))

        # Perform segmentation
        post = model.predict_proba(X.reshape((H*W, D))).reshape((H, W, K))

        # Perform segmentation
        Y_hat = model.predict(X.reshape((H*W, D))).reshape((H, W))

        # Set cluster assignments to correct tissue labels
        Y_hat = set_classes(Y_hat, z)

        # Compute error
        err[0, n, r] = np.mean(Y_hat != Y)

        if vis:
            plot_segmentations(Y, X[:, :, 0], Y_hat, show=True,
                               savefn='SCK_n' + str(n+1) + '_r' + str(r+1) + '.png')

        ''' Unsupervised Gaussian Mixture '''

        # Initialize model
        UGM = UnsupervisedGaussianMixture(num_components=K,
                                          num_channels=D,
                                          max_iter=max_iter,
                                          init_params='kmeans')

        # Segment image
        Y_hat, post, theta = UGM.segment(X)

        # Compute error
        err[1, n, r] = np.mean(Y_hat != Y)

        if vis:
            plot_segmentations(Y, X[:, :, 0], Y_hat, show=True,
                               savefn='UGM_n' + str(n+1) + '_r' + str(r+1) + '.png')

        ''' Semi-supervised Gausian Mixture'''

        # Initialize model
        SGM = SemisupervisedGaussianMixture(num_components=K,
                                            num_channels=D,
                                            max_iter=max_iter,
                                            init_params='nn')

        # Segment image
        Y_hat, nu, theta = SGM.segment(X, y)

        # Compute error
        err[2, n, r] = np.mean(Y_hat != Y)

        if vis:
            plot_segmentations(Y, X[:, :, 0], Y_hat, show=True,
                               savefn='SGM_n' + str(n+1) + '_r' + str(r+1) + '.png')

        '''Hidden Potts'''

        # Initialize model
        VHP = VariationalHiddenPotts(num_components=K,
                                     num_channels=D,
                                     max_iter=max_iter,
                                     init_params='nn',
                                     tissue_specific=False)

        # Estimate smoothing parameter
        beta_hat[n, :] = SHP.maximum_likelihood_beta(SHP.one_hot(Y),
                                                     max_iter=max_iter)

        # Segment image
        Y_h, post, theta = VHP.segment(X, y, beta=beta_hat[n])

        # Compute error
        err[3, n, r] = np.mean(Y_h != Y)

        # Plot images, plus error image
        if vis:
            plot_segmentations(Y, X[:, :, 0], Y_h, show=True,
                               savefn='VHP_n' + str(n+1) + '_r' + str(r+1) + '.png')

# Report error
print('Error scikit = ' + str(np.mean(err[0, :, :], axis=(0, 1))))
print('Error unsupervised GMM = ' + str(np.mean(err[1, :, :], axis=(0, 1))))
print('Error semi-supervised GMM = ' + str(np.mean(err[2, :, :], axis=(0, 1))))
print('Error semi-supervised VHP = ' + str(np.mean(err[3, :, :], axis=(0, 1))))