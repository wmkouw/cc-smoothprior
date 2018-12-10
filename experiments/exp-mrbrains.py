#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to test VB-EM Potts on Brainweb data.

Author: W.M.Kouw
Date: 30-10-2018
"""
import pandas as pd
import numpy as np
import numpy.random as rnd
import scipy.optimize as opt
import scipy.stats as st
import scipy.ndimage as nd
import matplotlib.pyplot as plt

from sklearn.mixture import BayesianGaussianMixture
from sklearn.neighbors import KNeighborsClassifier

from hPotts import VariationalHiddenPotts
from vGMI import UnsupervisedGaussianMixture, SemisupervisedGaussianMixture
from util import subject2image, set_classes
from vis import plot_segmentations, plot_posteriors


'''Experimental parameters'''

# Number to save results to
savenumber = '02'

# Number of repetitions
nR = 10

# Visualize predictions
vis = True

# Number of patients
nP = 5

# Number of classes
K = 4

# Number of channels
D = 1

# Shape of image
H = 256
W = 256

# Maximum number of iterations
max_iter = 50
x_tol = 1e-3

# Preallocate result array
err = np.ones((6, nP, nR))

'''Repeat experiment'''

beta_hat = np.zeros((nP, K))
at = np.zeros((K,))
bt = np.zeros((K,))
nt = np.zeros((K,))
mt = np.zeros((K, D))
Wt = np.zeros((D, D, K))

for r in range(nR):
    # Report progress
    print('At repetition ' + str(r+1) + '/' + str(nR) + '\n')

    for n in range(nP):
        # Report progress
        print('At patient ' + str(n+1) + '/' + str(nP) + '\n')

        # Filename current patient
        trn_dir = '../data/MRBrains/TrainingData/'
        fnX = trn_dir + str(n+1) + '/T1.nii'
        fnY = trn_dir + str(n+1) + '/LabelsForTesting.nii'

        # Load scan
        X = subject2image(fnX, normalize=True)
        Y = subject2image(fnY, seg=True)

        # Expand to channels
        X = np.atleast_3d(X)

        # Extract brain mask
        M = (Y != 0)

        # Strip skull
        X[~M] = 0

        # y = np.zeros((H, W, K), dtype='bool')
        # y[50, 50, 0] = True
        # y[100, 120, 1] = True
        # y[155, 120, 2] = True
        # y[90, 100, 3] = True
        # z = np.array([[50, 50, 0],
        #               [100, 130, 1],
        #               [155, 120, 2],
        #               [85, 95, 3]])

        # Given labels
        y = np.zeros((H, W, K), dtype='bool')
        z = np.zeros((K, 3), dtype='uint8')
        for k in range(K):

            # Indices of labels
            Yk = np.argwhere(Y == k)

            # Choose labels
            ix = rnd.choice(np.arange(Yk.shape[0]), size=1)

            # One-hot label image
            y[Yk[ix, 0], Yk[ix, 1], k] = 1
            z[k, :] = np.array([Yk[ix, 0], Yk[ix, 1], k])

        '''Scikit's VB GMM'''

        # Initialize model
        model = BayesianGaussianMixture(n_components=K,
                                        max_iter=max_iter,
                                        verbose=3)

        # Fit model
        model.fit(X.reshape((-1, 1)))

        # Segment image
        Y_h = model.predict(X.reshape((-1, 1))).reshape((H, W))

        # Obtain posteriors
        post = model.predict_proba(X.reshape((-1, 1))).reshape((H, W, K))

        # Set cluster assignments to correct tissue labels
        Y_h = set_classes(Y_h, z)

        # Compute error
        err[0, n, r] = np.mean(Y_h[M] != Y[M])

        if vis:

            fn_segs = 'exp-mrbrains_SCK_segs_p' + str(n+1) + '.png'
            fn_post = 'exp-mrbrains_SCK_post_p' + str(n+1) + '.png'

            plot_segmentations(Y, X[:, :, 0], Y_h, savefn=fn_segs)
            # plot_posteriors(post, savefn=fn_post)

        ''' Unsupervised Gaussian Mixture '''

        # Initialize model
        UGM = UnsupervisedGaussianMixture(num_components=K,
                                          num_channels=D,
                                          max_iter=max_iter,
                                          tol=x_tol,
                                          init_params='kmeans')

        # Segment image
        Y_h, post, theta = UGM.segment(X)

        # Set cluster assignments to correct tissue labels
        Y_h = set_classes(Y_h, z)

        # Compute error
        err[1, n, r] = np.mean(Y_h[M] != Y[M])

        if vis:

            fn_segs = 'exp-mrbrains_UGM_segs_p' + str(n+1) + '.png'
            fn_post = 'exp-mrbrains_UGM_post_p' + str(n+1) + '.png'

            plot_segmentations(Y, X[:, :, 0], Y_h, savefn=fn_segs)
            # plot_posteriors(post, savefn=fn_post)

        ''' Semi-supervised Gaussian Mixture '''

        # Initialize model
        SGM = SemisupervisedGaussianMixture(num_components=K,
                                            num_channels=D,
                                            max_iter=max_iter,
                                            tol=x_tol,
                                            init_params='nn')

        # Segment image
        Y_h, post, theta = SGM.segment(X, y)

        # Compute error
        err[2, n, r] = np.mean(Y_h[M] != Y[M])

        if vis:
            fn_segs = 'exp-mrbrains_SGM_segs_p' + str(n+1) + '.png'
            fn_post = 'exp-mrbrains_SGM_post_p' + str(n+1) + '.png'

            plot_segmentations(Y, X[:, :, 0], Y_h, savefn=fn_segs)
            # plot_posteriors(post, savefn=fn_post)

        ''' Unsupervised hidden Potts'''

        # Initialize model
        UHP = VariationalHiddenPotts(num_components=K,
                                     num_channels=D,
                                     max_iter=max_iter,
                                     tol=x_tol,
                                     init_params='kmeans',
                                     tissue_specific=True)

        # Estimate smoothing parameter
        beta_hat[n, :] = UHP.maximum_likelihood_beta(UHP.one_hot(Y),
                                                     max_iter=max_iter)

        # Segment image
        Y_h, post, theta = UHP.segment(X, beta=beta_hat[n, :])

        # Set cluster assignments to correct tissue labels
        Y_h = set_classes(Y_h, z)

        # Compute error
        err[3, n, r] = np.mean(Y_h[M] != Y[M])

        # Plot images, plus error image
        if vis:
            fn_segs = 'exp-mrbrains_UHP_segs_p' + str(n+1) + '.png'
            fn_post = 'exp-mrbrains_UHP_post_p' + str(n+1) + '.png'

            plot_segmentations(Y, X[:, :, 0], Y_h, savefn=fn_segs)
            # plot_posteriors(post, savefn=fn_post)

        ''' Semi-supervised hidden Potts'''

        # Initialize model
        SHP = VariationalHiddenPotts(num_components=K,
                                     num_channels=D,
                                     max_iter=max_iter,
                                     init_params='nn',
                                     tol=x_tol,
                                     tissue_specific=True)

        # Estimate smoothing parameter
        beta_hat[n, :] = SHP.maximum_likelihood_beta(SHP.one_hot(Y),
                                                     max_iter=max_iter)

        # Segment image
        Y_h, post, theta = SHP.segment(X, y, beta=beta_hat[n, :])

        # Compute error
        err[4, n, r] = np.mean(Y_h[M] != Y[M])

        # Plot images, plus error image
        if vis:
            fn_segs = 'exp-mrbrains_SHP_segs_p' + str(n+1) + '.png'
            fn_post = 'exp-mrbrains_SHP_post_p' + str(n+1) + '.png'

            plot_segmentations(Y, X[:, :, 0], Y_h, savefn=fn_segs)
            # plot_posteriors(post, savefn=fn_post)

        '''Nearest neighbours'''

        # Initialize model
        kNN = KNeighborsClassifier(n_neighbors=1)

        # Observation indicator vector
        O = np.any(y, axis=2)

        # Fit classifier
        kNN.fit(X[O, :], np.argmax(y[O, :], axis=1))

        # Segment image
        Y_h = kNN.predict(X.reshape((H*W, D))).reshape((H, W))

        # Compute error
        err[5, n, r] = np.mean(Y_h[M] != Y[M])

        # Plot images, plus error image
        if vis:
            fn_segs = 'exp-mrbrains_KNN_segs_p' + str(n+1) + '.png'

            plot_segmentations(Y, X[:, :, 0], Y_h, savefn=fn_segs)

# Report errors
print('Error scikit = ' + str(np.mean(err[0, :, :], axis=(0, 1))))
print('Error unsupervised GMM = ' + str(np.mean(err[1, :, :], axis=(0, 1))))
print('Error semi-supervised GMM = ' + str(np.mean(err[2, :, :], axis=(0, 1))))
print('Error unsupervised VHP = ' + str(np.mean(err[3, :, :], axis=(0, 1))))
print('Error semi-supervised VHP = ' + str(np.mean(err[4, :, :], axis=(0, 1))))
print('Error supervised kNN = ' + str(np.mean(err[5, :, :], axis=(0, 1))))
np.save('exp-mrbrains_errors_' + str(savenumber) + '.npy', err)
