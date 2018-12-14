#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Experiment script to test training on MRBrainS13 and testing on IBSR20.

Author: W.M.Kouw
Date: 13-12-2018
"""
import os
import numpy as np
import numpy.random as rnd
import scipy.ndimage as nd
import matplotlib.pyplot as plt

from scipy.spatial.distance import dice

from sklearn.mixture import BayesianGaussianMixture
from sklearn.neighbors import KNeighborsClassifier

from VGGUnet import VGGUnet
from hPotts import VariationalHiddenPotts
from vGMI import UnsupervisedGaussianMixture, SemisupervisedGaussianMixture
from util import subject2image, set_classes, ConvolutionalNetwork, filter_Sobel
from vis import plot_segmentation, plot_clustering, plot_scan


'''Experimental parameters'''

# Generic filename for this experiment
fn = 'exp-cross-mbib-'

# Number to save results to
savenumber = '04'

# Number of repetitions
nR = 5

# Visualize predictions
vis = False

# Number of source patients
nP_S = 5

# Number of target patients
nP_T = 18

# Number of classes
K = 4

# Number of channels
D = 1

# Shape of image
H = 256
W = 256

# Maximum number of iterations
max_iter = 30
x_tol = 1e-3

# Preallocate result array
err = np.ones((7, nP_T, nR))
dcc = np.ones((7, nP_T, nR))

''' Train classifier on Brainweb data set '''

# Construct Brainweb data set
X = np.zeros((nP_S, H, W, 3))
Y = np.zeros((nP_S, H, W, K))
beta_h = np.ones((nP_S, K))/2

for n in np.arange(nP_S):
    # Report progress
    print('At patient ' + str(n+1) + '/' + str(nP_S) + '\n')

    # Patient directory
    pdir_mb = '../data/MRBrains/TrainingData/'

    fnX = pdir_mb + str(n+1) + '/T1.nii'
    fnY = pdir_mb + str(n+1) + '/LabelsForTesting.nii'

    # Load scan
    scan = subject2image(fnX, slice_ix=69, slice_dim=2, normalize=True)
    segm = subject2image(fnY, slice_ix=69, slice_dim=2)

    # One-hot label image encoding
    hPotts = VariationalHiddenPotts(num_components=K,
                                    num_channels=D,
                                    tissue_specific=True)
    Y1 = hPotts.one_hot(segm)

    # Strip skull
    scan[segm == 0] = 0

    for d in range(3):
        X[n, :, :, d] = scan
    Y[n, :, :, :] = Y1

    if os.path.exists('beta_mrbrains.npy'):

        # Load previously estimated smoothing parameters
        beta_h = np.load('beta_mrbrains.npy')

    else:
        # Estimate smoothing parameters
        beta_h[n, :] = hPotts.maximum_likelihood_beta(Y1)

# Save smoothing parameter
if not os.path.exists('beta_mrbrains.npy'):
    np.save('beta_mrbrains.npy', beta_h)

# Take average smoothing parameter
beta = np.mean(beta_h, axis=0)
print('Beta = ' + str(beta))

# Train VGGU-net on data set
Unet = VGGUnet(K)
Unet.fit(X, Y, epochs=max_iter)

'''Repeat experiment'''

at = np.zeros((K,))
bt = np.zeros((K,))
nt = np.zeros((K,))
mt = np.zeros((K, D))
Wt = np.zeros((D, D, K))

for r in range(nR):
    # Report progress
    print('\nAt repetition ' + str(r+1) + '/' + str(nR))

    for n in range(nP_T):
        # Report progress
        print('At patient ' + str(n+1) + '/' + str(nP_T) + '\n')

        # Filename current patient
        trn_dir = '../data/IBSR/IBSR_' + str(n + 1).zfill(2)
        fnX = trn_dir + '/IBSR_' + str(n + 1).zfill(2) + '_ana.nii'
        fnY = trn_dir + '/IBSR_' + str(n + 1).zfill(2) + '_seg_ana.nii'

        # Load scan
        scan = subject2image(fnX,
                             slice_dim=1,
                             slice_ix=128,
                             flipud=True,
                             normalize=True)
        segm = subject2image(fnY,
                             slice_dim=2,
                             slice_ix=128,
                             flipud=True,
                             CMA=True)

        # Find brain mask
        M = (Y != 0)

        # Strip skull
        X[~M] = 0

        # Incude channels
        X = np.atleast_3d(X)

        # Random sampling weights
        sample_weights = np.exp(-filter_Sobel(X)[:, :, 0])

        # Given labels
        y = np.zeros((H, W, K), dtype='bool')
        z = np.zeros((K, 3), dtype='uint8')
        for k in range(K):

            # Indices of labels
            Yk = np.argwhere(Y == k)
            Wk = sample_weights[Y == k]

            # Choose labels
            ix = rnd.choice(np.arange(Yk.shape[0]), size=1, p=Wk / np.sum(Wk))

            # One-hot label image
            y[Yk[ix, 0], Yk[ix, 1], k] = True
            z[k, :] = np.array([Yk[ix, 0], Yk[ix, 1], k])

        if vis:

            fn_segs = fn + 'TRUE_scan_p' + str(n+1) + '_r' + str(r+1) + '.png'
            plot_scan(X[30:-30, 30:-30, 0], savefn=fn_segs)

            fn_segs = fn + 'TRUE_segs_p' + str(n+1) + '_r' + str(r+1) + '.png'
            plot_segmentation(Y[30:-30, 30:-30], savefn=fn_segs)

        ''' Source neural network '''

        X_ = np.zeros((1, H, W, 3))
        for c in range(3):
            X_[0, :, :, c] = X[:, :, 0]

        # Compute posteriors
        post = Unet.predict(X_)

        # Make predictions
        Y_h = np.argmax(post[0, :, :, :], axis=2)

        # Compute error
        err[6, n, r] = np.mean(Y_h[M] != Y[M])
        dcc[6, n, r] = dice(Y_h[M], Y[M])

        if vis:

            fn_segs = fn + 'sUN_segs_p' + str(n+1) + '_r' + str(r+1) + '.png'
            plot_segmentation(Y_h[30:-30, 30:-30], savefn=fn_segs)

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
        dcc[0, n, r] = dice(Y_h[M], Y[M])

        if vis:

            fn_segs = fn + 'SCK_segs' + str(n+1) + '_r' + str(r+1) + '.png'
            plot_segmentation(Y_h[30:-30, 30:-30], savefn=fn_segs)

            fn_segs = fn + 'SCK_segl' + str(n+1) + '_r' + str(r+1) + '.png'
            plot_clustering(X[30:-30, 30:-30, 0],
                            Y_h[30:-30, 30:-30],
                            mode='subpixel',
                            savefn=fn_segs)

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
        dcc[1, n, r] = dice(Y_h[M], Y[M])

        if vis:

            fn_segs = fn + 'UGM_segs' + str(n+1) + '_r' + str(r+1) + '.png'
            plot_segmentation(Y_h[30:-30, 30:-30], savefn=fn_segs)

            fn_segs = fn + 'UGM_segl' + str(n+1) + '_r' + str(r+1) + '.png'
            plot_clustering(X[30:-30, 30:-30, 0],
                            Y_h[30:-30, 30:-30],
                            mode='subpixel',
                            savefn=fn_segs)

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
        dcc[2, n, r] = dice(Y_h[M], Y[M])

        if vis:

            fn_segs = fn + 'SGM_segs_p' + str(n+1) + '_r' + str(r+1) + '.png'
            plot_segmentation(Y_h[30:-30, 30:-30], savefn=fn_segs)

        ''' Unsupervised hidden Potts'''

        # Initialize model
        UHP = VariationalHiddenPotts(num_components=K,
                                     num_channels=D,
                                     max_iter=max_iter,
                                     tol=x_tol,
                                     init_params='kmeans',
                                     tissue_specific=True)

        # Segment image
        Y_h, post, theta = UHP.segment(X, beta=beta)

        # Set cluster assignments to correct tissue labels
        Y_h = set_classes(Y_h, z)

        # Compute error
        err[3, n, r] = np.mean(Y_h[M] != Y[M])
        dcc[3, n, r] = dice(Y_h[M], Y[M])

        # Plot images, plus error image
        if vis:

            fn_segs = fn + 'UHP_segs' + str(n+1) + '_r' + str(r+1) + '.png'
            plot_segmentation(Y_h[30:-30, 30:-30], savefn=fn_segs)

            fn_segs = fn + 'UHP_segl' + str(n+1) + '_r' + str(r+1) + '.png'
            plot_clustering(X[30:-30, 30:-30, 0],
                            Y_h[30:-30, 30:-30],
                            mode='subpixel',
                            savefn=fn_segs)

        ''' Semi-supervised hidden Potts'''

        # Initialize model
        SHP = VariationalHiddenPotts(num_components=K,
                                     num_channels=D,
                                     max_iter=max_iter,
                                     init_params='nn',
                                     tol=x_tol,
                                     tissue_specific=True)

        # Segment image
        Y_h, post, theta = SHP.segment(X, y, beta=beta)

        # Compute error
        err[4, n, r] = np.mean(Y_h[M] != Y[M])
        dcc[4, n, r] = dice(Y_h[M], Y[M])

        # Plot images, plus error image
        if vis:

            fn_segs = fn + 'SHP_segs_p' + str(n+1) + '_r' + str(r+1) + '.png'
            plot_segmentation(Y_h[30:-30, 30:-30], savefn=fn_segs)

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
        dcc[5, n, r] = dice(Y_h[M], Y[M])

        # Plot images, plus error image
        if vis:

            fn_segs = fn + 'KNN_segs_p' + str(n+1) + '_r' + str(r+1) + '.png'
            plot_segmentation(Y_h[30:-30, 30:-30], savefn=fn_segs)

# Save error results
np.save('results/' + fn + '_errors_' + str(savenumber) + '.npy', err)
np.save('results/' + fn + '_dice_' + str(savenumber) + '.npy', dcc)

# Report errors
print('Mean error:')
print('Error SCK = ' + str(np.mean(err[0, :, :], axis=(0, 1))))
print('Error sUN = ' + str(np.mean(err[6, :, :], axis=(0, 1))))
print('Error UGM = ' + str(np.mean(err[1, :, :], axis=(0, 1))))
print('Error SGM = ' + str(np.mean(err[2, :, :], axis=(0, 1))))
print('Error UHP = ' + str(np.mean(err[3, :, :], axis=(0, 1))))
print('Error SHP = ' + str(np.mean(err[4, :, :], axis=(0, 1))))
print('Error kNN = ' + str(np.mean(err[5, :, :], axis=(0, 1))))

# Report errors
print('Standard error of the mean:')
print('SEM SCK = ' + str(np.std(err[0, :, :]) / np.sqrt(nR)))
print('SEM sUN = ' + str(np.std(err[6, :, :]) / np.sqrt(nR)))
print('SEM UGM = ' + str(np.std(err[1, :, :]) / np.sqrt(nR)))
print('SEM SGM = ' + str(np.std(err[2, :, :]) / np.sqrt(nR)))
print('SEM UHP = ' + str(np.std(err[3, :, :]) / np.sqrt(nR)))
print('SEM SHP = ' + str(np.std(err[4, :, :]) / np.sqrt(nR)))
print('SEM kNN = ' + str(np.std(err[5, :, :]) / np.sqrt(nR)))

print('Mean DICE:')
print('DICE SCK = ' + str(np.mean(dcc[0, :, :], axis=(0, 1))))
print('DICE sUN = ' + str(np.mean(dcc[6, :, :], axis=(0, 1))))
print('DICE UGM = ' + str(np.mean(dcc[1, :, :], axis=(0, 1))))
print('DICE SGM = ' + str(np.mean(dcc[2, :, :], axis=(0, 1))))
print('DICE UHP = ' + str(np.mean(dcc[3, :, :], axis=(0, 1))))
print('DICE SHP = ' + str(np.mean(dcc[4, :, :], axis=(0, 1))))
print('DICE kNN = ' + str(np.mean(dcc[5, :, :], axis=(0, 1))))
