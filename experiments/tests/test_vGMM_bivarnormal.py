#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test variational GMM implementation on Old Faithful data set.

Author: W.M.Kouw
Date: 05-11-2018
"""
import pandas as pd
import numpy as np
import numpy.random as rnd
import numpy.linalg as alg
import scipy.optimize as opt
import scipy.stats as st

from vGMM import UnsupervisedGaussianMixture, SemisupervisedGaussianMixture

import matplotlib.pyplot as plt

'''Experimental parameters'''

# Visualize predictions
vis = True

# Number of iterations
nI = 1000

# x-Tolerance
tol = 1e-12

# Sample size
N = (50, 50)

# Number of components
nK = 2

''' Generate data '''

# Means
mu0_0 = np.array([-1, -1])
mu0_1 = np.array([1, 1])

# Covariances
Si0_0 = np.array(([[5.0, -0.8], [-0.8, 0.8]]))
Si0_1 = np.array(([[0.1, 0.0], [0.0, 0.2]]))

# Generate bivariate normal
X0 = st.multivariate_normal(mean=mu0_0, cov=Si0_0).rvs(size=(N[0], 1))
X1 = st.multivariate_normal(mean=mu0_1, cov=Si0_1).rvs(size=(N[1], 1))
X = np.concatenate((X0, X1), axis=0)

# Labels
Y = np.concatenate((np.zeros((N[0],)), np.ones((N[1],))), axis=0)

# Labels that are given
y = np.array([[rnd.randint(0, N[0], size=1), 0],
              [rnd.randint(N[0], N[0]+N[1], size=1), 1]])

# Visualize data
if vis:
    fig = plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c='k')
    plt.scatter(X[y[0, 0], 0], X[y[0, 0], 1], c='b', label='y=0')
    plt.scatter(X[y[1, 0], 0], X[y[1, 0], 1], c='r', label='y=1')
    plt.legend()
    plt.title('Data set')
    plt.savefig('viz/bivarNormal_SGM_scatter01.png',
                bbox_inches=None,
                padding='tight')

''' Unsupervised model '''

# Initialize model
UGM = UnsupervisedGaussianMixture(num_components=2, max_iter=nI, tol=tol)

# Segment image
pred, post, params = UGM.fit(X)
UGM_ = {'pred': pred, 'post': post, 'params': params}

''' Semi-supervised model '''

# Initialize model
SGM = SemisupervisedGaussianMixture(num_components=2, max_iter=nI, tol=tol)

# Segment image
pred, post, params = SGM.fit(X, y)
SGM_ = {'pred': pred, 'post': post, 'params': params}

''' Report results '''

# Print error
print('Error UGM = ' + str(np.mean(UGM_['pred'] != Y)))
print('Error SGM = ' + str(np.mean(SGM_['pred'] != Y)))

# Unpack parameters
atu, btu, ntu, mtu, Wtu = UGM_['params']
ats, bts, nts, mts, Wts = SGM_['params']

# Expected posterior precision
Siu = np.zeros((2, 2, 2))
Sis = np.zeros((2, 2, 2))
for k in range(2):
    Siu[:, :, k] = alg.inv(ntu[k]*Wtu[:, :, k])
    Sis[:, :, k] = alg.inv(nts[k]*Wts[:, :, k])

# Visualize posterior parameters
if vis:

    # Span grid
    xlim = fig.gca().get_xlim()
    ylim = fig.gca().get_ylim()
    tx, ty = np.meshgrid(np.linspace(*xlim, 101), np.linspace(*ylim, 101))
    txy = np.dstack((tx, ty))

    plt.figure()

    # Create same scatterplot
    plt.scatter(X[:, 0], X[:, 1], c=UGM_['post'][:, 1], cmap='bwr')
    plt.colorbar()
    plt.scatter(X[y[0, 0], 0], X[y[0, 0], 1], c='b', label='y=0')
    plt.scatter(X[y[1, 0], 0], X[y[1, 0], 1], c='r', label='y=1')

    # Create bivariate normals with posterior parameters
    px_0 = st.multivariate_normal(mean=mtu[0, :], cov=Siu[:, :, 0])
    px_1 = st.multivariate_normal(mean=mtu[1, :], cov=Siu[:, :, 1])

    plt.contour(tx, ty, px_0.pdf(txy), colors='b')
    plt.contour(tx, ty, px_1.pdf(txy), colors='r')
    plt.legend()
    plt.title('Iso-probability lines')
    plt.savefig('viz/normal_UGM_iso01.png', bbox_inches=None, padding='tight')

    plt.figure()

    # Create same scatterplot
    plt.scatter(X[:, 0], X[:, 1], c=SGM_['post'][:, 1], cmap='bwr')
    plt.colorbar()
    plt.scatter(X[y[0, 0], 0], X[y[0, 0], 1], c='b', label='y=0')
    plt.scatter(X[y[1, 0], 0], X[y[1, 0], 1], c='r', label='y=1')

    # Create bivariate normals with posterior parameters
    px_0 = st.multivariate_normal(mean=mts[0, :], cov=Sis[:, :, 0])
    px_1 = st.multivariate_normal(mean=mts[1, :], cov=Sis[:, :, 1])

    plt.contour(tx, ty, px_0.pdf(txy), colors='b')
    plt.contour(tx, ty, px_1.pdf(txy), colors='r')
    plt.legend()
    plt.title('Iso-probability lines')
    plt.savefig('viz/normal_SGM_iso01.png', bbox_inches=None, padding='tight')

    plt.show()
