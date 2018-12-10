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

from vGMM import variationalGaussianMixture

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

'''Experimental parameters'''

# Visualize predictions
vis = True

# Number of iterations
nI = 5

# Number of components
nK = 2

'''Repeat experiment'''

# Generate bivariate normal
data = pd.read_csv('../data/Old Faithful/old_faithful.csv')
X = data.as_matrix()

# Visualize data
if vis:
    data.plot.scatter(x='TimeEruption', y='TimeWaiting')

# Initialize model
model = variationalGaussianMixture(num_components=2)

# Segment image
pred, post, params = model.fit(X, max_iter=nI)

# Visualize predictions
if vis:
    fig = plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=post)

'''Posteriors for hyperparameters.'''

at, bt, nt, mt, Wt = params

# Visualize posterior parameters
if vis:

    # Span grid
    x = np.linspace(0, 5, 50)
    y = np.linspace(0, 5, 50)
    X, Y = np.meshgrid(x, y)

    # Initialize grid probability surface
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y

    # Create bivariate normals with posterior parameters
    for k in range(nK):

        # Define random variate
        px = st.multivariate_normal(mean=mt[0], cov=alg.inv(Wt[k]))

        # Draw contour
        plt.contour(X, Y, px.pdf(pos))
