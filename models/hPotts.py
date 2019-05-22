#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Class of variatonal Gaussian Mixture Image models.

It serves as a baseline for a hidden Potts-MRF for Bayesian unsupervised image
segmentation.

Author: W.M. Kouw
Date: 29-11-2018
"""
import numpy as np
import numpy.random as rnd

from numpy.linalg import inv, cholesky, slogdet
from scipy.misc import logsumexp
from scipy.special import betaln, digamma, gammaln, multigammaln
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.image import extract_patches_2d

import matplotlib.pyplot as plt


class VariationalMixture(object):
    """
    Superclass of variational mixture models.

    Methods are functions common to all submixture models.
    """

    def multidigamma(self, n, p):
        """
        Multivariate digamma function.

        This function is necessary for expectations and partition functions of
        Wishart distributions. See also:
        https://en.wikipedia.org/wiki/Multivariate_gamma_function

        Parameters
        ----------
        nu : float
            Degrees of freedom.
        p : int
            Dimensionality.

        Returns
        -------
        Pp : float
            p-th order multivariate digamma function.

        """
        # Check for appropriate degree of freedom
        if not n > (p-1):
            raise ValueError('Degrees of freedom too low for dimensionality.')

        # Preallocate
        Pp = 0

        # Sum from d=1 to D
        for d in range(1, p+1):

            # Digamma function of degrees of freedom and dimension
            Pp += digamma((n + 1 - d)/2)

        return Pp

    def log_partition_Wishart(self, W, n):
        """
        Logarithmic partition function of the Wishart distribution.

        To compute variational expectations, the partition of the Wishart
        distribution is sometimes needed. The current computation follows
        Appendix B, equations B.78 to B.82 from Bishop's "Pattern Recognition &
        Machine Learning."

        Parameters
        ----------
        W : array
            Positive definite, symmetric precision matrix.
        nu : int
            Degrees of freedom.

        Returns
        -------
        B : float
            Partition of Wishart distribution.

        """
        # Extract dimensionality
        D, D_ = W.shape

        # Check for symmetric matrix
        if not D == D_:
            raise ValueError('Matrix is not symmetric.')

        # Check for appropriate degree of freedom
        if not n > D-1:
            raise ValueError('Degrees of freedom too low for dimensionality.')

        # Compute partition function
        B = (-n/2)*slogdet(W)[1] - (n*D/2)*np.log(2) - multigammaln(n, D)

        return B

    def entropy_Wishart(self, W, n):
        """
        Entropy of the Wishart distribution.

        To compute variational expectations, the entropy of the Wishart
        distribution is sometimes needed. The current computation follows
        Appendix B, equations B.78 to B.82 from Bishop's "Pattern Recognition &
        Machine Learning."

        Parameters
        ----------
        W : array
            Positive definite, symmetric precision matrix.
        nu : int
            Degrees of freedom.

        Returns
        -------
        H : float
            Entropy of Wishart distribution.

        """
        # Extract dimensionality
        D, D_ = W.shape

        # Check for symmetric matrix
        if not D == D_:
            raise ValueError('Matrix is not symmetric.')

        # Check for appropriate degree of freedom
        if not n > D-1:
            raise ValueError('Degrees of freedom too low for dimensionality.')

        # Expected log-determinant of precision matrix
        E = self.multidigamma(n, D) + D*np.log(2) + slogdet(W)[1]

        # Entropy
        H = -self.log_partition_Wishart(W, n) - (n - D - 1)/2 * E + n*D/2

        return H

    def distW(self, X, S):
        """
        Compute weighted distance.

        Parameters
        ----------
        X : array
            Vectors (N by D) or (H by W by D).
        W : array
            Weights (D by D).

        Returns
        -------
        array
            Weighted distance for each vector.

        """
        if not S.shape[0] == S.shape[1]:
            raise ValueError('Weight matrix not symmetric.')

        if not X.shape[-1] == S.shape[0]:
            raise ValueError('Dimensionality of data and weights mismatch.')

        if len(X.shape) == 2:

            # Shapes
            N, D = X.shape

            # Preallocate
            A = np.zeros((N,))

            # Loop over samples
            for n in range(N):

                # Compute weighted inner product between vectors
                A[n] = X[n, :] @ S @ X[n, :].T

        elif len(X.shape) == 3:

            # Shape
            H, W, D = X.shape

            # Preallocate
            A = np.zeros((H, W))

            # Loop over samples
            for h in range(H):
                for w in range(W):

                    # Compute weighted inner product between vectors
                    A[h, w] = X[h, w, :] @ S @ X[h, w, :].T

        return A

    def one_hot(self, A):
        """
        Map array to pages with binary encodings.

        Parameters
        ----------
        A : array
            2-dimensional array of integers

        Returns
        -------
        B : array (height by width by number of unique integers in A)
            3-dimensional array with each page as an indicator of value in A.

        """
        # Unique values
        labels = np.unique(A)

        # Preallocate new array
        B = np.zeros((*A.shape, len(labels)))

        # Loop over unique values
        for i, label in enumerate(labels):

            B[:, :, i] = (A == label)

        return B


class VariationalGaussianMixture(VariationalMixture):
    """
    Variational Gaussian Mixture Image model.

    This implementation multivariate images (height by width by channel).
    It is based on the RPubs note by Andreas Kapourani:
    https://rpubs.com/cakapourani/variational-bayes-gmm
    """

    def __init__(self, num_channels=1,
                 num_components=2,
                 init_params='nn',
                 max_iter=10,
                 tol=1e-5):
        """
        Model-specific constructors.

        Parameters
        ----------
        num_channels : int
            Number of channels of image (def: 1).
        num_components : int
            Number of components (def: 2).
        theta0 : tuple
            Prior hyperparameters.
        max_iter : int
            Maximum number of iterations to run for (def: 10).
        tol : float
            Tolerance on change in x-value (def: 1e-5).

        Returns
        -------
        None

        """
        # Store data dimensionality
        if num_channels >= 1:
            self.D = num_channels
        else:
            raise ValueError('Number of channels must be larger than 0.')

        # Store model parameters
        if num_components >= 2:
            self.K = num_components
        else:
            raise ValueError('Too few components specified')

        # Optimization parameters
        self.init_params = init_params
        self.max_iter = max_iter
        self.tol = tol

        # Set prior hyperparameters
        self.set_prior_hyperparameters(D=num_channels,
                                       K=num_components)

    def set_prior_hyperparameters(self, D, K,
                                  a0=np.array([0.1]),
                                  b0=np.array([0.1]),
                                  n0=np.array([2.0]),
                                  m0=np.array([0.0]),
                                  W0=np.array([1.0])):
        """
        Set hyperparameters of prior distributions.

        Default prior hyperparameters are minimally informative symmetric
        parameters.

        Parameters
        ----------
        D : int
            Dimensionality of data.
        K : int
            Number of components.
        a0 : float / array (components by None)
            Hyperparameters of Dirichlet distribution on component weights.
        b0 : float / array (components by None)
            Scale parameters for hypermean normal distribution.
        n0 : array (components by None)
            Degrees of freedom for Wishart precision prior.
        m0 : array (components by dimensions)
            Hypermeans.
        W0 : array (dimensions by dimensions by components)
            Wishart precision parameters.

        Returns
        -------
        theta : tuple

        """
        # Expand alpha's if necessary
        if not a0.shape[0] == K:
            a0 = np.tile(a0[0], (K,))

        # Expand beta's if necessary
        if not b0.shape[0] == K:
            b0 = np.tile(b0[0], (K,))

        # Expand nu's if necessary
        if not n0.shape[0] == K:

            # Check for sufficient degrees of freedom
            if n0[0] < D:

                print('Cannot set Wishart degrees of freedom lower than data \
                      dimensionality.\n Setting it to data dim.')
                n0 = np.tile(D, (K,))

            else:
                n0 = np.tile(n0[0], (K,))

        # Expand hypermeans if necessary
        if not np.all(m0.shape == (K, D)):

            # If mean vector given, replicate to each component
            if len(m0.shape) == 2:
                if m0.shape[1] == D:
                    m0 = np.tile(m0, (K, 1))

            else:
                m0 = np.tile(m0[0], (K, D))

        # Expand hypermeans if necessary
        if not np.all(W0.shape == (D, D, K)):

            # If single covariance matrix given, replicate to each component
            if len(W0.shape) == 2:
                if np.all(m0.shape[:2] == (D, D)):
                    W0 = np.tile(W0, (1, 1, K))

            else:
                W0_ = np.zeros((D, D, K))
                for k in range(K):
                    W0_[:, :, k] = W0[0]*np.eye(D)

        # Store tupled parameters as model attribute
        self.theta0 = (a0, b0, n0, m0, W0_)

    def initialize_posteriors(self, X, Y):
        """
        Initialize posterior hyperparameters

        Parameters
        ----------
        X : array
            Observed image (height by width by channels)

        Returns
        -------
        theta : tuple
            Set of parameters.

        """
        # Current shape
        H, W, D = X.shape

        # Reshape arrays
        X = X.reshape((H*W, D))
        Y = Y.reshape((H*W, self.K))

        if self.init_params == 'random':

            # Dirichlet concentration hyperparameters
            at = np.ones((self.K,))*(H*W)/2

            # Normal precision-scale hyperparameters
            bt = np.ones((self.K,))*(H*W)/2

            # Wishart degrees of freedom
            nt = np.ones((self.K,))*(H*W)/2

            mt = np.zeros((self.K, D))
            Wt = np.zeros((D, D, self.K))
            for k in range(self.K):

                # Hypermeans
                mt[k, :] = np.mean(X, axis=0) + rnd.randn(1, D)*.1

                # Hyperprecisions
                Wt[:, :, k] = np.cov(X.T) / bt[k] + rnd.randn(D, D)*.1

            # Initialize variational posterior responsibilities
            rho = np.ones((H, W, self.K)) / self.K

        elif self.init_params in ('kmeans', 'k-means'):

            # Fit k-means to data and obtain cluster assignment
            label = KMeans(n_clusters=self.K, n_init=1).fit(X).labels_

            # Set rho based on cluster labels
            rho = np.zeros((H*W, self.K))
            rho[np.arange(H*W), label] = 1

            # Dirichlet concentration hyperparameters
            at = np.sum(rho, axis=0)

            # Normal precision-scale hyperparameters
            bt = np.sum(rho, axis=0)

            # Wishart degrees of freedom
            nt = np.sum(rho, axis=0)

            mt = np.zeros((self.K, D))
            Wt = np.zeros((D, D, self.K))
            for k in range(self.K):

                # Hypermeans
                mt[k, :] = np.sum(rho[:, [k]] * X, axis=0) / np.sum(rho[:, k])

                # Weighted covariance
                C = (rho[:, [k]] * (X - mt[k, :])).T @ (X - mt[k, :])

                # Check for zero precision
                C = np.maximum(1e-24, C)

                # Set hyperprecisions
                if D == 1:
                    Wt[:, :, k] = 1 / (nt[k] * C)
                else:
                    Wt[:, :, k] = inv(nt[k] * C)

            # Reshape responsibilities
            rho = rho.reshape((H, W, self.K))

        elif self.init_params in ('nn', 'knn'):

            # Observation indicator vector
            O = np.any(Y == 1, axis=1)

            if np.sum(O) == 0.0:
                raise ValueError('Cannot use \'nn\' without labels.')

            # Call instance of k-nearest neighbour classifier
            kNN = KNeighborsClassifier(n_neighbors=1, weights='distance')

            # Fit classifier to labeled data
            kNN.fit(X[O, :], np.argmax(Y[O, :], axis=1))

            # Set responsibilities based on kNN prediction
            rho = np.zeros((H*W, self.K))
            rho[~O, :] = kNN.predict_proba(X[~O, :])
            rho[O, :] = Y[O, :].astype('float64')

            # Concentration hyperparameters
            at = np.sum(rho, axis=0)

            # Precision-scale hyperparameters
            bt = np.sum(rho, axis=0)

            # Wishart degrees of freedom
            nt = np.sum(rho, axis=0)

            mt = np.zeros((self.K, D))
            Wt = np.zeros((D, D, self.K))
            for k in range(self.K):

                # Hypermeans
                mt[k, :] = X[Y[:, k] == 1, :]

                # Hyperprecisions
                Wt[:, :, k] = np.eye(D)

            # Reshape responsibilities
            rho = rho.reshape((H, W, self.K))

        else:
            raise ValueError('Provided method not recognized.')

        return (at, bt, nt, mt, Wt), rho

    def free_energy(self, X, rho, thetat, report=True):
        """
        Compute free energy term to monitor progress.

        Parameters
        ----------
        X : array
            Observed image (height by width by channels).
        rho : array
            Array of variational parameters (height by width by channels).
        thetat : array
            Parameters of variational posteriors.
        theta0 : array
            Parameters of variational priors.
        report : bool
            Print value of free energy function.

        Returns
        -------
        rho : array
            Updated array of variational parameters.

        """
        # Shapes
        H, W, D = X.shape

        # Reshape arrays
        X = X.reshape((H*W, D))
        rho = rho.reshape((H*W, self.K))

        # Unpack parameter sets
        a0, b0, n0, m0, W0 = self.theta0
        at, bt, nt, mt, Wt = thetat

        # Preallocate terms for energy function
        E1 = 0
        E2 = 0
        E3 = 0
        E4 = 0
        E5 = 0
        E6 = 0
        E7 = 0

        # Loop over classes
        for k in range(self.K):

            ''' Convenience variables '''

            # Proportion assigned to each component
            Nk = np.sum(rho[:, k], axis=0)

            # Responsibility-weighted mean
            xk = np.sum(rho[:, [k]] * X, axis=0) / Nk

            # Reponsibility-weighted variance
            Sk = ((X - xk) * rho[:, [k]]).T @ (X - xk) / Nk

            # Mahalanobis distance from hypermean
            mWm = (mt[k, :] - m0[k, :]).T @ Wt[:, :, k] @ (mt[k, :] - m0[k, :])

            # Mahalanobis distance from responsibility-weighted mean
            xWx = (xk - mt[k, :]) @ Wt[:, :, k] @ (xk - mt[k, :]).T

            # Entropy-based terms
            Elog_pik = digamma(at[k]) - digamma(np.sum(at))
            Elog_Lak = (D*np.log(2) + slogdet(Wt[:, :, k])[1] +
                        self.multidigamma(nt[k], D))

            ''' Energy function '''

            # First term
            E1 += Nk/2*(Elog_Lak - D / bt[k] -
                        nt[k]*(np.trace(Sk @ Wt[:, :, k]) + xWx) -
                        D*np.log(2*np.pi))

            # Second term
            E2 += np.sum(rho[:, k] * Elog_pik, axis=0)

            # Third term
            E3 += (a0[k] - 1)*Elog_pik + (gammaln(np.sum(a0)) -
                                          np.sum(gammaln(a0))) / self.K

            # Fourth term
            E4 += 1/2*(D*np.log(b0[k] / (2*np.pi)) +
                       Elog_Lak -
                       D*b0[k]/bt[k] -
                       b0[k]*nt[k]*mWm +
                       (n0[k] - D - 1)*Elog_Lak -
                       2*self.log_partition_Wishart(Wt[:, :, k], nt[k]) +
                       nt[k]*np.trace(inv(W0[:, :, k])*Wt[:, :, k]))

            # Ignore underflow error from log rho
            with np.errstate(under='ignore') and np.errstate(divide='ignore'):

                # Set -inf to most negative number
                lrho = np.maximum(np.log(rho[:, k]), np.finfo(rho.dtype).min)

                # Fifth term
                E5 += np.sum(rho[:, k] * lrho, axis=0)

            # Sixth term
            E6 += (at[k] - 1)*Elog_pik + (gammaln(np.sum(at)) -
                                          np.sum(gammaln(at))) / self.K

            # Seventh term
            E7 += (Elog_Lak/2 +
                   D/2*np.log(bt[k] / (2*np.pi)) -
                   D/2 - self.entropy_Wishart(Wt[:, :, k], nt[k]))

        # Compute free energy term
        F = E1 + E2 + E3 + E4 - E5 - E6 - E7

        # Print free energy
        if report:
            print('Free energy = ' + str(F))

        return F

    def expectation_step(self, X, Y, rho, thetat, savefn=''):
        """
        Perform expectation step.

        Parameters
        ----------
        X : array
            Observed image (height by width by channels).
        thetat : array
            Current iteration of parameters of variational posteriors.

        Returns
        -------
        rho : array
            Updated array of variational parameters / responsibilities.

        """
        # Shape of variational parameter array
        H, W, D = X.shape

        # Reshape arrays
        X = X.reshape((H*W, D))
        Y = Y.reshape((H*W, self.K))
        rho = rho.reshape((H*W, self.K))

        # Observation indicator vector
        O = np.any(Y != 0, axis=1)

        # Unpack tuple of hyperparameters
        at, bt, nt, mt, Wt = thetat

        # Initialize logarithmic rho
        lrho = np.zeros((np.sum(~O), self.K))

        for k in range(self.K):

            # Compute expected log mixing coefficient
            E1 = digamma(at[k]) - digamma(np.sum(at))

            # Compute exponentiated expected log precision
            E2 = (D*np.log(2) + slog_det(Wt[:, :, k])[1] +
                  self.multidigamma(nt[k], D))

            # Compute expected hypermean and hyperprecision
            E3 = D/bt[k] + self.distW(X[~O, :] - mt[k, :], nt[k]*Wt[:, :, k])

            # Update variational parameter at current pixels
            lrho[:, k] = E1 + E2/2 - E3/2

        # Subtract largest number from log_rho
        lrho = lrho - np.max(lrho, axis=1)[:, np.newaxis]

        # Exponentiate and normalize
        rho[~O, :] = np.exp(lrho) / np.sum(np.exp(lrho), axis=1)[:, np.newaxis]

        # Check for underflow problems
        if np.any(np.sum(rho, axis=1) == 0.0):
            raise RuntimeError('Variational parameter underflow.')

        return rho.reshape((H, W, self.K))

    def maximization_step(self, X, rho, thetat):
        """
        Perform maximization step from variational-Bayes-EM.

        Parameters
        ----------
        X : array
            Observed image (height by width by channels).
        rho : array
            Array of variational parameters (height by width by classes).
        thetat : array
            Current iteration of hyperparameters of posteriors.

        Returns
        -------
        thetat : array
            Next iteration of hyperparameters of posteriors.

        """
        # Shape of image
        H, W, D = X.shape

        # Reshape arrays
        X = X.reshape((H*W, D))
        rho = rho.reshape((H*W, self.K))

        # Unpack parameter sets
        a0, b0, n0, m0, W0 = self.theta0
        at, bt, nt, mt, Wt = thetat

        # Iterate over classes
        for k in range(self.K):

            # Total responsibility for class k
            Nk = np.sum(rho[:, k], axis=0)

            # Responsibility-weighted mean for class k
            xk = np.sum(rho[:, [k]] * X, axis=0) / Nk

            # Responsibility-weighted covariance for class k
            Sk = ((X - xk) * rho[:, [k]]).T @ (X - xk) / Nk

            # Update alpha
            at[k] = a0[k] + Nk

            # Update nu
            nt[k] = n0[k] + Nk

            # Update beta
            bt[k] = b0[k] + Nk

            # Update hypermean
            mt[k, :] = (b0[k]*m0[k, :] + Nk*xk) / (b0[k] + Nk)

            # Update hyperprecision
            Wt[:, :, k] = inv(inv(W0[:, :, k]) + Nk*Sk + (b0[k]*Nk) / bt[k] *
                              (xk - m0[k, :]).T @ (xk - m0[k, :]))

        return at, bt, nt, mt, Wt

    def expectation_maximization(self, X, Y):
        """
        Perform Variational Bayes Expectation-Maximization.

        Parameters
        ----------
        X : array (instances by features)
            Data array.
        Y : array
            Observed labels (height by width by classes).

        Returns
        -------
        rho : array (instances by components)
            Variational parameters of posterior for label image.

        """
        # Get shape of image
        H, W, D = X.shape

        # Initialize posterior hyperparameters
        thetat, rho = self.initialize_posteriors(X, Y)

        # Initialize old energy variable
        F_ = np.inf

        for t in range(self.max_iter):

            # Monitor progress every tenth iteration
            if t % (self.max_iter/10) == 0:

                # Report progress
                print('Iteration ' + '{0:03}'.format(t+1) + '/' +
                      str(self.max_iter) + '\t', end='')

                # Compute free energy to monitor progress
                F = self.free_energy(X, rho, thetat, report=True)

                if np.abs(F - F_) <= self.tol:
                    print('Step size is below tolerance threshold.')
                    break

                # Update old energy
                F_ = F

            # Expectation step
            rho = self.expectation_step(X, Y, rho, thetat)

            # Expectation step
            thetat = self.maximization_step(X, rho, thetat)

        # Return segmentation along with estimated parameters
        return rho, thetat

    def segment(self, X, Y=None):
        """
        Fit model to data and segment image.

        Parameters
        ----------
        X : array.
            Observed image (height by width by channels).
        Y : array
            Observed labels (height by width by classes).

        Returns
        -------
        pred : array
            Segmentation produced by the model.
        post : array
            Posterior indicator distributions.
        theta : tuple of arrays
            Posterior hyperparameters of parameter distributions.

        """
        # Check shape of image
        H, W, D = X.shape

        # Check if dimensionality of given data matches prior dimensionality.
        if not self.D == D:

            # Report
            print('Re-setting priors.')

            # Set dimensionality attribute
            self.D = D

            # Set prior hyperparameters
            self.set_prior_hyperparameters(D=D, K=self.K)

        # If no Y is provided
        if Y is None:
            Y = np.zeros((H, W, self.K))

        # Perform VB-EM for segmenting the image
        post, params = self.expectation_maximization(X, Y)

        # Compute most likely class
        pred = np.argmax(post, axis=2)

        # Return segmented image, variational posteriors and parameters
        return pred, post, params


class VariationalHiddenPotts(VariationalMixture):
    """
    Variational Gaussian Mixture model with a hidden Potts-Markov Random Field.

    This implementation multivariate images (height by width by channel).
    It is based on the RPubs note by Andreas Kapourani:
    https://rpubs.com/cakapourani/variational-bayes-gmm
    and the paper "Variational Bayes for estimating the parameters of a hidden
    Potts model" by McGrory, Titterington, Reeves and Pettitt (2008).
    """

    def __init__(self, num_channels=1,
                 num_components=2,
                 tissue_specific=False,
                 init_params='kmeans',
                 max_iter=10,
                 tol=1e-5):
        """
        Model-specific constructors.

        Parameters
        ----------
        num_channels : int
            Number of channels of image (def: 1).
        num_components : int
            Number of components (def: 2).
        tissue_specific : bool
            Whether to model each tissue's smoothness separately (def: False).
        init_params : str
            How to initialize posterior parameters.
            Options: 'random', 'kmeans', 'nn' (def: 'kmeans').
        max_iter : int
            Maximum number of iterations to run for (def: 10).
        tol : float
            Tolerance on change in x-value (def: 1e-5).

        Returns
        -------
        None

        """
        # Store data dimensionality
        if num_channels >= 1:
            self.D = num_channels
        else:
            raise ValueError('Number of channels must be larger than 0.')

        # Store model parameters
        if num_components >= 2:
            self.K = num_components
        else:
            raise ValueError('Too few components specified')

        # Whether to use tissue-specific smoothness parameters
        self.Bk = tissue_specific

        # Optimization parameters
        self.init_params = init_params
        self.max_iter = max_iter
        self.tol = tol

        # Set prior hyperparameters
        self.set_prior_hyperparameters(D=num_channels,
                                       K=num_components)

    def set_prior_hyperparameters(self, D, K,
                                  a0=np.array([0.1]),
                                  b0=np.array([0.1]),
                                  n0=np.array([2.0]),
                                  m0=np.array([0.0]),
                                  W0=np.array([1.0])):
        """
        Set hyperparameters of prior distributions.

        Default prior hyperparameters are minimally informative symmetric
        parameters.

        Parameters
        ----------
        D : int
            Dimensionality of data.
        K : int
            Number of components.
        a0 : float / array (components by None)
            Hyperparameters of Dirichlet distribution on component weights.
        b0 : float / array (components by None)
            Scale parameters for hypermean normal distribution.
        n0 : array (components by None)
            Degrees of freedom for Wishart precision prior.
        m0 : array (components by dimensions)
            Hypermeans.
        W0 : array (dimensions by dimensions by components)
            Wishart precision parameters.

        Returns
        -------
        theta : tuple

        """
        # Expand alpha's if necessary
        if not a0.shape[0] == K:
            a0 = np.tile(a0[0], (K,))

        # Expand beta's if necessary
        if not b0.shape[0] == K:
            b0 = np.tile(b0[0], (K,))

        # Expand nu's if necessary
        if not n0.shape[0] == K:

            # Check for sufficient degrees of freedom
            if n0[0] < D:

                print('Cannot set Wishart degrees of freedom lower than data \
                      dimensionality.\n Setting it to data dim.')
                n0 = np.tile(D, (K,))

            else:
                n0 = np.tile(n0[0], (K,))

        # Expand hypermeans if necessary
        if not np.all(m0.shape == (K, D)):

            # If mean vector given, replicate to each component
            if len(m0.shape) == 2:
                if m0.shape[1] == D:
                    m0 = np.tile(m0, (K, 1))

            else:
                m0 = np.tile(m0[0], (K, D))

        # Expand hypermeans if necessary
        if not np.all(W0.shape == (D, D, K)):

            # If single covariance matrix given, replicate to each component
            if len(W0.shape) == 2:
                if np.all(m0.shape[:2] == (D, D)):
                    W0 = np.tile(W0, (1, 1, K))

            else:
                W0_ = np.zeros((D, D, K))
                for k in range(K):
                    W0_[:, :, k] = W0[0]*np.eye(D)

        # Store tupled parameters as model attribute
        self.theta0 = (a0, b0, n0, m0, W0_)

    def initialize_posteriors(self, X, Y):
        """
        Initialize posterior hyperparameters

        Parameters
        ----------
        X : array
            Observed image (height by width by channels)

        Returns
        -------
        theta : tuple
            Set of parameters.

        """
        # Current shape
        H, W, D = X.shape

        # Reshape arrays
        X = X.reshape((H*W, D))
        Y = Y.reshape((H*W, self.K))

        if self.init_params == 'random':

            # Dirichlet concentration hyperparameters
            at = np.ones((self.K,))*(H*W)/2

            # Normal precision-scale hyperparameters
            bt = np.ones((self.K,))*(H*W)/2

            # Wishart degrees of freedom
            nt = np.ones((self.K,))*(H*W)/2

            mt = np.zeros((self.K, D))
            Wt = np.zeros((D, D, self.K))
            for k in range(self.K):

                # Hypermeans
                mt[k, :] = np.mean(X, axis=0) + rnd.randn(1, D)*.1

                # Hyperprecisions
                Wt[:, :, k] = np.cov(X.T) / bt[k] + rnd.randn(D, D)*.1

            # Initialize variational posterior responsibilities
            rho = np.ones((H, W, self.K)) / self.K

        elif self.init_params in ('kmeans', 'k-means'):

            # Fit k-means to data and obtain cluster assignment
            label = KMeans(n_clusters=self.K, n_init=1).fit(X).labels_

            # Set rho based on cluster labels
            rho = np.zeros((H*W, self.K))
            rho[np.arange(H*W), label] = 1

            # Dirichlet concentration hyperparameters
            at = np.sum(rho, axis=0)

            # Normal precision-scale hyperparameters
            bt = np.sum(rho, axis=0)

            # Wishart degrees of freedom
            nt = np.sum(rho, axis=0)

            mt = np.zeros((self.K, D))
            Wt = np.zeros((D, D, self.K))
            for k in range(self.K):

                # Hypermeans
                mt[k, :] = np.sum(rho[:, [k]] * X, axis=0) / np.sum(rho[:, k])

                # Hyperprecisions
                Wt[:, :, k] = np.eye(D)

            # Reshape responsibilities
            rho = rho.reshape((H, W, self.K))

        elif self.init_params in ('nn', 'knn'):

            # Observation indicator vector
            O = np.any(Y == 1, axis=1)

            if np.sum(O) == 0.0:
                raise ValueError('Cannot use \'nn\' without labels.')

            # Call instance of k-nearest neighbour classifier
            kNN = KNeighborsClassifier(n_neighbors=1)

            # Fit classifier to labeled data
            kNN.fit(X[O, :], np.argmax(Y[O, :], axis=1))

            # Set responsibilities based on kNN prediction
            rho = np.zeros((H*W, self.K))
            rho[~O, :] = kNN.predict_proba(X[~O, :])
            rho[O, :] = Y[O, :].astype('float64')

            # Concentration hyperparameters
            at = np.sum(rho, axis=0)

            # Precision-scale hyperparameters
            bt = np.sum(rho, axis=0)

            # Wishart degrees of freedom
            nt = np.sum(rho, axis=0)

            mt = np.zeros((self.K, D))
            Wt = np.zeros((D, D, self.K))
            for k in range(self.K):

                # Focuse hypermeans on labeled pixels
                mt[k, :] = np.mean(X[Y[:, k] == 1, :], axis=0)

                # Hyperprecisions
                Wt[:, :, k] = np.eye(D)

            # Reshape responsibilities
            rho = rho.reshape((H, W, self.K))

        else:
            raise ValueError('Provided method not recognized.')

        return (at, bt, nt, mt, Wt), rho

    def neighbourhood(self, A, index, pad=False):
        """
        Extract a neighbourhood of pixels around current pixel.

        Parameters
        ----------
        A : array
            Array from which to extract the pixel's neighbourhood.
        index : [int, int]
            Row and column index of current pixel.
        pad : bool
            Whether to pad a border with zeros to the array (def: False)

        Returns
        -------
        delta_i : vector of neighbours of current pixel

        """
        if pad:

            # Check for list index
            if not isinstance(index, list):
                raise ValueError('If padding, index must be a list.')

            # Pad array with zeros
            A = np.pad(A, [1, 1], mode='constant', constant_values=0)

            # Correct index due to padding
            index[0] += 1
            index[1] += 1

        # Preallocate neighbourhood list
        delta_i = []

        # Left
        delta_i.append(A[index[0]-1, index[1]])

        # Top
        delta_i.append(A[index[0], index[1]-1])

        # Right
        delta_i.append(A[index[0]+1, index[1]])

        # Bottom
        delta_i.append(A[index[0], index[1]+1])

        return np.array(delta_i)

    def mean_field_Potts(self, beta, Z):
        r"""
        Mean-field variational approximation to Potts log-likelihood function.

        logp(z_i | z_{d_i}, beta) = \sum_{k=1}^{K} beta_k * z_{ik}
            \sum_{j \in \delta_{ik}} z_{jk} - \log \sum_{z_{i'}}
            \exp(\sum_{k=1}^{K} beta_k*z_{i'k} \sum_{j \in \delta_{ik}} z_{jk})

        Parameters
        ----------
        beta : float
            Smoothing parameter / granularity coefficient
        Z : array (height x width x number of classes)
            Label field to fit

        Returns
        -------
        nlogq : float
            Negative log-likelihood of current label field given current beta.

        """
        # Shape
        H, W, K = Z.shape

        # Test for binary label image
        for k in range(K):
            if not np.all(np.unique(Z[:, :, k]) == [0, 1]):
                raise ValueError("Label field is not binary in page " + str(k))

        # Pad array to avoid repeated padding during neighbourhood extraction
        Z0 = np.pad(Z, [(1, 1), (1, 1), (0, 0)],
                    mode='constant',
                    constant_values=0)

        if self.Bk:

            # Initialize intermediate terms
            chi_i = 0
            ksi_i = 0

            # Select current class
            for k in range(K):

                # Extract neighbourhoods
                d_ik = extract_patches_2d(Z0[:, :, k], patch_size=(3, 3))

                # Compute sum over neighbourhood
                d_ik = np.sum(d_ik, axis=(1, 2))

                # First sum is neighbourhood comparison
                chi_i += beta[k] * Z[:, :, k].reshape((-1,)) * d_ik

                # Second sum is purely over neighbourhood
                ksi_i += np.exp(beta[k] * d_ik)

            # Update negative log-likelihood
            nll = np.sum(-chi_i + np.log(ksi_i), axis=0)

            return nll

        else:

            # Initialize intermediate terms
            chi_i = 0
            ksi_i = 0

            # Select current class
            for k in range(K):

                # Extract neighbourhoods
                d_ik = extract_patches_2d(Z0[:, :, k], patch_size=(3, 3))

                # Compute sum over neighbourhood
                d_ik = np.sum(d_ik, axis=(1, 2))

                # First sum is neighbourhood comparison
                chi_i += beta * Z[:, :, k].reshape((-1,)) * d_ik

                # Second sum is purely over neighbourhood
                ksi_i += np.exp(beta * d_ik)

            # Update negative log-likelihood
            nll = np.sum(-chi_i + np.log(ksi_i), axis=0)

            return nll

    def mean_field_Potts_grad(self, beta, Z):
        r"""
        Partial derivative of mean-field Potts log-likelihood w.r.t. beta.

        Derivative has the following form:

        d/db log q(z|b) = \sum_{k=1}^{K} z_{ik} \sum_{j \in \delta_{ik}} z_{jk}
            - \log \sum_{k} \exp(beta_k * \sum_{j \in \delta_{ik}} z_{jk})

        Parameters
        ----------
        beta : array(float)
            Smoothing parameters / granularity coefficients.
        Z : array (height by width by number of classes)
            Label field to fit.

        Returns
        -------
        dB : float
            Value of partial derivative for current beta.

        """
        # Shape
        H, W, K = Z.shape

        # Test for binary label images
        for k in range(K):
            if not np.all(np.unique(Z[:, :, k]) == [0, 1]):
                raise ValueError("Label field is not binary in page " + str(k))

        # Pad array to avoid repeated padding during neighbourhood extraction
        Z0 = np.pad(Z, [(1, 1), (1, 1), (0, 0)],
                    mode='constant',
                    constant_values=0)

        if self.Bk:

            # Extract neighbourhood
            d_ik = np.zeros((H*W, self.K))

            # Preallocate
            dqdb = np.zeros((self.K, ))
            ksi_i = 0

            # Compute denominator first
            for k in range(K):

                # Extract neighbourhoods
                neighbourhoods = extract_patches_2d(Z0[:, :, k],
                                                    patch_size=(3, 3))

                # Compute sum over neighbourhood
                d_ik[:, k] = np.sum(neighbourhoods, axis=(1, 2))

                # Denominator term
                ksi_i += np.exp(beta[k]*d_ik[:, k])

            # Loop over classes
            for k in range(K):

                # First term
                chi_i = Z[:, :, k].reshape((-1, )) * d_ik[:, k]

                # Numerator
                psi_i = np.exp(beta[k]*d_ik[:, k]) * d_ik[:, k]

                # Update partial derivative
                dqdb[k] += np.sum(-chi_i + psi_i / ksi_i, axis=0)

            return dqdb

        else:

            # Extract neighbourhood
            d_ik = np.zeros((H*W, self.K))

            # Preallocate
            dqdb = np.zeros((self.K, ))
            ksi_i = 0

            # Compute denominator first
            for k in range(K):

                # Extract neighbourhoods
                neighbourhoods = extract_patches_2d(Z0[:, :, k],
                                                    patch_size=(3, 3))

                # Compute sum over neighbourhood
                d_ik[:, k] = np.sum(neighbourhoods, axis=(1, 2))

                # Denominator term
                ksi_i += np.exp(beta * d_ik[:, k])

            # Loop over classes
            for k in range(K):

                # First term
                chi_i = Z[:, :, k].reshape((-1, )) * d_ik[:, k]

                # Numerator
                psi_i = np.exp(beta * d_ik[:, k]) * d_ik[:, k]

                # Update partial derivative
                dqdb[k] += np.sum(-chi_i + psi_i / ksi_i, axis=0)

            return dqdb

    def maximum_likelihood_beta(self, Z,
                                ub=None,
                                verbose=True,
                                max_iter=5):
        """
        Estimate beta on mean-field Potts likelihood.

        Parameters
        ----------
        Z : array (height x width x number of classes)
            Label field
        lb : [(int, int)]
            List of tuple of integers describing the lower and upper bound for
            the smoothing parameter.
        verbose : bool
            Report final beta estimate.
        max_iter : int
            Maximum number of iterations for EM.

        Returns
        -------
        beta_hat : float
            Estimated beta, or smoothing parameter, for given label field.

        """
        # Check if Z is the right shape
        if not len(Z.shape) == 3:
            raise ValueError('One-hot encoding of array necessary.')

        # Check for tissue-specific
        if self.Bk:

            # Set initial value
            beta0 = np.ones((self.K, ))

            # Set upper bounds
            lb = [(0, ub) for k in range(self.K)]

        else:

            # Set initial value
            beta0 = np.array([1.0])

            # Set bounds
            lb = [(0, ub)]

        # Start optimization procedure
        beta_h = minimize(fun=self.mean_field_Potts,
                          x0=beta0,
                          args=Z,
                          method='L-BFGS-B',
                          jac=self.mean_field_Potts_grad,
                          bounds=lb,
                          options={'disp': verbose, 'maxiter': max_iter})

        # Check value
        if np.any(beta_h.x > 1e2):
            print('Warning: beta_hat is very large.')

        if self.Bk:
            return beta_h.x
        else:
            return beta_h.x[0]

    def free_energy(self, X, rho, thetat, report=True):
        """
        Compute free energy term to monitor progress.

        Parameters
        ----------
        X : array
            Observed image (height by width by channels).
        rho : array
            Array of variational parameters (height by width by channels).
        thetat : array
            Parameters of variational posteriors.
        theta0 : array
            Parameters of variational priors.
        report : bool
            Print value of free energy function.

        Returns
        -------
        rho : array
            Updated array of variational parameters.

        """
        # Shapes
        H, W, D = X.shape

        # Reshape arrays
        X = X.reshape((H*W, D))
        rho = rho.reshape((H*W, self.K))

        # Unpack parameter sets
        a0, b0, n0, m0, W0 = self.theta0
        at, bt, nt, mt, Wt = thetat

        # Preallocate terms for energy function
        E1 = 0
        E2 = 0
        E3 = 0
        E4 = 0
        E5 = 0
        E6 = 0
        E7 = 0

        # Loop over classes
        for k in range(self.K):

            ''' Convenience variables '''

            # Proportion assigned to each component
            Nk = np.sum(rho[:, k], axis=0)

            # Responsibility-weighted mean
            xk = np.sum(rho[:, [k]] * X, axis=0) / Nk

            # Reponsibility-weighted variance
            Sk = ((X - xk) * rho[:, [k]]).T @ (X - xk) / Nk

            # Mahalanobis distance from hypermean
            mWm = (mt[k, :] - m0[k, :]).T @ Wt[:, :, k] @ (mt[k, :] - m0[k, :])

            # Mahalanobis distance from responsibility-weighted mean
            xWx = (xk - mt[k, :]) @ Wt[:, :, k] @ (xk - mt[k, :]).T

            # Entropy-based terms
            Elog_pik = digamma(at[k]) - digamma(np.sum(at))
            Elog_Lak = (D*np.log(2) +
                        slogdet(Wt[:, :, k])[1] +
                        self.multidigamma(nt[k], D))

            ''' Energy function '''

            # First term
            E1 += Nk/2*(Elog_Lak - D / bt[k] -
                        nt[k]*(np.trace(Sk @ Wt[:, :, k]) + xWx) -
                        D*np.log(2*np.pi))

            # Second term
            E2 += np.sum(rho[:, k] * Elog_pik, axis=0)

            # Third term
            E3 += (a0[k] - 1)*Elog_pik + (gammaln(np.sum(a0)) -
                                          np.sum(gammaln(a0))) / self.K

            # Fourth term
            E4 += 1/2*(D*np.log(b0[k] / (2*np.pi)) +
                       Elog_Lak -
                       D*b0[k]/bt[k] -
                       b0[k]*nt[k]*mWm +
                       (n0[k] - D - 1)*Elog_Lak -
                       2*self.log_partition_Wishart(Wt[:, :, k], nt[k]) +
                       nt[k]*np.trace(inv(W0[:, :, k])*Wt[:, :, k]))

            # Ignore underflow error from log rho
            with np.errstate(under='ignore') and np.errstate(divide='ignore'):

                # Set -inf to most negative number
                lrho = np.maximum(np.log(rho[:, k]), np.finfo(rho.dtype).min)

                # Fifth term
                E5 += np.sum(rho[:, k] * lrho, axis=0)

            # Sixth term
            E6 += (at[k] - 1)*Elog_pik + (gammaln(np.sum(at)) -
                                          np.sum(gammaln(at))) / self.K

            # Seventh term
            E7 += (Elog_Lak/2 +
                   D/2*np.log(bt[k] / (2*np.pi)) -
                   D/2 - self.entropy_Wishart(Wt[:, :, k], nt[k]))

        # Compute free energy term
        F = E1 + E2 + E3 + E4 - E5 - E6 - E7

        # Print free energy
        if report:
            print('Free energy = ' + str(F))

        return F

    def expectation_step(self, X, Y, rho, thetat, beta):
        """
        Perform expectation step.

        Parameters
        ----------
        X : array
            Observed image (height by width by channels).
        thetat : array
            Current iteration of parameters of variational posteriors.

        Returns
        -------
        rho : array
            Updated array of variational parameters / responsibilities.

        """
        # Shape of variational parameter array
        H, W, D = X.shape

        # Observation indicator vector
        M = np.all(~Y, axis=2)

        # Pad variational parameter array, to avoid repeated padding
        rho0 = np.pad(rho, [(1, 1), (1, 1), (0, 0)],
                      mode='constant', constant_values=0)

        # Unpack tuple of hyperparameters
        at, bt, nt, mt, Wt = thetat

        # Initialize logarithmic rho
        lrho = np.zeros((H, W, self.K), dtype='float64')

        for k in range(self.K):

            # Compute expected log mixing coefficient
            E1 = digamma(at[k]) - digamma(np.sum(at))

            # Compute exponentiated expected log precision
            E2 = (slogdet(Wt[:, :, k])[1] + self.multidigamma(nt[k], D))/2

            # Compute expected hypermean and hyperprecision
            E3 = -(D/bt[k] + self.distW(X - mt[k, :], nt[k]*Wt[:, :, k]))/2

            # Extract neighbourhoods
            d_ik = extract_patches_2d(rho0[:, :, k], patch_size=(3, 3))

            # Compute sum over neighbourhood
            d_ik = np.sum(d_ik, axis=(1, 2)).reshape((H, W))

            # Compute Potts regularizer
            if self.Bk:
                E4 = beta[k] * d_ik
            else:
                E4 = beta * d_ik

            # Update variational parameter at current pixels
            lrho[:, :, k] = E1 + E2 + E3 + E4

        # Subtract largest number from log_rho
        lrho[M, :] = lrho[M, :] - np.max(lrho[M, :], axis=1)[:, np.newaxis]

        # Exponentiate and normalize
        rho[M, :] = (np.exp(lrho[M, :]) /
                     np.sum(np.exp(lrho[M, :]), axis=1)[:, np.newaxis])

        # Check for underflow problems
        if np.any(np.abs(np.sum(rho, axis=2) - 1.0) > 1e-12):
            raise RuntimeError('Variational parameter underflow.')

        return rho

    def maximization_step(self, X, rho, thetat):
        """
        Perform maximization step from variational-Bayes-EM.

        Parameters
        ----------
        X : array
            Observed image (height by width by channels).
        rho : array
            Array of variational parameters (height by width by classes).
        thetat : array
            Current iteration of hyperparameters of posteriors.

        Returns
        -------
        thetat : array
            Next iteration of hyperparameters of posteriors.

        """
        # Shape of image
        H, W, D = X.shape

        # Reshape arrays
        X = X.reshape((H*W, D))
        rho = rho.reshape((H*W, self.K))

        # Unpack parameter sets
        a0, b0, n0, m0, W0 = self.theta0
        at, bt, nt, mt, Wt = thetat

        # Iterate over classes
        for k in range(self.K):

            # Total responsibility for class k
            Nk = np.sum(rho[:, k], axis=0)

            # Responsibility-weighted mean for class k
            xk = np.sum(rho[:, [k]] * X, axis=0) / Nk

            # Responsibility-weighted covariance for class k
            Sk = ((X - xk) * rho[:, [k]]).T @ (X - xk) / Nk

            # Update alpha
            at[k] = a0[k] + Nk

            # Update nu
            nt[k] = n0[k] + Nk

            # Update beta
            bt[k] = b0[k] + Nk

            # Update hypermean
            mt[k, :] = (b0[k]*m0[k, :] + Nk*xk) / (b0[k] + Nk)

            # Update hyperprecision
            Wt[:, :, k] = inv(inv(W0[:, :, k]) + Nk*Sk + (b0[k]*Nk) / bt[k] *
                              (xk - m0[k, :]).T @ (xk - m0[k, :]))

        return at, bt, nt, mt, Wt

    def expectation_maximization(self, X, Y, beta):
        """
        Perform Variational Bayes Expectation-Maximization.

        Parameters
        ----------
        X : array (instances by features)
            Data array.
        Y : array
            Observed labels (height by width by classes).
        beta : array
            Tissue smoothness parameter(s).

        Returns
        -------
        rho : array (instances by components)
            Variational parameters of posterior for label image.

        """
        # Get shape of image
        H, W, D = X.shape

        # Initialize posterior hyperparameters
        thetat, rho = self.initialize_posteriors(X, Y)

        # Initialize old energy variable
        F_ = np.inf

        for t in range(self.max_iter):

            # Monitor progress every tenth iteration
            if t % (self.max_iter/10) == 0:

                # Report progress
                print('Iteration ' + '{0:03}'.format(t+1) + '/' +
                      str(self.max_iter) + '\t', end='')

                # Compute free energy to monitor progress
                F = self.free_energy(X, rho, thetat, report=True)

                if np.abs(F - F_) <= self.tol:
                    print('Step size is below tolerance threshold.')
                    break

                # Update old energy
                F_ = F

            # Expectation step
            rho = self.expectation_step(X, Y, rho, thetat, beta)

            # Expectation step
            thetat = self.maximization_step(X, rho, thetat)

        # Return segmentation along with estimated parameters
        return rho, thetat

    def segment(self, X, Y=None, beta=1.0):
        """
        Fit model to data and segment image.

        Parameters
        ----------
        X : array.
            Observed image (height by width by channels).
        Y : array
            Observed labels (height by width by classes).
        beta : array
            Smoothness parameter(s).

        Returns
        -------
        pred : array
            Segmentation produced by the model.
        post : array
            Posterior indicator distributions.
        theta : tuple of arrays
            Posterior hyperparameters of parameter distributions.

        """
        # Check shape of image
        H, W, D = X.shape

        # Check if dimensionality of given data matches prior dimensionality.
        if not self.D == D:

            # Report
            print('Re-setting priors.')

            # Set dimensionality attribute
            self.D = D

            # Set prior hyperparameters
            self.set_prior_hyperparameters(D=D, K=self.K)

        # Check for tissue-specific
        if len(beta) > 1:
            self.tissue_specific = True

        # If no Y is provided
        if Y is None:
            Y = np.zeros((H, W, self.K), dtype='bool')

        # Perform VB-EM for segmenting the image
        post, params = self.expectation_maximization(X, Y, beta)

        # Compute most likely class
        pred = np.argmax(post, axis=2)

        # Return segmented image, variational posteriors and parameters
        return pred, post, params
