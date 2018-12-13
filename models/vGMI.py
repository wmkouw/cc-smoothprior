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

from numpy.linalg import inv, cholesky
from scipy.misc import logsumexp
from scipy.special import betaln, digamma, gammaln
from scipy.spatial.distance import cdist

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
from vis import plot_posteriors


class VariationalMixture(object):
    """
    Superclass of variational mixture models.

    Methods are functions common to all submixture models.
    """

    def log_multivariate_gamma(self, n, p):
        """
        Logarithmic multivariate gamma function.

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
        Gp : float
            p-th order multivariate gamma function.

        """
        # Check for appropriate degree of freedom
        if not n > (p-1):
            raise ValueError('Degrees of freedom too low for dimensionality.')

        # Preallocate
        Gp = 0

        # Product from d=1 to p
        for d in range(1, p+1):

            # Gamma function of degrees of freedom and dimension
            Gp += gammaln((n + 1 - d)/2)

        return (p * (p-1) / 4)*np.log(np.pi) + Gp

    def multivariate_digamma(self, n, p):
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

        # Compute log-multivariate gamma
        lmG = self.log_multivariate_gamma(n, D)

        # Compute partition function
        B = (-n/2)*self.log_det(W) - (n*D/2)*np.log(2) - lmG

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
        E = self.multivariate_digamma(n, D) + D*np.log(2) + self.log_det(W)

        # Entropy
        H = -self.log_partition_Wishart(W, n) - (n - D - 1)/2 * E + n*D/2

        return H

    def log_det(self, A):
        """
        Numerically stable computation of log determinant of a matrix.

        Parameters
        ----------
        A : array
            Expecting a positive definite, symmetric matrix.

        Returns
        -------
        float
            Log-determinant of given matrix.

        """
        # Perform cholesky decomposition
        L = cholesky(A)

        # Stable log-determinant
        return np.sum(2*np.log(np.diag(L)))

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


class UnsupervisedGaussianMixture(VariationalMixture):
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

    def initialize_posteriors(self, X):
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
                Wt[:, :, k] = np.eye(D)

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
            Elog_Lak = (D*np.log(2) +
                        self.log_det(Wt[:, :, k]) +
                        self.multivariate_digamma(nt[k], D))

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

    def expectation_step(self, X, thetat, savefn=''):
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

        # Unpack tuple of hyperparameters
        at, bt, nt, mt, Wt = thetat

        # Initialize logarithmic rho
        log_rho = np.zeros((H*W, self.K))

        for k in range(self.K):

            # Compute expected log mixing coefficient
            E1 = digamma(at[k]) - digamma(np.sum(at))

            # Compute exponentiated expected log precision
            E2 = (D*np.log(2) + self.log_det(Wt[:, :, k]) +
                  self.multivariate_digamma(nt[k], D))

            # Compute expected hypermean and hyperprecision
            E3 = D/bt[k] + self.distW(X - mt[k, :], nt[k]*Wt[:, :, k])

            # Update variational parameter at current pixels
            log_rho[:, k] = E1 + E2/2 - E3/2

        # Subtract largest number from log_rho
        log_rho = log_rho - np.max(log_rho, axis=1)[:, np.newaxis]

        # Exponentiate and normalize
        rho = np.exp(log_rho) / np.sum(np.exp(log_rho), axis=1)[:, np.newaxis]

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

    def expectation_maximization(self, X):
        """
        Perform Variational Bayes Expectation-Maximization.

        Parameters
        ----------
        X : array (instances by features)
            Data array.

        Returns
        -------
        rho : array (instances by components)
            Variational parameters of posterior for label image.

        """
        # Get shape of image
        H, W, D = X.shape

        # Initialize posterior hyperparameters
        thetat, rho = self.initialize_posteriors(X)

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
            rho = self.expectation_step(X, thetat, savefn=('rho_t' + str(t)))

            # Expectation step
            thetat = self.maximization_step(X, rho, thetat)

        # Return segmentation along with estimated parameters
        return rho, thetat

    def segment(self, X):
        """
        Fit model to data and segment image.

        Parameters
        ----------
        X : array.
            Observed image (height by width by channels).

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

        # Perform VB-EM for segmenting the image
        post, params = self.expectation_maximization(X)

        # Compute most likely class
        pred = np.argmax(post, axis=2)

        # Return segmented image, variational posteriors and parameters
        return pred, post, params


class SemisupervisedGaussianMixture(VariationalMixture):
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

        # Observation indicator vector
        O = np.any(Y == 1, axis=1)

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
                Wt[:, :, k] = np.eye(D)

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

                # Hypermean
                mt[k, :] = np.mean(X[Y[:, k] == 1, :], axis=0)

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
            Elog_Lak = (D*np.log(2) +
                        self.log_det(Wt[:, :, k]) +
                        self.multivariate_digamma(nt[k], D))

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

    def expectation_step(self, X, Y, rho, thetat):
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

        # # Reshape arrays
        # X = X.reshape((H*W, D))
        # Y = Y.reshape((H*W, self.K))
        # rho = rho.reshape((H*W, self.K))

        # Observation indicator vector
        M = np.all(Y == False, axis=2)

        # Unpack tuple of hyperparameters
        at, bt, nt, mt, Wt = thetat

        # Initialize logarithmic rho
        lrho = np.zeros((H, W, self.K), dtype='float64')

        for k in range(self.K):

            # Compute expected log mixing coefficient
            E1 = digamma(at[k]) - digamma(np.sum(at))

            # Compute exponentiated expected log precision
            E2 = (D*np.log(2) + self.log_det(Wt[:, :, k]) + self.multivariate_digamma(nt[k], D))

            # Compute expected hypermean and hyperprecision
            E3 = D/bt[k] + self.distW(X - mt[k, :], nt[k]*Wt[:, :, k])

            # Update variational parameter at current pixels
            lrho[:, :, k] = E1 + E2/2 - E3/2

        # Subtract largest number from log_rho
        lrho[M, :] = lrho[M, :] - np.max(lrho[M, :], axis=1)[:, np.newaxis]

        # Exponentiate and normalize
        rho[M, :] = np.exp(lrho[M, :]) / np.sum(np.exp(lrho[M, :]), axis=1)[:, np.newaxis]

        # Check for underflow problems
        if np.any(np.abs(np.sum(rho, axis=2) - 1.0) > 1e-12):
            raise RuntimeError('Variational parameter underflow.')

        return rho#.reshape((H, W, self.K))

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

    def segment(self, X, Y):
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

        # Perform VB-EM for segmenting the image
        post, params = self.expectation_maximization(X, Y)

        # Compute most likely class
        pred = np.argmax(post, axis=2)

        # Return segmented image, variational posteriors and parameters
        return pred, post, params
