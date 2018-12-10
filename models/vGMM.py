#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Class of variatonal Gaussian Mixture Models.

It serves as a baseline for a hidden Potts-MRF for Bayesian unsupervised image
segmentation.

Author: W.M. Kouw
Date: 05-11-2018
"""
import math
import numpy as np
import numpy.random as rnd

from numpy.linalg import inv, cholesky
from scipy.misc import logsumexp
from scipy.special import betaln, digamma, gammaln
from scipy.spatial.distance import cdist

import numpy.linalg as alg
import scipy.special as sp
import scipy.optimize as opt
import sklearn.cluster as cl


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
            Pp += sp.digamma((n + 1 - d)/2)

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
        L = alg.cholesky(A)

        # Stable log-determinant
        return np.sum(2*np.log(np.diag(L)))

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
    Variational Gaussian Mixture Model.

    This implementation is based on the RPubs note by Andreas Kapourani:
    https://rpubs.com/cakapourani/variational-bayes-gmm
    It models multivariate samples.
    """

    def __init__(self, data_dimensionality=2,
                 num_components=2,
                 theta=(0, 0, 0, 0, 1),
                 max_iter=10,
                 tol=1e-5):
        """
        Model-specific constructors.

        Parameters
        ----------
        data_dimensionality : int
            Number of dimensions of the data.
        num_components : int
            Number of components.
        theta0 : tuple
            Prior hyperparameters.
        max_iter : int
            Maximum number of iterations to run for.
        tol : float
            Tolerance on change in x-value.

        Returns
        -------
        None

        """
        # Store data dimensionality
        if data_dimensionality >= 1:
            self.D = data_dimensionality
        else:
            raise ValueError('Dimensionality must be larger than 0.')

        # Store model parameters
        if num_components >= 2:
            self.K = num_components
        else:
            raise ValueError('Too few components specified')

        # Optimization parameters
        self.max_iter = max_iter
        self.tol = tol

        # Set standard attributes
        self.priors_given = False

        # Set prior hyperparameters
        self.set_prior_hyperparameters(D=data_dimensionality, K=num_components)

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

        # Remember if priors are given
        #TODO if (a0.shape != 0 or b0 != 0 or n0 != 0 or m0 != 0 or W0 != 1):
        self.priors_given = True

    def free_energy(self, X, rho, thetat, theta0, report=True):
        """
        Compute free energy term to monitor progress.

        Parameters
        ----------
        X : array
            Observed image.
        rho : array
            Array of variational parameters.
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
        N, D = X.shape
        K = rho.shape[1]

        # Unpack parameter sets
        a0, b0, n0, m0, W0 = theta0
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
        for k in range(K):

            ''' Convenience variables '''

            # Proportion assigned to each component
            Nk = np.sum(rho[:, k], axis=0)

            # Responsibility-weighted mean
            xk = np.sum(rho[:, [k]] * X, axis=0) / Nk

            # Reponsibility-weighted variance
            Sk = (X - xk).T @ np.diag(rho[:, k]) @ (X - xk)

            # Mahalanobis distance from hypermean
            mWm = (mt[k, :] - m0[k, :]).T @ Wt[:, :, k] @ (mt[k, :] - m0[k, :])

            # Mahalanobis distance from responsibility-weighted mean
            xWx = (xk - mt[k, :]) @ Wt[:, :, k] @ (xk - mt[k, :]).T

            # Entropy-based terms
            Elog_pik = sp.digamma(at[k]) - sp.digamma(np.sum(at))
            Elog_Lak = (D*np.log(2) +
                        self.log_det(Wt[:, :, k]) +
                        self.multivariate_digamma(nt[k], D))

            ''' Energy function '''

            # First term
            E1 += Nk/2*(Elog_Lak - D / bt[k] -
                        nt[k]*(np.trace(Sk @ Wt[:, :, k]) + xWx) -
                        D*np.log(2*math.pi))

            # Second term
            E2 += np.sum(rho[:, k] * Elog_pik, axis=0)

            # Third term
            E3 += (a0[k] - 1)*Elog_pik + (sp.gammaln(np.sum(a0)) -
                                          np.sum(sp.gammaln(a0))) / K

            # Fourth term
            E4 += 1/2*(D*np.log(b0[k] / (2*math.pi)) +
                       Elog_Lak -
                       D*b0[k]/bt[k] -
                       b0[k]*nt[k]*mWm +
                       (n0[k] - D - 1)*Elog_Lak -
                       2*self.log_partition_Wishart(Wt[:, :, k], nt[k]) +
                       nt[k]*np.trace(alg.inv(W0[:, :, k])*Wt[:, :, k]))

            # Fifth term
            E5 += np.sum(rho[:, k] * np.log(rho[:, k]), axis=0)

            # Sixth term
            E6 += (at[k] - 1)*Elog_pik + (sp.gammaln(np.sum(at)) -
                                          np.sum(sp.gammaln(at))) / K

            # Seventh term
            E7 += (Elog_Lak/2 +
                   D/2*np.log(bt[k] / (2*math.pi)) -
                   D/2 - self.entropy_Wishart(Wt[:, :, k], nt[k]))

        # Compute free energy term
        F = E1 + E2 + E3 + E4 - E5 - E6 - E7

        # Print free energy
        if report:
            print('Free energy = ' + str(F))

        return F

    def expectation_step(self, X, rho, thetat):
        """
        Perform expectation step.

        Parameters
        ----------
        X : array (instances by features)
            Observed data points
        rho : array (instances by 1)
            Array of variational parameters.
        thetat : array
            Current iteration of parameters of variational posteriors.

        Returns
        -------
        rho : array (instances by 1)
            Updated array of variational parameters.

        """
        # Shape of variational parameter array
        N, D = X.shape

        # Unpack tuple of hyperparameters
        at, bt, nt, mt, Wt = thetat

        for k in range(self.K):

            # Compute expected log mixing coefficient
            E1 = sp.digamma(at[k]) - sp.digamma(np.sum(at))

            # Compute exponentiated expected log precision
            E2 = (D*np.log(2) +
                    self.log_det(Wt[:, :, k]) +
                    self.multivariate_digamma(nt[k], D))

            for i in range(N):

                # Compute expected hypermean and hyperprecision
                X_ = (X[i, :] - mt[k, :])
                E3 = D/bt[k] + nt[k] * X_.T @ Wt[:, :, k] @ X_

                # Update variational parameter at current pixels
                rho[i, k] = E1 + E2/2 - E3/2

        # Compute normalization term
        norm = logsumexp(rho, axis=1)
        for k in range(self.K):
            rho[:, k] -= norm

        # Exponentiate rho
        rho = np.exp(rho)

        if np.any(np.sum(rho, axis=1) == 0.0):
            raise RuntimeError('Underflow for variational parameters')

        return rho

    def maximization_step(self, X, rho, thetat, theta0):
        """
        Perform maximization step from variational-Bayes-EM.

        Parameters
        ----------
        X : array (instances by features)
            Observed data points
        rho : array (instances by 1)
            Array of variational parameters.
        thetat : array
            Current iteration of hyperparameters of posteriors.
        theta0 : array
            Hyperparameters of priors.

        Returns
        -------
        thetat : array
            Next iteration of hyperparameters of posteriors.

        """
        # Shapes
        N, D = X.shape

        # Unpack parameter sets
        a0, b0, n0, m0, W0 = theta0
        at, bt, nt, mt, Wt = thetat

        # Iterate over classes
        for k in range(self.K):

            # Convenience variables
            Nk = np.sum(rho[:, k], axis=0)
            xk = np.sum(rho[:, [k]] * X, axis=0) / Nk
            Sk = (X - xk).T @ np.diag(rho[:, k]) @ (X - xk)

            # Update alpha
            at[k] = a0[k] + Nk

            # Update nu
            nt[k] = n0[k] + Nk

            # Update beta
            bt[k] = b0[k] + Nk

            # Update hypermean
            mt[k, :] = (b0[k]*m0[k, :] + Nk*xk) / (b0[k] + Nk)

            # Update hypervariance
            Wt[:, :, k] = alg.inv(alg.inv(W0[:, :, k]) + Sk +
                                  (b0[k]*Nk) / (b0[k] + Nk) *
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
        N, D = X.shape

        # Initialize posterior hyperparameters
        at = np.ones((self.K,))*N/2
        bt = np.ones((self.K,))*2
        nt = np.ones((self.K,))*4
        mt = np.zeros((self.K, D))
        Wt = np.zeros((D, D, self.K))
        for k in range(self.K):
            mt[k, :] = np.mean(X, axis=0) + rnd.randn(1, D)*.1
            Wt[:, :, k] = np.eye(D)*1

        # Pack parameters into sets
        thetat = (at, bt, nt, mt, Wt)

        # Initialize variational posterior responsibilities
        rho = np.ones((N, self.K), dtype='float64') / self.K

        # Initialize old energy variable
        F_ = np.inf

        for t in range(self.max_iter):

            # Monitor progress every tenth iteration
            if t % (self.max_iter/10) == 0:

                # Report progress
                print('Iteration ' + '{0:03}'.format(t+1) + '/' +
                      str(self.max_iter) + '\t', end='')

                # Compute free energy to monitor progress
                F = self.free_energy(X, rho, thetat, self.theta0, report=True)

                if np.abs(F - F_) <= self.tol:
                    print('Step size is below tolerance threshold.')
                    break

                # Update old energy
                F_ = F

            # Expectation step
            rho = self.expectation_step(X, rho, thetat)

            # Expectation step
            thetat = self.maximization_step(X, rho, thetat, self.theta0)

        # Return segmentation along with estimated parameters
        return rho, thetat

    def fit(self, X):
        """
        Fit model to data.

        Parameters
        ----------
        X : array (instances by features)
            Data array.

        Returns
        -------
        pred : array
            Segmentation produced by the model.
        post : array
            Posterior indicator distributions.
        theta : tuple of arrays
            Posterior hyperparameters of parameter distributions.

        """
        # Check if dimensionality of given data matches prior dimensionality.
        if not self.D == X.shape[1]:

            # Check if priors were set
            if not self.priors_given:
                self.set_prior_hyperparameters(D=X.shape[1], K=self.K)
            else:
                self.set_prior_hyperparameters(D=X.shape[1],
                                               K=self.K,
                                               a0=self.theta0[0],
                                               b0=self.theta0[1],
                                               n0=self.theta0[2],
                                               m0=self.theta0[3],
                                               W0=self.theta0[4])

        # Perform VB-EM for segmenting the image
        post, params = self.expectation_maximization(X)

        # Compute most likely class
        pred = np.argmax(post, axis=1)

        # Return segmented image, variational posteriors and parameters
        return pred, post, params


class SemisupervisedGaussianMixture(VariationalMixture):
    """
    Variational Semi-Supervised Gaussian Mixture Model.

    This implementation is adapted from the RPubs note by Andreas Kapourani:
    https://rpubs.com/cakapourani/variational-bayes-gmm
    It models multivariate samples.
    """

    def __init__(self, data_dimensionality=2,
                 num_components=2,
                 theta=(0, 0, 0, 0, 1),
                 max_iter=10,
                 tol=1e-5):
        """
        Model-specific constructors.

        Parameters
        ----------
        data_dimensionality : int
            Number of dimensions of the data.
        num_components : int
            Number of components.
        theta0 : tuple
            Prior hyperparameters.
        max_iter : int
            Maximum number of iterations to run for.
        tol : float
            Tolerance on change in x-value.

        Returns
        -------
        None

        """
        # Store data dimensionality
        if data_dimensionality >= 1:
            self.D = data_dimensionality
        else:
            raise ValueError('Dimensionality must be larger than 0.')

        # Store model parameters
        if num_components >= 2:
            self.K = num_components
        else:
            raise ValueError('Too few components specified')

        # Optimization parameters
        self.max_iter = max_iter
        self.tol = tol

        # Set standard attributes
        self.priors_given = False

        # Set prior hyperparameters
        self.set_prior_hyperparameters(D=data_dimensionality, K=num_components)

    def set_prior_hyperparameters(self, D, K,
                                  a0=np.array([0.1]),
                                  b0=np.array([0.1]),
                                  n0=np.array([2.0]),
                                  m0=np.array([0.0]),
                                  W0=np.array([0.1])):
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

        # Remember if priors are given
        #TODO if (a0.shape != 0 or b0 != 0 or n0 != 0 or m0 != 0 or W0 != 1):
        self.priors_given = True

    def free_energy(self, X, rho, thetat, theta0, report=True):
        """
        Compute free energy term to monitor progress.

        Parameters
        ----------
        X : array
            Observed image.
        rho : array
            Array of variational parameters.
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
        N, D = X.shape
        K = rho.shape[1]

        # Unpack parameter sets
        a0, b0, n0, m0, W0 = theta0
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
        for k in range(K):

            ''' Convenience variables '''

            # Proportion assigned to each component
            Nk = np.sum(rho[:, k], axis=0)

            # Responsibility-weighted mean
            xk = np.sum(rho[:, [k]] * X, axis=0) / Nk

            # Reponsibility-weighted variance
            Sk = (X - xk).T @ np.diag(rho[:, k]) @ (X - xk)

            # Mahalanobis distance from hypermean
            mWm = (mt[k, :] - m0[k, :]).T @ Wt[:, :, k] @ (mt[k, :] - m0[k, :])

            # Mahalanobis distance from responsibility-weighted mean
            xWx = (xk - mt[k, :]) @ Wt[:, :, k] @ (xk - mt[k, :]).T

            # Entropy-based terms
            Elog_pik = sp.digamma(at[k]) - sp.digamma(np.sum(at))
            Elog_Lak = (D*np.log(2) +
                        self.log_det(Wt[:, :, k]) +
                        self.multivariate_digamma(nt[k], D))

            ''' Energy function '''

            # First term
            E1 += Nk/2*(Elog_Lak - D / bt[k] -
                        nt[k]*(np.trace(Sk @ Wt[:, :, k]) + xWx) -
                        D*np.log(2*math.pi))

            # Second term
            E2 += np.sum(rho[:, k] * Elog_pik, axis=0)

            # Third term
            E3 += (a0[k] - 1)*Elog_pik + (sp.gammaln(np.sum(a0)) -
                                          np.sum(sp.gammaln(a0))) / K

            # Fourth term
            E4 += 1/2*(D*np.log(b0[k] / (2*math.pi)) +
                       Elog_Lak -
                       D*b0[k]/bt[k] -
                       b0[k]*nt[k]*mWm +
                       (n0[k] - D - 1)*Elog_Lak -
                       2*self.log_partition_Wishart(Wt[:, :, k], nt[k]) +
                       nt[k]*np.trace(alg.inv(W0[:, :, k])*Wt[:, :, k]))

            # Fifth term
            E5 += np.sum(rho[:, k] * np.log(rho[:, k]), axis=0)

            # Sixth term
            E6 += (at[k] - 1)*Elog_pik + (sp.gammaln(np.sum(at)) -
                                          np.sum(sp.gammaln(at))) / K

            # Seventh term
            E7 += (Elog_Lak/2 +
                   D/2*np.log(bt[k] / (2*math.pi)) -
                   D/2 - self.entropy_Wishart(Wt[:, :, k], nt[k]))

        # Compute free energy term
        F = E1 + E2 + E3 + E4 - E5 - E6 - E7

        # Print free energy
        if report:
            print('Free energy = ' + str(F))

        return F

    def expectation_step(self, X, y, rho, thetat):
        """
        Perform expectation step.

        Parameters
        ----------
        X : array (instances by features)
            Observed data points.
        y : array (instances by 2)
            Labeled samples, first column is index, second is class.
        rho : array (instances by 1)
            Array of variational parameters.
        thetat : array
            Current iteration of parameters of variational posteriors.

        Returns
        -------
        rho : array (instances by 1)
            Updated array of variational parameters.

        """
        # Shape of variational parameter array
        N, D = X.shape

        # Unpack tuple of hyperparameters
        at, bt, nt, mt, Wt = thetat

        for i in range(N):

            if i in y[:, 0]:
                continue

            else:
                for k in range(self.K):

                    # Compute expected log mixing coefficient
                    E1 = sp.digamma(at[k]) - sp.digamma(np.sum(at))

                    # Compute exponentiated expected log precision
                    E2 = (D*np.log(2) +
                          self.log_det(Wt[:, :, k]) +
                          self.multivariate_digamma(nt[k], D))

                    # Compute expected hypermean and hyperprecision
                    X_ = (X[i, :] - mt[k, :])
                    E3 = D/bt[k] + nt[k] * X_.T @ Wt[:, :, k] @ X_

                    # Update variational parameter at current pixels
                    rho[i, k] = E1 + E2/2 - E3/2

                # Numerically stable computation of normalization
                norm = logsumexp(rho[i, :])

                if np.isnan(norm):
                    raise RuntimeError('Underflow for variational parameters')

                for k in range(self.K):
                    rho[i, :] -= norm

        # Exponentiate rho
        rho = np.exp(rho)

        return rho

    def maximization_step(self, X, rho, thetat, theta0):
        """
        Perform maximization step from variational-Bayes-EM.

        Parameters
        ----------
        X : array (instances by features)
            Observed data points
        rho : array (instances by 1)
            Array of variational parameters.
        thetat : array
            Current iteration of hyperparameters of posteriors.
        theta0 : array
            Hyperparameters of priors.

        Returns
        -------
        thetat : array
            Next iteration of hyperparameters of posteriors.

        """
        # Unpack parameter sets
        a0, b0, n0, m0, W0 = theta0
        at, bt, nt, mt, Wt = thetat

        # Iterate over classes
        for k in range(self.K):

            # Convenience variables
            Nk = np.sum(rho[:, k], axis=0)
            xk = np.sum(rho[:, [k]] * X, axis=0) / Nk
            Sk = (X - xk).T @ np.diag(rho[:, k]) @ (X - xk)
            C = (xk - m0[k, :]).T @ (xk - m0[k, :])

            # Update alpha
            at[k] = a0[k] + Nk

            # Update nu
            nt[k] = n0[k] + Nk

            # Update beta
            bt[k] = b0[k] + Nk

            # Update hypermean
            mt[k, :] = (b0[k]*m0[k, :] + Nk*xk) / (b0[k] + Nk)

            # Update hypervariance
            Wt[:, :, k] = alg.inv(alg.inv(W0[:, :, k]) + Sk +
                                  (b0[k]*Nk) / (b0[k] + Nk) * C)

        return at, bt, nt, mt, Wt

    def expectation_maximization(self, X, y):
        """
        Perform Variational Bayes Expectation-Maximization.

        Parameters
        ----------
        X : array (instances by features)
            Data array.
        y : array (instances by 2)
            Labeled samples, first column is index, second is class.

        Returns
        -------
        rho : array (instances by components)
            Variational parameters of posterior for label image.

        """
        # Get shape of image
        N, D = X.shape

        # Initialize posterior hyperparameters
        at = np.ones((self.K,))*N/2
        bt = np.ones((self.K,))*2
        nt = np.ones((self.K,))*4
        mt = np.zeros((self.K, D))
        Wt = np.zeros((D, D, self.K))
        for k in range(self.K):
            #TODO improve y indexing
            mt[k, :] = np.mean(X[y[y[:, 1] == k, 0][0], :], axis=0)
            Wt[:, :, k] = np.eye(D)*10

        # Pack parameters into sets
        thetat = (at, bt, nt, mt, Wt)

        # Initialize variational posterior responsibilities
        rho = np.ones((N, self.K), dtype='float64') / self.K

        # Check labeled samples
        for i in range(y.shape[0]):

            # Fix rho to 1 for observed
            rho[y[i, 0], y[i, 1]] = 1.0

            # Loop over other components
            for l in np.setdiff1d(y[i, 1], np.arange(self.K)):

                # Fix rho to 0 for other components
                rho[y[i, 0], l] = 0.0

        # Initialize old energy variable
        F_ = np.inf

        for t in range(self.max_iter):

            # Monitor progress every tenth iteration
            if t % (self.max_iter/10) == 0:

                # Report progress
                print('Iteration ' + '{0:03}'.format(t+1) + '/' +
                      str(self.max_iter) + '\t', end='')

                # Compute free energy to monitor progress
                F = self.free_energy(X, rho, thetat, self.theta0, report=True)

                if np.abs(F - F_) <= self.tol:
                    print('Step size is below tolerance threshold.')
                    break

                # Update old energy
                F_ = F

            # Expectation step
            rho = self.expectation_step(X, y, rho, thetat)

            # Expectation step
            thetat = self.maximization_step(X, rho, thetat, self.theta0)

        # Return segmentation along with estimated parameters
        return rho, thetat

    def fit(self, X, y):
        """
        Fit model to data.

        Parameters
        ----------
        X : array (instances by features)
            Data array.
        y : array (instances by 2)
            Labeled samples, first column is index, second is class.

        Returns
        -------
        pred : array
            Segmentation produced by the model.
        post : array
            Posterior indicator distributions.
        theta : tuple of arrays
            Posterior hyperparameters of parameter distributions.

        """
        # Check if dimensionality of given data matches prior dimensionality.
        if not self.D == X.shape[1]:

            # Check if priors were set
            if not self.priors_given:
                self.set_prior_hyperparameters(D=X.shape[1], K=self.K)
            else:
                self.set_prior_hyperparameters(D=X.shape[1],
                                               K=self.K,
                                               a0=self.theta0[0],
                                               b0=self.theta0[1],
                                               n0=self.theta0[2],
                                               m0=self.theta0[3],
                                               W0=self.theta0[4])

        # Perform VB-EM
        post, params = self.expectation_maximization(X, y)

        # Compute most likely class
        pred = np.argmax(post, axis=1)

        # Return segmented image, variational posteriors and parameters
        return pred, post, params
