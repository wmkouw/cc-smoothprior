#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Class of variatonal Gaussian Mixture Models.

It serves as a baseline for a hidden Potts-MRF for Bayesian unsupervised image
segmentation.

Author: W.M. Kouw
Date: 05-11-2018
"""
import numpy as np
import numpy.random as rnd
import scipy.special as sp
import scipy.optimize as opt
import sklearn.cluster as cl


class variationalGaussianMixture(object):
    """
    Variational Gaussian Mixture Model.

    This implementation is based on the note on variational mixtures of
    Gaussians from Oxford Robotics: www.robots.ox.ac.uk/~sjrob/Pubs/vbmog.ps.gz
    """

    def __init__(self, num_components=2):
        """
        Initialize variables for an instance of a hidden Potts-MRF.

        Parameters
        ----------
        num_components : int
            Number of hidden components

        Returns
        -------
        None

        """
        # Store model parameters
        if num_components >= 2:
            self.nK = num_components
        else:
            raise ValueError('Too few components specified')

    def one_hot(self, A):
        """
        Map array to pages with binary encodings.

        Parameters
        ----------
        A : array (height by width)
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

    def KL_Dir(self, la_q, la_p):
        """KL-divergence between two Dirichlet distributions."""
        # Derive number of components from parameters
        K = len(la_q)

        # Check for same number of components
        if not K == len(la_p):
            raise ValueError('Parameter sets are of different lengths.')

        # Compute sum terms
        la_qt = np.sum(la_q)
        la_pt = np.sum(la_p)

        # First term
        t1 = sp.gammaln(la_qt) - sp.gammaln(la_pt)

        # Loop over classes for other terms
        t2 = 0
        t3 = 0
        for k in range(K):

            # Second term
            t2 += (la_q[k] - la_p[k])*(sp.digamma(la_q[k]) - sp.digamma(la_qt))

            # Third term
            t3 += sp.gammaln(la_p[k]) - sp.gammaln(la_q[k])

        return t1 + t2 + t3

    def KL_Gam(self, be_q, ce_q, be_p, ce_p):
        """KL-divergence between two Gamma distributions."""
        # Derive number of components from parameters
        K = len(be_q)

        # Check for same number of components
        if not K == len(ce_q) and K == len(be_p) and K == len(ce_p):
            raise ValueError('Parameter sets are of different lengths.')

        # Loop over components
        D = 0
        for k in range(K):

            # Evaluate KL-divergence for each component
            D += ((ce_q[k] - 1) * sp.digamma(ce_q[k]) -
                  np.log(be_q[k]) -
                  ce_q[k] -
                  sp.gammaln(ce_q[k]) +
                  sp.gammaln(ce_p[k]) +
                  ce_p[k] * np.log(be_p[k]) -
                  (ce_p[k] - 1) * (sp.digamma(ce_q[k]) + np.log(be_q[k])) +
                  be_q[k] * ce_q[k] / be_p[k])

        return D

    def KL_Norm(self, mu_q, tau_q, mu_p, tau_p):
        """KL-divergence between two normal distributions."""
        # Derive number of components from parameters
        K = len(mu_q)

        # Check for same number of components
        if not K == len(tau_q) and K == len(mu_p) and K == len(tau_p):
            raise ValueError('Parameter sets are of different lengths.')

        # Invert precisions
        si2_q = 1/tau_q
        si2_p = 1/tau_p

        # Loop over components
        D = 0
        for k in range(K):

            D += (0.5*si2_p[k] / si2_q[k] +
                  (mu_q[k]**2 + mu_p[k]**2 + si2_q[k]**2 - 2*mu_q[k]*mu_p[k]) /
                  (2*si2_p[k]) - 0.5)

        return D

    def free_energy(self, X, nu, theta, theta0, report=True):
        """
        Compute free energy term to monitor progress.

        Parameters
        ----------
        X : array
            Observed image.
        nu : array
            Array of variational parameters.
        theta : array
            Parameters of variational posteriors.
        theta0 : array
            Parameters of variational priors.
        report : bool
            Print value of free energy function.

        Returns
        -------
        nu : array
            Updated array of variational parameters.

        """
        # Shapes
        H, W, K = nu.shape
        N = H*W

        # Unpack parameter sets
        la, be, ce, em, ve = theta
        la0, be0, ce0, em0, ve0 = theta0

        # Compute KL-term for mixing coefficients
        KLmixc = self.KL_Dir(la, la0)

        # Compute KL-term for precisions
        KLprec = self.KL_Gam(be, ce, be0, ce0)

        # Compute KL-term for means
        KLnorm = self.KL_Norm(em, ve, em0, ve0)

        # Loop over classes
        Hq = 0
        L = 0
        for k in range(K):

            # Compute convenience variables
            c1 = np.mean(nu[:, :, k], axis=(0, 1))
            c2 = np.sum(nu[:, :, k], axis=(0, 1))
            c3 = np.mean(nu[:, :, k]*X, axis=(0, 1))
            c4 = np.mean(nu[:, :, k]*X**2, axis=(0, 1))
            c5 = sp.digamma(la[k]) - sp.digamma(np.sum(la))
            c6 = sp.digamma(ce[k]) + np.log(be[k])
            c7 = c4 + c1*(em[k]**2 + ve[k]) - 2*em[k]*c3

            # Compute entropy term
            Hq += np.sum(-nu[:, :, k]*np.log(nu[:, :, k]), axis=(0, 1))

            # Compute likelihood term
            L += c2*(c5 + c6/2) - N/2*(be[k]*ce[k])*c7

        # Compute free energy term
        F = Hq + L - KLmixc - KLprec - KLnorm

        # Print free energy
        if report:
            print('Free energy = ' + str(F))

        return F

    def expectation_step(self, X, nu, theta):
        """
        Perform expectation step.

        Parameters
        ----------
        X : array
            Observed image.
        nu : array
            Array of variational parameters.
        theta : array
            Parameters of variational posterior of Potts model.

        Returns
        -------
        nu : array
            Updated array of variational parameters.

        """
        # Shape of variational parameter array
        H, W, K = nu.shape

        # Unpack tuple of hyperparameters
        la, be, ce, em, ve = theta

        for h in range(H):
            for w in range(W):
                for k in range(K):

                    # Compute expected log mixing coefficient
                    E1 = sp.digamma(la[k]) - sp.digamma(np.sum(la))

                    # Compute expected log precision
                    E2 = sp.digamma(ce[k]) + np.log(be[k])

                    # Compute expected precision
                    E3 = be[k]*ce[k]

                    # Expanded square term
                    E4 = X[h, w]**2 + em[k]**2 + ve[k] - 2*em[k]*X[h, w]

                    # Update variational parameter at current pixel
                    nu[h, w, k] = np.exp(E1 + E2/2 - 1/2*E3*E4)

                # Normalize nu_i to 1
                if np.sum(nu[h, w, :]) == 0.0:
                    raise RuntimeError('Underflow for variational parameters')
                else:
                    nu[h, w, :] = nu[h, w, :] / np.sum(nu[h, w, :])

        return nu

    def maximization_step(self, X, nu, theta, theta0):
        """
        Perform maximization step from variational-Bayes-EM.

        Parameters
        ----------
        X : array
            Observed image
        nu : array
            Variational parameters consisting of current segmentation.
        theta : array
            Hyperparameters of posteriors
        theta0 : array
            Hyperparameters of priors.

        Returns
        -------
        theta : array
            Estimated hyperparameters of posteriors.

        """
        # Shapes
        H, W, K = nu.shape
        N = H*W

        # Unpack parameter sets
        la, be, ce, em, ve = theta
        la0, be0, ce0, em0, ve0 = theta0

        # Iterate over classes
        for k in range(K):

            # Precompute convenience variables
            c1 = np.mean(nu[:, :, k], axis=(0, 1))
            c2 = np.sum(nu[:, :, k], axis=(0, 1))
            c3 = np.mean(nu[:, :, k]*X, axis=(0, 1))
            c4 = np.mean(nu[:, :, k]*X**2, axis=(0, 1))

            # Update la
            la[k] = la0[k] + c2

            # Update be
            si2_k = c4 + c1*(em[k]**2 + ve[k]) - 2*em[k]*c3
            invbe = N/2.*si2_k + 1./be0[k]
            be[k] = 1./invbe

            # Update ce
            ce[k] = c2/2. + ce0[k]

            # Compute data-dependent terms
            em_data = c3 / c1
            ta_data = c2 * (be[k]*ce[k])

            # Update ve
            ta0 = 1/ve0[k]
            ta_k = ta0 + ta_data
            ve[k] = 1./ta_k

            # Update em
            em[k] = (ta0*em0[k] + ta_data*em_data)*ve[k]

        return la, be, ce, em, ve

    def expectation_maximization(self, X, K, max_iter=1, tol=1e-5):
        """
        Perform variational Bayes Expectation-Maximization.

        Parameters
        ----------
        X : array (height by width)
            Image to be segmented.
        K : int
            Number of classes to segment image into.
        max_iter : int
            Maximum number of iterations to run EM for.

        Returns
        -------
        nu : array (height by width by number of classes)
            Variational parameters of posterior for label image.

        """
        # Get shape of image
        H, W = X.shape

        # Initialize hyperparameters
        la0 = np.tile(5., [K, ])
        be0 = np.tile(2., [K, ])
        ce0 = np.tile(0.1, [K, ])
        em0 = np.tile(0.0, [K, ])
        ve0 = np.tile(1.0, [K, ])

        # Initialize posterior hyperparameters
        la = rnd.randn(K,)*0.5 + la0
        be = rnd.randn(K,)*0.5 + be0
        ce = rnd.randn(K,)*0.5 + ce0

        # Use k-means to help initialize several parameters
        kM = cl.KMeans(n_clusters=K).fit(X.reshape((-1, 1)))
        Xp = kM.predict(X.reshape((-1, 1)))
        Xt = kM.transform(X.reshape((-1, 1)))

        # Initialize means with cluster centers
        em = kM.cluster_centers_.reshape((-1, ))

        # Initialize variances with cluster distance variances
        ve = np.zeros((K, ))
        for k in range(K):
            ve[k] = np.var(Xt[Xp == k])

        # Initialize class indicators with k-means predictions
        nu = self.one_hot(Xp.reshape((H, W)))

        # Pack parameters into sets
        theta0 = (la0, be0, ce0, em0, ve0)
        theta = (la, be, ce, em, ve)

        # Initialize old energy variable
        F_ = np.inf

        for r in range(max_iter):

            # Report progress
            print('Iteration ' + str(r+1) + '/' + str(max_iter) + '\t', end='')

            # Expectation step
            nu = self.expectation_step(X, nu, theta)

            # Expectation step
            theta = self.maximization_step(X, nu, theta, theta0)

            # Compute free energy to monitor progress
            F = self.free_energy(X, nu, theta, theta0, report=True)

            # Check tolerance
            if np.abs(F - F_) <= tol:
                print('Step size is below tolerance threshold.')
                break

            # Update old energy
            F_ = F

        # Return segmentation along with estimated parameters
        return nu, theta

    def segment(self, X, K, max_iter=1):
        """
        Segment an image.

        Parameters
        ----------
        X : array
            Image to be segmented.
        K : int
            Number of classes
        max_iter : int
            Maximum number of iterations to perform optimization

        Returns
        -------
        Z : array
            Segmentation produced by the model.
        nu : array
            Posterior indicator distributions.
        theta : tuple of arrays
            Posterior hyperparameters of parameter distributions.

        """
        # Perform VB-EM for segmenting the image
        nu, theta = self.expectation_maximization(X, K, max_iter)

        # Compute most likely class
        Z_hat = np.argmax(nu, axis=2)

        # Return segmented image, variational posteriors and parameters
        return Z_hat, nu, theta
