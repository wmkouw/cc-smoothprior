#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Class of hidden Potts-Markov Random Fields

The goal is to fit the smoothing parameter / granularity coefficient of the
Potts prior on a series of external label fields. This maximum likelihood
estimate is used later on in the overal Bayesian image segmentation model.

Author: W.M. Kouw
Date: 18-09-2018
"""
import numpy as np
import numpy.random as rnd
import scipy.special as sp
import scipy.optimize as opt
import sklearn.cluster as cl


class hiddenPotts(object):
    """Hidden Potts-Markov Random Field."""

    def __init__(self, patch_size=(3, 3), num_iter=5):
        """
        Initialize variables for an instance of a hidden Potts-MRF.

        Parameters
        ----------
        patch_size : (int, int)
            Size of the neighbourhood on which the variational approximation
            depends (def: (1, 1))
        num_iter : int
            Number of iterations to run VB-EM.

        Returns
        -------
        None

        """
        # Store model parameters
        if np.all(patch_size >= (1, 1)):
            self.patch_size = patch_size
        else:
            raise ValueError('Neighbourhood size is too small')

    def Hamiltonian(self, z):
        r"""
        Compute Hamiltonian function for Potts prior.

        H(z) = \prod_{n=1}^{N} \prod_{n' \in V(n)} \delta(z_n == z_{n'}) .

        where V(n) denotes the neighbourhood of the current pixel.

        Parameters
        ----------
        z : array(N x 1)
            Label image.

        Returns
        -------
        H : int
            Value of H(z)

        """
        # Count neighbourhood equality
        H = 0

        # Pad array with zeros
        z = np.pad(z, [1, 1], mode='constant', constant_values=-1)

        # Looping over every pixel
        for i in range(1, z.shape[0]-1):
            for j in range(1, z.shape[1]-1):

                # Left neighbour
                if z[i, j-1] == z[i, j]:
                    H += 1

                # Top neighbour
                if z[i-1, j] == z[i, j]:
                    H += 1

                # Right neighbour
                if z[i, j+1] == z[i, j]:
                    H += 1

                # Bottom neighbour
                if z[i+1, j] == z[i, j]:
                    H += 1

        # Return counter
        return H

    def mean_field_Potts(self, beta, Z):
        r"""
        Mean-field variational approximation to Potts log-likelihood function.

        logp(z_i | z_{d_i}, beta) = 2*beta*\sum_{k=1}^{K} z_{ik}
            \sum_{j \in \delta_{ik}} z_{jk} - \log \sum_{z_{i'}}
            \exp(2*beta*\sum_{k=1}^{K} z_{i'k} \sum_{j \in \delta_{ik}} z_{jk})

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

        # Test for one-hot in each page
        for k in range(K):
            if not np.all(np.unique(Z[:, :, k]) == [0, 1]):
                raise ValueError("Label field is not binary in page " + str(k))

        # Initialize negative log-likelihood
        nll = 0

        # Loop over pixels
        for h in range(H):
            for w in range(W):

                # Initialize intermediate terms
                chi_i = 0
                ksi_i = 0

                # Select current class
                for k in range(K):

                    # Extract neighbourhood of current class
                    d_ik = self.neighbourhood(Z[:, :, k], (h, w))

                    # First sum is neighbourhood comparison
                    chi_i += Z[h, w, k]*np.sum(d_ik)

                    # Second sum is purely over neighbourhood
                    ksi_i += np.exp(2*beta*np.sum(d_ik))

                # Update negative log-likelihood
                nll += -2*beta*chi_i + np.log(ksi_i)

        return nll

    def mean_field_Potts_grad(self, beta, Z):
        r"""
        Partial derivative of mean-field Potts log-likelihood w.r.t. beta.

        Derivative has the following form:

        d/db log q(z|b) = \sum_{i=1}^{n} 2 \sum_{l=1}^{K} z_{il}
            \sum_{j \in delta_i}

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

        # Check for binary-class image
        if K == 1:
            Z = np.concatenate((Z, 1 - Z), axis=2)
            K = 2

        # Test for one-hot in each page
        for k in range(K):
            if not np.all(np.unique(Z[:, :, k]) == [0, 1]):
                raise ValueError("Label field is not binary in page " + str(k))

        # Initialize log-likelihood
        dqdb = 0

        # Loop over pixels
        for h in range(H):
            for w in range(W):

                # Initialize intermediate terms
                chi_i = 0
                ksi_i = 0
                psi_i = np.zeros((K,))

                # Select current class
                for k in range(K):

                    # Extract neighbourhood of current pixel
                    d_ik = self.neighbourhood(Z[:, :, k], (h, w))

                    # First term
                    chi_i += Z[h, w, k]*np.sum(d_ik)

                    # Denominator
                    ksi_i += np.exp(2*beta*np.sum(d_ik))

                    # Numerator
                    psi_i[k] = np.exp(2*beta*np.sum(d_ik))*(2*np.sum(d_ik))

                # Update partial derivative
                dqdb += -2*chi_i + np.sum(psi_i / ksi_i)

        return np.array(dqdb)

    def maximum_likelihood_beta(self, Z,
                                lb=[(0, None)],
                                verbose=False,
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
        if len(Z.shape) == 2:

            # Map to one-hot encoding
            Z = self.one_hot(Z)

        # Initial value
        beta0 = np.array([1.0])

        # Start optimization procedure
        beta_hat = opt.minimize(fun=self.mean_field_Potts,
                                x0=beta0,
                                args=Z,
                                method='L-BFGS-B',
                                jac=self.mean_field_Potts_grad,
                                bounds=lb,
                                options={'disp': True,
                                         'maxiter': max_iter})

        # Report value
        if verbose:
            print(beta_hat.x[0])

        # Check value
        if beta_hat.x[0] > 1e2:
            print('Warning: beta_hat is very large.')

        # Return
        return beta_hat.x[0]

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

    def neighbourhood(self, A, index):
        """
        Extract a neighbourhood of pixels around current pixel.

        Parameters
        ----------
        A : array
            Array from which to extract the pixel's neighbourhood.
        index : (int, int)
            Row and column index of current pixel.

        Returns
        -------
        delta_i : vector of neighbours of current pixel

        """
        # Shape of array
        H, W = A.shape

        if np.all(self.patch_size == (1, 1)):

            # Initialize neighbourhood list
            delta_i = []

            # Check for current pixel at top-left boundary
            if np.all(index == (0, 0)):

                delta_i.append(A[0, 1])
                delta_i.append(A[1, 0])

            # Check for current pixel at top boundary
            elif (index[0] == 0) and (index[1] != 0 and index[1] != W-1):

                delta_i.append(A[0, index[1]-1])
                delta_i.append(A[0, index[1]+1])
                delta_i.append(A[1, index[1]])

            # Check for current pixel at top-right boundary
            elif np.all(index == (0, W-1)):

                delta_i.append(A[0, W-2])
                delta_i.append(A[1, W-1])

            # Check for current pixel at right boundary
            elif (index[0] != 0 and index[0] != H-1) and (index[1] == H-1):

                delta_i.append(A[index[0]-1, H-1])
                delta_i.append(A[index[0]+1, H-1])
                delta_i.append(A[index[0], H-2])

            # Check for current pixel at bottom-right boundary
            elif np.all(index == (H-1, W-1)):

                delta_i.append(A[H-2, W-1])
                delta_i.append(A[H-1, W-2])

            # Check for current pixel at bottom boundary
            elif (index[0] == H-1) and (index[1] != 0 and index[1] != W-1):

                delta_i.append(A[H-1, index[1]-1])
                delta_i.append(A[H-1, index[1]+1])
                delta_i.append(A[H-2, index[1]])

            # Check for current pixel at bottom-left boundary
            elif np.all(index == (H-1, 0)):

                delta_i.append(A[H-1, 1])
                delta_i.append(A[H-2, 0])

            # Check for current pixel at left boundary
            elif (index[0] != 0 and index[0] != H-1) and (index[1] == 0):

                delta_i.append(A[index[0]-1, 0])
                delta_i.append(A[index[0]+1, 0])
                delta_i.append(A[index[0], 1])

            else:

                delta_i.append(A[index[0]-1, index[1]])
                delta_i.append(A[index[0], index[1]-1])
                delta_i.append(A[index[0], index[1]+1])
                delta_i.append(A[index[0]+1, index[1]])

            # Return list, formatted to array
            return np.array(delta_i)

        else:

            # Patch step size
            vstep = int((self.patch_size[0] - 1) / 2)
            hstep = int((self.patch_size[1] - 1) / 2)

            # Pad image to allow slicing at the edges
            A = np.pad(A, [vstep, hstep], mode='constant', constant_values=0)

            # Define slices
            vslice = slice(index[0]-vstep+vstep, index[0]+vstep+1+vstep)
            hslice = slice(index[1]-hstep+hstep, index[1]+hstep+1+hstep)

            # Initialize neighbourhood list
            return A[vslice, hslice]

    def expectation_step(self, X, nu, theta, beta):
        """
        Perform expectation step from variational-Bayes-EM.

        Parameters
        ----------
        X : array
            Observed image.
        nu : array
            Array of variational parameters.
        theta : array
            Parameters of variational posterior of Potts model.
        beta : float
            Smoothing parameter.
        patch_size : (int, int)
            Size of the neighbourhood for mean-field.

        Returns
        -------
        q : array
            Updated array of variational parameters.

        """
        # Shape of variational parameter array
        H, W, K = nu.shape

        # Unpack tuple of hyperparameters
        em, la, ga, ks = theta

        for h in range(H):
            for w in range(W):
                for k in range(K):

                    # Take sum over neighbourhood
                    d_ik = self.neighbourhood(nu[:, :, k], (h, w))

                    # Compute expectation of log(tau_l) w.r.t. phi_l
                    E_log_tau_l = sp.digamma(ga[k] / 2) - np.log(ks[k] / 2)

                    # Compute expectation of tau_l w.r.t. phi_l
                    E_tau_l = ga[k] / ks[k]

                    # Compute expectation of log p(y_i|phi_l) w.r.t. phi_l
                    E_log_py = (E_log_tau_l/2 -
                                E_tau_l*(X[h, w] - em[k])**2 / 2 -
                                1/(2*ks[k]))

                    # Update variational parameter at current pixel
                    nu[h, w, k] = np.exp(E_log_py + 2*beta*np.sum(d_ik))

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
            Hyperparameters for variational Potts-MRF.
        theta0 : array
            Initial values for hyperparameters.

        Returns
        -------
        theta : array
            Updated hyperparameters for variational Potts-MRF.

        """
        # Check number of classes
        K = nu.shape[2]

        # Unpack tuples of hyperparameters
        em, la, ga, ks = theta
        em0, la0, ga0, ks0 = theta0

        # Iterate over classes
        for k in range(K):

            # Update lambda
            la[k] = la0[k] + np.sum(nu[:, :, k], axis=(0, 1))

            # Update gamma
            ga[k] = ga0[k] + np.sum(nu[:, :, k], axis=(0, 1))

            # Update em
            em[k] = (la0[k]*em0[k] + np.sum(X*nu[:, :, k], axis=(0, 1)))/la[k]

            # Update ksi
            ks[k] = (ks0[k] + np.sum(X**2*nu[:, :, k], axis=(0, 1)) +
                     la0[k]*em0[k]**2 - la[k]*em[k]**2)

        return em, la, ga, ks

    def expectation_maximization(self, X, K, beta, num_iter=1):
        """
        Perform variational Bayes Expectation-Maximization.

        Parameters
        ----------
        X : array (height by width)
            Image to be segmented.
        K : int
            Number of classes to segment image into.
        beta : float
            Smoothing parameter / granularity coefficient of Potts model.
        num_iter : int
            Number of iterations to run EM for.

        Returns
        -------
        nu : array (height by width by number of classes)
            Variational parameters of posterior for label image.

        """
        # Get shape of image
        H, W = X.shape

        # Initialize hyperparameters
        em0 = np.zeros((K,), dtype='float32')
        la0 = np.ones((K,))*0.05
        ga0 = np.ones((K,))*1.0
        ks0 = np.ones((K,))*1.0

        # Initialize posterior hyperparameters
        ga = rnd.randn(K,)*0.5 + ga0
        ks = rnd.randn(K,)*0.5 + ks0

        # Use k-means to help initialize parameters
        kM = cl.KMeans(n_clusters=K, max_iter=500).fit(X.reshape((-1, 1)))
        Xp = kM.predict(X.reshape((-1, 1)))
        Xt = kM.transform(X.reshape((-1, 1)))

        # Initialize means with cluster centers
        em = kM.cluster_centers_.reshape((-1, ))

        # Initialize precision with cluster distance variances
        la = np.zeros((K, ))
        for k in range(K):
            la[k] = 1/np.var(Xt[Xp == k])

        # Initialize class indicators with k-means predictions
        nu = self.one_hot(Xp.reshape((H, W)))

        # Pack parameters into sets
        theta0 = (em0, la0, ga0, ks0)
        theta = (em, la, ga, ks)

        for r in range(num_iter):

            # Report progress
            print('At iteration ' + str(r+1) + '/' + str(num_iter))

            # Expectation step
            nu = self.expectation_step(X, nu, theta, beta)

            # Expectation step
            theta = self.maximization_step(X, nu, theta, theta0)

        # Return segmentation along with estimated parameters
        return nu, theta

    def segment(self, X, K, beta, num_iter=1):
        """
        Segment an image.

        Parameters
        ----------
        X : array
            Image to be segmented.
        K : int
            Number of classes.
        beta : int
            Smoothness parameter for Potts prior.
        num_iter : int
            Number of iterations to perform optimization.

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
        nu, theta = self.expectation_maximization(X, K, beta, num_iter)

        # Compute most likely class
        Z_hat = np.argmax(nu, axis=2)

        # Return segmented image, variational posteriors and parameters
        return Z_hat, nu, theta
