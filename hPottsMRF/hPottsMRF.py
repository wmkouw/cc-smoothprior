#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Class of hidden Potts-Markov Random Fields

The goal is to fit the smoothing parameter / granularity coefficient of the
Potts prior on a series of external label fields. This maximum likelihood
estimate is used later on in the overal Bayesian image segmentation model.

Author: W.M.Kouw
Date: 18-09-2018
"""
import numpy as np
import numpy.random as rnd
import scipy.special as sp
import scipy.optimize as opt


class hiddenPottsMarkovRandomField(object):
    """Hidden Potts-Markov Random Field."""

    def __init__(self, neighbourhood_size=(3, 3), num_iter=5):
        """
        Initialize variables for an instance of a hidden Potts-MRF.

        Parameters
        ----------
        neighbourhood_size : (int, int)
            Size of the neighbourhood on which the variational approximation
            depends (def: (1, 1))
        num_iter : int
            Number of iterations to run VB-EM.

        Returns
        -------
        None

        """
        # Store model parameters
        if np.all(neighbourhood_size >= (3, 3)):
            self.neighbourhood_size = neighbourhood_size
        else:
            raise ValueError('Neighbourhood size is too small')

        # Optimization parameters
        self.num_iter = num_iter

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
        """
        Mean-field variational approximation to Potts log-likelihood function.

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

        # Pad shape with zero edge for easier neighbourhood extraction
        Z = np.pad(Z, [1, 1], mode='constant', constant_values=0)

        # Initialize negative log-likelihood
        nll = 0

        # Loop over pixels
        for h in range(1, H-1):
            for w in range(1, W-1):

                # Initialize intermediate terms
                chi_ik = 0
                ksi_ik = 0

                # Select current class
                for k in range(K):

                    # Extract neighbourhood of current class
                    delta_ik = Z[h-1:h+2, w-1:w+2, k]

                    # First sum is neighbourhood comparison
                    chi_ik += np.sum(Z[h, w, k] * delta_ik) - Z[h, w, k]**2

                    # Second sum is purely over neighbourhood
                    ksi_ik += np.exp(2*beta*(np.sum(delta_ik) - Z[h, w, k]))

                # Update negative log-likelihood
                nll += -2*beta*chi_ik + np.log(ksi_ik)

        return nll

    def mean_field_Potts_grad(self, beta, Z):
        r"""
        Partial derivative of mean-field Potts log-likelihood w.r.t. beta.

        Derivative has the following form:

        d/db log q(z|b) = \sum_{i=1}^{n} 2 \sum_{l=1}^{K} z_{il} 
            \sum_{j \in delta_i} 

        Parameters
        ----------
        beta : float
            Smoothing parameter / granularity coefficient
        Z : array (height by width by number of classes)
            Label field to fit


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

        # Pad shape with zero edge for easier neighbourhood extraction
        Z = np.pad(Z, [1, 1], mode='constant', constant_values=0)

        # Initialize log-likelihood
        dqdb = 0

        # Loop over pixels
        for h in range(1, H-1):
            for w in range(1, W-1):

                # Initialize intermediate terms
                chi_ik = 0
                ksi_ik = 0
                tau_ik = 0

                # Select current class
                for k in range(K):

                    # Extract neighbourhood of current class
                    delta_ik = np.ravel(Z[h-1:h+2, w-1:w+2, k])

                    # First sum is neighbourhood comparison
                    chi_ik += 2*(np.sum(Z[h, w, k] * delta_ik) - Z[h, w, k]**2)

                    # Second sum
                    ksi_ik += 2*(np.sum(delta_ik) - Z[h, w, k])

                    # Second sum is purely over neighbourhood
                    tau_ik += np.exp(2*beta*(np.sum(delta_ik) - Z[h, w, k]))

                # Update partial derivative
                dqdb += -chi_ik + ksi_ik / tau_ik

        return dqdb

    def maximum_likelihood_beta(self, Z, lb=[(0, None)], verbose=False):
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

        Returns
        -------
        beta_hat : float
            Estimated beta, or smoothing parameter, for given label field

        """
        # Check if Z is the right shape
        if len(Z.shape) == 2:

            # Map to one-hot encoding
            Z = self.one_hot(Z)

        # Initial value
        beta0 = np.array([10.0])

        # Start optimization procedure
        beta_hat = opt.minimize(fun=self.mean_field_Potts,
                                x0=beta0,
                                args=(Z),
                                method='L-BFGS-B',
                                # jac=self.mean_field_Potts_grad,
                                bounds=lb,
                                options={'disp': True}
                                )

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

    def neighbourhood(self, A, index, patch=True):
        """
        Extract a neighbourhood of pixels around current pixel.

        Parameters
        ----------
        A : array
            Array from which to extract the pixel's neighbourhood.
        index : (int, int)
            Row and column index of current pixel.
        patch : bool
            Whether to pair only with direct upper, lower and side pixels.

        Returns
        -------
        delta_i : vector of neighbours of current pixel

        """
        # Shape of array
        H, W = A.shape

        if not patch:

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
            vstep = int((self.neighbourhood_size[0] - 1) / 2)
            hstep = int((self.neighbourhood_size[1] - 1) / 2)

            # Pad image to allow slicing at the edges
            A = np.pad(A, [vstep, hstep], mode='constant', constant_values=0)

            # Define slices
            vslice = slice(index[0]-vstep+vstep, index[0]+vstep+1+vstep)
            hslice = slice(index[1]-hstep+hstep, index[1]+hstep+1+hstep)

            # Initialize neighbourhood list
            return A[vslice, hslice]

    def expectation_step(self, y, nu, theta, beta, neighbourhood_size=(1, 1)):
        """
        Perform expectation step from variational-Bayes-EM.

        Parameters
        ----------
        y : array
            Observed image.
        nu : array
            Array of variational parameters.
        theta : array
            Parameters of variational posterior of Potts model.
        beta : float
            Smoothing parameter.
        neighbourhood_size : (int, int)
            Size of the neighbourhood for mean-field.

        Returns
        -------
        nu : array
            Updated array of variational parameters.

        """
        # Shape of variational parameter array
        H, W, K = nu.shape

        # Unpack tuple of hyperparameters
        mu, la, ga, ks = theta

        for h in range(H):
            for w in range(W):
                for k in range(K):

                    # Compute expectation of log(tau_l) w.r.t. phi_l
                    E_log_tau_l = sp.digamma(ga[k] / 2) - np.log(ks[k] / 2)

                    # Compute expectation of tau_l w.r.t. phi_l
                    E_tau_l = ga[k] / ks[k]

                    # Compute expectation of log p(y_i|phi_l) w.r.t. phi_l
                    E_log_py = (E_log_tau_l/2 -
                                E_tau_l*(y[h, w] - mu[k])**2 / 2 -
                                1 / (2*ks[k]))

                    # Take sum over neighbourhood
                    nu_di = self.neighbourhood(nu[:, :, k], (h, w))

                    # Update variational parameter at current pixel
                    nu[h, w, k] = np.exp(E_log_py + 2*beta*np.sum(nu_di))

                # Normalize nu_i to 1
                nu[h, w, :] = nu[h, w, :] / np.sum(nu[h, w, :])

        return nu

    def maximization_step(self, y, nu, theta, theta0):
        """
        Perform maximization step from variational-Bayes-EM.

        Parameters
        ----------
        y : array
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
        mu, la, ga, ks = theta
        mu0, la0, ga0, ks0 = theta0

        # Iterate over classes
        for k in range(K):

            # Update lambda
            la[k] = la0[k] + np.sum(nu[:, :, k], axis=(0, 1))

            # Update gamma
            ga[k] = ga0[k] + np.sum(nu[:, :, k], axis=(0, 1))

            # Update mu
            mu[k] = (la0[k]*mu0[k] + np.sum(y*nu[:, :, k], axis=(0, 1)))/la[k]

            # Update ksi
            ks[k] = (ks0[k] + np.sum(y**2*nu[:, :, k], axis=(0, 1)) +
                     la0[k]*mu0[k]**2 - la[k]*mu[k]**2)

        return mu, la, ga, ks

    def expectation_maximization(self, y, K, beta):
        """
        Perform variational Bayes Expectation-Maximization.

        Parameters
        ----------
        y : array (height by width)
            Image to be segmented.
        K : int
            Number of classes to segment image into.
        beta : float
            Smoothing parameter / granularity coefficient of Potts model.

        Returns
        -------
        nu : array (height by width by number of classes)
            Variational parameters of posterior for label image.

        """
        # Get shape of image
        H, W = y.shape

        # Initialize hyperparameters
        mu0 = np.zeros((K,))
        la0 = np.zeros((K,))
        ga0 = np.ones((K,)) / 2
        ks0 = np.ones((K,)) / 2
        theta0 = (mu0, la0, ga0, ks0)

        # Copy hyperparameter array for updating
        theta = np.copy(theta0)

        # Initialize variational parameters array
        nu = rnd.randn(H, W, K)

        for r in range(self.num_iter):

            # Report progress
            print('At iteration ' + str(r+1) + '/' + str(self.num_iter))

            # Expectation step
            nu = self.expectation_step(y, nu, theta, beta)

            # Expectation step
            theta = self.maximization_step(y, nu, theta, theta0)

        # Return segmentation along with estimated parameters
        return nu, theta

    def segment(self, y, K, Q=[], beta=1.0, output_params=False):
        """
        Segment an image.

        Parameters
        ----------
        y : array
            image to be segmented.
        K : int
            number of classes
        Q : array
            segmented image to copy smoothness from.
        beta : float
            Smoothing parameter.
        output_params : bool
            Whether to output the estimated hyperparameters.

        Returns
        -------
        z : array
            segmentation produced by the model.

        """
        # Check for auxiliary segmentation
        if np.any(Q):
            beta = self.maximum_likelihood_beta(Q, verbose=True)

        # Perform VB-EM for segmenting the image
        nu, theta = self.expectation_maximization(y, K, beta)

        # Return segmented image
        if output_params:
            return nu, theta
        else:
            return nu
