"""
Set of utility functions.

Author: Wouter M. Kouw
Last updated: 30-10-2018
"""
import numpy as np
import numpy.random as rnd
import scipy.optimize as opt
import scipy.ndimage as nd
import nibabel as nib
from time import time

import matplotlib.pyplot as plt
from tomopy.misc import phantom as ph

from pystruct.inference import inference_dispatch, compute_energy
from pystruct.utils import make_grid_edges


def generate_checkerboard(shape=10):
    """Generate a checkerboard image."""
    # Checkerboard phantom
    L = ph.checkerboard(size=shape, dtype='uint8')[0, :, :]

    # Normalize and binarize
    L = np.round(L/255.).astype('uint8')

    return L


def generate_Potts(shape=(10, 10),
                   ncolors=2,
                   beta=1.0,
                   inference='max-product'):
    """Generate Potts image."""
    # Generate initial normal image
    x = rnd.normal(size=(*shape, ncolors))

    # Unary potentials
    unaries = x.reshape(-1, ncolors)

    # Pairwise potentials
    pairwise = beta*np.eye(ncolors)

    # Generate edge matrix
    edges = make_grid_edges(x)

    # Start clock
    start = time()

    # Infer image
    y = inference_dispatch(unaries, pairwise, edges,
                           inference_method=inference)

    # End clock
    took = time() - start
    print('Inference took ' + str(took) + ' seconds')

    # Compute energy
    energy = compute_energy(unaries, pairwise, edges, y)

    # Return inferred image and energy
    return np.reshape(y, shape), energy


def subject2image(fn, imsize=(256, 256), slice_ix=69, seg=False,
                  K=[0, 1, 2, 3], normalize=False):
    """Load subject images."""
    # Recognize file format
    file_fmt = fn[-3:]

    # Check for file format
    if file_fmt == 'raw':

        # Read binary and reshape to image
        im = np.fromfile(fn, count=np.prod(imsize), dtype='uint8')

        # Reshape and rotate
        im = nd.rotate(im.reshape(imsize), 90)

        if seg:

            # Restrict classes of segmentations
            labels = np.unique(im)
            for lab in np.setdiff1d(labels, K):
                im[im == lab] = 0

        else:

            # Normalize pixels
            if normalize:
                im[im < 0] = 0
                im[im > 255] = 255
                im = im / 255.

    elif file_fmt == 'nii':

        # Collect image and pixel dims
        im = nib.load(fn).get_data()
        hd = nib.load(fn).header
        zd = hd.get_zooms()

        if len(im.shape) == 4:
            im = im[:, :, :, 0]
            zd = zd[:-1]

        # Interpolate to normalized pixel dimension
        im = nd.zoom(im, zd, order=0)

        # Slice image
        im = im[:, :, slice_ix]

        # Transpose image
        im = im.T

        # Pad image
        pdl = np.ceil((np.array(imsize) - im.shape)/2.).astype('int64')
        pdh = np.floor((np.array(imsize) - im.shape)/2.).astype('int64')
        im = np.pad(im, ((pdl[0], pdh[0]), (pdl[1], pdh[1])), 'constant')

        # Normalize pixels
        if normalize:
            im[im < 0] = 0
            # im[im>hd.sizeof_hdr] = hd.sizeof_hdr
            im = im / float(1023)

    else:
        print('File format unknown')

    return im


def set_classes(Y, z, vis=False):
    """
    Set labels of classes predicted by the clustering methods.

    The clustering algorithms assign an arbitrary number to each class, but we
    want the numbering to be consistent with the labeling in the segmenation
    files. Given 1 labeled sample per class from the target scan, the numerical
    labeling can be corrected.

    Parameters
    ----------
    Y : array
        Segmentation of a scan.
    z : array (sparse format)
        3 voxel indices with their class labels.
    vis : bool
        Whether to visualize results (for sanity checking)

    Returns
    -------
    Y : array
        Segmentation with new numbers assigned to each class.

    """
    # Check number of classes
    classes = np.unique(Y)
    K = len(classes)

    # Check whether enough labels were provided
    if K > len(np.unique(z[:, 2])):
        raise ValueError('Not enough labeled samples provided.')

    Z = np.copy(Y)

    # Loop over target samples
    for k in range(z.shape[0]):

        # Check label currently assigned to target sample ij
        current_class = Y[z[k, 0], z[k, 1]]

        # Set values in segmentation
        Z[Y == current_class] = z[k, 2]

    if vis:
        # Initialize figure
        _, ax = plt.subplots(ncols=2, figsize=(15, 5))

        # Plot segmentation
        im0 = ax[0].imshow(Y)
        ax[0].set_title('Clustering')
        plt.colorbar(im0, ax=ax[0])

        # Plot observation
        im1 = ax[1].imshow(Z)
        ax[1].set_title('Cluster re-numbering')
        plt.colorbar(im1, ax=ax[1])

        plt.show()

    return Z
