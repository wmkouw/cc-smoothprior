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

import keras.models as km
import keras.layers as kl
import keras.regularizers as kr
import keras.utils as ku

import sklearn.model_selection as sm
import sklearn.linear_model as sl
import sklearn.svm as sv
import sklearn.feature_extraction.image as sf

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


def pad2size(I, size):
    """Pad an image to fit a given size."""
    # Check image size
    imsize = I.shape

    # Get pad lenght and heights
    pdl = np.ceil((np.array(size) - imsize) / 2)
    pdh = np.floor((np.array(size) - imsize) / 2)

    # Cast to integers
    pdl = pdl.astype('int8')
    pdh = pdh.astype('int8')

    # Pad image
    if len(imsize) == 3:
        return np.pad(I, ((pdl[0], pdh[0]),
                          (pdl[1], pdh[1]),
                          (pdl[2], pdh[2])), 'constant')
    else:
        return np.pad(I, ((pdl[0], pdh[0]),
                          (pdl[1], pdh[1])), 'constant')


def subject2image(fn, imsize=(256, 256),
                  slice_ix=69,
                  slice_dim=2,
                  flipud=False,
                  seg=False,
                  CMA=False,
                  K=[0, 1, 2, 3],
                  normalize=False):
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
        if slice_dim == 0:
            im = im[slice_ix, :, :]

        elif slice_dim == 1:
            im = im[:, slice_ix, :]

        elif slice_dim == 2:
            im = im[:, :, slice_ix]

        else:
            raise ValueError('Slice dim is too high.')

        # Transpose image
        im = im.T

        # Flip image if necessary
        if flipud:
            im = np.flipud(im)

        # Pad image
        im = pad2size(im, imsize)

        # Normalize pixels
        if normalize:
            im[im < 0] = 0
            # im[im>hd.sizeof_hdr] = hd.sizeof_hdr
            im = im / float(1023)

        if CMA:
            # Restrict classes of segmentations
            im = CMA_to_4classes(im)

    else:
        print('File format unknown')

    return im


def ConvolutionalNetwork(num_classes,
                         input_shape=(256, 256),
                         num_filters=(8, 8),
                         l2=0.0001):
    """
    Construct classifier with optimal regularization parameter.

    Parameters
    ----------
    input_shape : tuple
        Shape of input images (def: (256, 256))

    Returns
    -------
    net : Keras object
        Compiled vanilla CNN

    """
    # Start sequential model
    net = km.Sequential()

    # Convolutional part
    net.add(kl.Conv2D(num_filters[0],
                      kernel_size=(3, 3),
                      activation='relu',
                      padding='valid',
                      kernel_regularizer=kr.l2(l2),
                      input_shape=input_shape))
    net.add(kl.MaxPooling2D(pool_size=(2, 2), padding='valid'))
    net.add(kl.Dropout(0.2))
    for f in range(len(num_filters) - 1):
        net.add(kl.Conv2D(num_filters[f + 1],
                          kernel_size=(3, 3),
                          activation='relu',
                          padding='valid',
                          kernel_regularizer=kr.l2(l2)))
        net.add(kl.MaxPooling2D(pool_size=(2, 2), padding='valid'))

    # Fully-connected part
    net.add(kl.Flatten())
    net.add(kl.Dense(8, activation='relu',
                     kernel_regularizer=kr.l2(l2)))
    net.add(kl.Dense(num_classes, activation='softmax'))

    # Compile network architecture
    net.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

    return net


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


def filter_Sobel(im):
    """Filter an image with its spatial derivatives."""
    # x-direction Sobel filter response
    imx = nd.filters.sobel(im, 1)

    # y-direction Sobel filter response
    imy = nd.filters.sobel(im, 0)

    # Return magnitude of spatial derivates
    return np.sqrt(imx**2 + imy**2)


def reject_outliers(data, m=2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s < m]


def CMA_to_4classes(L):
    """
    Map CMA's automatic segmentation to {BCK,CSF,GM,WM}.

    CMA's automatic segmentation protocol lists over 80 different tissue
    classes. Here we map these back to the four we used in the paper:
    background (BCK), cerebrospinal fluid (CSF), gray matter (GM) and
    white matter (WM). Sets Brainstem and Cerebellum to background
    (16=0, 6,7,8,45,46,47=0).

    Parameters
    ----------
    L : array
        Label matrix

    Returns
    -------
    L : array
        Label matrix

    """
    # Number of subjects
    nI = L.shape[0]

    # Re-map to
    L = -L
    for i in range(nI):
        L[i][L[i] == -0] = 0
        L[i][L[i] == -1] = 0
        L[i][L[i] == -2] = 3
        L[i][L[i] == -3] = 2
        L[i][L[i] == -4] = 1
        L[i][L[i] == -5] = 1
        L[i][L[i] == -6] = 0
        L[i][L[i] == -7] = 0
        L[i][L[i] == -8] = 0
        L[i][L[i] == -9] = 2
        L[i][L[i] == -10] = 2
        L[i][L[i] == -11] = 2
        L[i][L[i] == -12] = 2
        L[i][L[i] == -13] = 2
        L[i][L[i] == -14] = 1
        L[i][L[i] == -15] = 1
        L[i][L[i] == -16] = 0
        L[i][L[i] == -17] = 2
        L[i][L[i] == -18] = 2
        L[i][L[i] == -19] = 2
        L[i][L[i] == -20] = 2
        L[i][L[i] == -21] = 0
        L[i][L[i] == -22] = 0
        L[i][L[i] == -23] = 2
        L[i][L[i] == -24] = 1
        L[i][L[i] == -25] = 0
        L[i][L[i] == -26] = 2
        L[i][L[i] == -27] = 2
        L[i][L[i] == -28] = 2
        L[i][L[i] == -29] = 0
        L[i][L[i] == -30] = 0
        L[i][L[i] == -31] = 0
        L[i][L[i] == -32] = 0
        L[i][L[i] == -33] = 0
        L[i][L[i] == -34] = 0
        L[i][L[i] == -35] = 0
        L[i][L[i] == -36] = 0
        L[i][L[i] == -37] = 0
        L[i][L[i] == -38] = 0
        L[i][L[i] == -39] = 0
        L[i][L[i] == -40] = 0
        L[i][L[i] == -41] = 3
        L[i][L[i] == -42] = 2
        L[i][L[i] == -43] = 1
        L[i][L[i] == -44] = 1
        L[i][L[i] == -45] = 0
        L[i][L[i] == -46] = 0
        L[i][L[i] == -47] = 0
        L[i][L[i] == -48] = 2
        L[i][L[i] == -49] = 2
        L[i][L[i] == -50] = 2
        L[i][L[i] == -51] = 2
        L[i][L[i] == -52] = 2
        L[i][L[i] == -53] = 2
        L[i][L[i] == -54] = 2
        L[i][L[i] == -55] = 2
        L[i][L[i] == -56] = 2
        L[i][L[i] == -57] = 0
        L[i][L[i] == -58] = 2
        L[i][L[i] == -59] = 2
        L[i][L[i] == -60] = 2
        L[i][L[i] == -61] = 0
        L[i][L[i] == -62] = 0
        L[i][L[i] == -63] = 0
        L[i][L[i] == -64] = 0
        L[i][L[i] == -65] = 0
        L[i][L[i] == -66] = 0
        L[i][L[i] == -67] = 0
        L[i][L[i] == -68] = 0
        L[i][L[i] == -69] = 0
        L[i][L[i] == -70] = 0
        L[i][L[i] == -71] = 0
        L[i][L[i] == -72] = 0
        L[i][L[i] == -73] = 0
        L[i][L[i] == -74] = 0
        L[i][L[i] == -75] = 0
        L[i][L[i] == -76] = 0
        L[i][L[i] == -77] = 0
        L[i][L[i] == -78] = 0
        L[i][L[i] == -79] = 0
        L[i][L[i] == -80] = 0
        L[i][L[i] == -81] = 0
        L[i][L[i] == -82] = 0
        L[i][L[i] == -83] = 0
        L[i][L[i] == -84] = 3

    return L