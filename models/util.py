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
        im = pad2size(im, imsize)

        # Normalize pixels
        if normalize:
            im[im < 0] = 0
            # im[im>hd.sizeof_hdr] = hd.sizeof_hdr
            im = im / float(1023)

    else:
        print('File format unknown')

    return im

def set_classifier(X, y, classifier='lr', c_range=np.logspace(-5, 4, 10),
                   num_folds=2):
    """
    Construct classifier with optimal regularization parameter.

    Parameters
    ----------
    X : array
        Data matrix, number of samples by number of features.
    y : array
        Label vector or matrix, number of samples by 1, or number of samples by
        number of classes.
    classifier : str
        Type of classifier to use, options are: 'lr' (logistic regression),
        'linsvc' (linear support vector classifier),
        'rbfsvc' (radial basis function support vector classifier),
        'cnn' (convolutional neural network), (def='lr').
    c_range : array
        Range of possible regularization parameters,
        (def=np.logspace(-5, 4, 10)).
    num_folds : int
        Number of folds to use in cross-validation.

    Returns
    -------
    sklearn classifier
        A trained classifier with a predict function.

    """
    # Data shape
    N = X.shape[0]

    # Number of classes
    if len(y.shape) == 2:

        # Check number of classes by number of columns of label matrix
        num_classes = y.shape[1]

    else:
        # Check number of classes by the number of unique entries
        num_classes = len(np.unique(y))

    if classifier == 'lr':

        # Grid search over regularization parameters C
        if N > num_classes:

            # Set up a grid search cross-validation procedure
            modelgs = sm.GridSearchCV(sl.LogisticRegression(),
                                      param_grid=dict(C=c_range),
                                      cv=num_folds)

            # Fit grid search model
            modelgs.fit(X, y)

            # Extract best regularization parameter
            bestC = modelgs.best_estimator_.C

        else:
            # Standard regularization parameter
            bestC = 1

        # Set up model with optimal C
        return sl.LogisticRegression(C=bestC)

    elif classifier == 'linsvc':

        # Grid search over regularization parameters C
        if N > num_classes:

            # Set up a grid search cross-validation procedure
            modelgs = sm.GridSearchCV(sv.LinearSVC(),
                                      param_grid=dict(C=c_range),
                                      cv=num_folds)

            # Fit grid search model
            modelgs.fit(X, y)

            # Extract best regularization parameter
            bestC = modelgs.best_estimator_.C

        else:
            # Standard regularization parameter
            bestC = 1

        # Train model with optimal C
        return sv.LinearSVC(C=bestC)

    elif classifier == 'rbfsvc':

        # Grid search over regularization parameters C
        if N > num_classes:

            # Set up a grid search cross-validation procedure
            modelgs = sm.GridSearchCV(sv.SVC(), cv=num_folds,
                                      param_grid=dict(C=c_range))

            # Fit grid search model
            modelgs.fit(X, y)

            # Extract best regularization parameter
            bestC = modelgs.best_estimator_.C

        else:
            # Standard regularization parameter
            bestC = 1

        # Train model with optimal C
        return sv.SVC(C=bestC)

    elif classifier == 'cnn':

        # Start sequential model
        net = km.Sequential()

        # Convolutional part
        net.add(kl.Conv2D(8, kernel_size=(3, 3),
                          activation='relu',
                          padding='valid',
                          kernel_regularizer=kr.l2(0.001),
                          input_shape=(X.shape[1], X.shape[2], 1)))
        net.add(kl.MaxPooling2D(pool_size=(2, 2), padding='valid'))
        net.add(kl.Dropout(0.2))
        net.add(kl.Flatten())

        # Fully-connected part
        net.add(kl.Dense(16, activation='relu',
                         kernel_regularizer=kr.l2(0.001)))
        net.add(kl.Dropout(0.2))
        net.add(kl.Dense(8, activation='relu',
                         kernel_regularizer=kr.l2(0.001)))
        net.add(kl.Dense(num_classes, activation='softmax'))

        # Compile network architecture
        net.compile(loss='categorical_crossentropy',
                    optimizer='rmsprop',
                    metrics=['accuracy'])

        return net

    else:
        print('Classifier unknown')


def classify(X, y, val=[], classifier='lr', c_range=np.logspace(-5, 4, 10),
             num_folds=2, num_epochs=2, batch_size=32, verbose=True):
    """
    Classify sets of patches.

    Parameters
    ----------
    X : array
        Data matrix, number of samples by number of features.
    y : array
        Label vector or matrix, number of samples by 1, or number of samples by
        number of classes.
    val : list[X, y]
        Validation data and label set, (def = []).
    classifier : str
        Type of classifier to use, options are: 'lr' (logistic regression),
        'linsvc' (linear support vector classifier),
        'rbfsvc' (radial basis function support vector classifier),
        'cnn' (convolutional neural network), (def='lr').
    c_range : array
        Range of possible regularization parameters,
        (def=np.logspace(-5, 4, 10)).
    num_folds : int
        Number of folds to use in cross-validation.
    num_epochs : int
        Number of epochs in training a neural network.
    batch_size : int
        Size of the batches to cut the data set into.
    verbose : bool
        Verbosity during training.

    Returns
    -------
    err : float
        Classification error.

    """
    # Number of classes
    num_classes = len(np.unique(y))

    if classifier == 'cnn':

        # Switch labels to categorical array
        y = ku.to_categorical(y - np.min(y), num_classes)

    else:
        # If data has more than 2 dimensions, reshape it back to 2.
        if len(X.shape) > 2:
            X = X.reshape((X.shape[0], -1))

    # Train regularized classifier
    model = set_classifier(X, y, classifier=classifier, num_folds=num_folds,
                           c_range=c_range)

    if classifier.lower() in ['lr', 'linsvc', 'rbfsvc']:

        if val:

            # Number of validation samples
            N = val[0].shape[0]

            # Error on validation data
            err = 1 - model.fit(X, y).score(val[0].reshape((N, -1)), val[1])

        else:
            # Error cross-validated over training data
            err = np.mean(1 - sm.cross_val_score(model, X, y=y, cv=num_folds))

    elif classifier == 'cnn':

        if val:
            # Validation
            valy = ku.to_categorical(val[1] - np.min(val[1]), num_classes)

            # Fit model
            model.fit(X, y, batch_size=batch_size, epochs=num_epochs,
                      validation_split=0.2, shuffle=True)

            # Error on validation data
            err = 1 - model.test_on_batch(val[0], valy)[1]

        else:
            # Error cross-validated over training data
            err = np.mean(1 - sm.cross_val_score(model, X, y=y, cv=num_folds))
    else:
        raise ValueError('Classifier unknown')

    # Report
    if verbose:
        print('Classification error = ' + str(err))

    return err


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


def imfilter_Sobel(im):
    """Filter an image with its spatial derivatives."""
    # x-direction Sobel filter response
    imx = nd.filters.sobel(im, 1)

    # y-direction Sobel filter response
    imy = nd.filters.sobel(im, 0)

    # Return magnitude of spatial derivates
    return np.sqrt(imx**2 + imy**2)
