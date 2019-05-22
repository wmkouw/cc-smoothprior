"""
Set of visualization functions.

Author: Wouter M. Kouw
Last updated: 30-10-2018
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_posteriors(posts, show=False, savefn=''):
    """Plot posteriors for each class."""
    # Number of classes
    K = posts.shape[2]

    # Initialize subplots
    fg, ax = plt.subplots(ncols=K, figsize=(K*10, 10))

    for k in range(K):

        # Plot prediction
        im = ax[k].imshow(posts[:, :, k], vmin=0.0, vmax=1.0)
        ax[k].set_title('Component ' + str(k))
        divider = make_axes_locatable(ax[k])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax[k].get_xaxis().set_visible(False)
        ax[k].get_yaxis().set_visible(False)

    if show:
        # Pause to show figure
        plt.show()

    if savefn:
        # Save figure without padding
        fg.savefig(savefn, bbox_inches='tight', pad_inches=0.0)


def plot_clustering(X, Y_hat,
                    color=[1, 0, 0],
                    mode='outer',
                    show=False,
                    savefn=''):
    """Plot boundaries produced by segmentation."""
    # Initialize subplots
    fg, ax = plt.subplots(ncols=1, figsize=(10, 10))

    # Plot prediction
    im = ax.imshow(mark_boundaries(X, Y_hat,
                                   color=color,
                                   mode=mode,
                                   background_label=0))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    if show:
        # Pause to show figure
        plt.show()

    if savefn:
        # Save figure without padding
        fg.savefig(savefn, bbox_inches='tight', pad_inches=0.0)

def plot_scan(X, show=False, savefn=''):
    """Plot scan."""
    # Initialize subplots
    fg, ax = plt.subplots(ncols=1, figsize=(10, 10))

    # Plot prediction
    im = ax.imshow(X, vmin=0.0, vmax=1.0, cmap='bone')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    if show:
        # Pause to show figure
        plt.show()

    if savefn:
        # Save figure without padding
        fg.savefig(savefn, bbox_inches='tight', pad_inches=0.0)

def plot_segmentation(Y_hat, show=False, savefn=''):
    """Plot true segmentation, observation and estimated segmentation."""
    # Initialize subplots
    fg, ax = plt.subplots(ncols=1, figsize=(10, 10))

    # Plot prediction
    im = ax.imshow(Y_hat)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    if show:
        # Pause to show figure
        plt.show()

    if savefn:
        # Save figure without padding
        fg.savefig(savefn, bbox_inches='tight', pad_inches=0.0)


def plot_segmentations(Y, X, Y_hat, show=False, savefn=''):
    """Plot true segmentation, observation and estimated segmentation."""
    # Initialize subplots
    fg, ax = plt.subplots(ncols=3, figsize=(15, 5))

    # Plot segmentation
    im0 = ax[0].imshow(Y)
    ax[0].set_title('Y')
    divider = make_axes_locatable(ax[0])
    cax0 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im0, cax=cax0)
    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)

    # Plot observation
    im1 = ax[1].imshow(X, vmin=0.0, vmax=1.0, cmap='bone')
    ax[1].set_title('X')
    divider = make_axes_locatable(ax[1])
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax1)
    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)

    # Plot prediction
    im2 = ax[2].imshow(Y_hat)
    ax[2].set_title('Y_hat')
    divider = make_axes_locatable(ax[2])
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax2)
    ax[2].get_xaxis().set_visible(False)
    ax[2].get_yaxis().set_visible(False)

    if show:
        # Pause to show figure
        plt.show()

    if savefn:
        # Save figure without padding
        fg.savefig(savefn, bbox_inches='tight', pad_inches=0.0)


def plot_2segmentations(Y, X, Y_hat, Y_hat2, show=False, savefn=''):
    """Plot 2 segmentations."""
    # Initialize subplots
    fg, ax = plt.subplots(ncols=4, figsize=(20, 5))

    # Plot segmentation
    im0 = ax[0].imshow(Y)
    ax[0].set_title('Y')
    divider = make_axes_locatable(ax[0])
    cax0 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im0, cax=cax0)
    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)

    # Plot observation
    im1 = ax[1].imshow(X, vmin=0.0, vmax=1.0, cmap='bone')
    ax[1].set_title('X')
    divider = make_axes_locatable(ax[1])
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax1)
    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)

    # Plot prediction
    im2 = ax[2].imshow(Y_hat)
    ax[2].set_title('Y_hat')
    divider = make_axes_locatable(ax[2])
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax2)
    ax[2].get_xaxis().set_visible(False)
    ax[2].get_yaxis().set_visible(False)

    # Plot second prediction
    im3 = ax[3].imshow(Y_hat2)
    ax[3].set_title('Y_hat2')
    divider = make_axes_locatable(ax[3])
    cax3 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im3, cax=cax3)
    ax[3].get_xaxis().set_visible(False)
    ax[3].get_yaxis().set_visible(False)

    if show:
        # Pause to show figure
        plt.show()

    if savefn:
        # Save figure without padding
        fg.savefig(savefn, bbox_inches='tight', pad_inches=0.0)
