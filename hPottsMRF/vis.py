"""
Set of visualization functions.

Author: Wouter M. Kouw
Last updated: 30-10-2018
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_segmentation(Y_hat, show=False, savefn=''):
    """Plot true segmentation, observation and estimated segmentation."""
    # Initialize subplots
    fg, ax = plt.subplots(ncols=1, figsize=(10, 10))

    # Plot prediction
    im = ax.imshow(Y_hat)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

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

    # Plot observation
    im1 = ax[1].imshow(X, vmin=0.0, vmax=1.0, cmap='bone')
    ax[1].set_title('X')
    divider = make_axes_locatable(ax[1])
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax1)

    # Plot prediction
    im2 = ax[2].imshow(Y_hat)
    ax[2].set_title('Y_hat')
    divider = make_axes_locatable(ax[2])
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax2)

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

    # Plot observation
    im1 = ax[1].imshow(X, vmin=0.0, vmax=1.0, cmap='bone')
    ax[1].set_title('X')
    divider = make_axes_locatable(ax[1])
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax1)

    # Plot prediction
    im2 = ax[2].imshow(Y_hat)
    ax[2].set_title('Y_hat')
    divider = make_axes_locatable(ax[2])
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax2)

    # Plot second prediction
    im3 = ax[3].imshow(Y_hat2)
    ax[3].set_title('Y_hat2')
    divider = make_axes_locatable(ax[3])
    cax3 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im3, cax=cax3)

    if show:
        # Pause to show figure
        plt.show()

    if savefn:
        # Save figure without padding
        fg.savefig(savefn, bbox_inches='tight', pad_inches=0.0)
