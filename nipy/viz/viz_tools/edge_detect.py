# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Edget detection routines
"""
import warnings

import numpy as np
from scipy import ndimage, signal

try:
    # Protect this import as it is compiled code
    from nipy.algorithms.statistics import quantile
except ImportError, e:
    warnings.warn('Could not import fast quantile function: %s' % e)
    quantile = None

                   
################################################################################
# Edge detection

def _fast_abs_percentile(map, percentile=80):
    """ A fast version of the percentile of the absolute value.
    """
    if hasattr(map, 'mask'):
        map = np.asarray(map[np.logical_not(map.mask)])
    map = np.abs(map)
    map = map.ravel()
    if quantile is not None:
        return quantile(map, .01*percentile)
    map.sort()
    nb = map.size
    return map[.01*percentile*nb]


def _orientation_kernel(t):
    """ structure elements for calculating the value of neighbors in several 
        directions
    """ 
    sin = np.sin
    pi  = np.pi
    t = pi*t
    arr =  np.array([[sin(t),        sin(t+.5*pi),  sin(t+pi)    ],
                     [sin(t+1.5*pi), 0,             sin(t+1.5*pi)],
                     [sin(t+pi),     sin(t+.5*pi),  sin(t)       ]])
    return np.round(.5*((1+arr))**2).astype(np.bool)


def _edge_detect(image, high_threshold=.75, low_threshold=.4):
    """ Edge detection for 2D images based on Canny filtering.

        Parameters
        ==========
        image: 2D array
            The image on which edge detection is applied
        high_threshold: float, optional
            The quantile defining the upper threshold of the hysteries 
            thresholding: decrease this to keep more edges
        low_threshold: float, optional
            The quantile defining the lower threshold of the hysteries 
            thresholding: decrease this to extract wider edges

        Returns
        ========
        grad_mag: 2D array of floats
            The magnitude of the gradient
        edge_mask: 2D array of booleans
            A mask of where have edges been detected

        Notes
        ======
        This function is based on a Canny filter, however it has been
        taylored to visualization purposes on brain images: don't use it
        in the general case.

        It computes the norm of the gradient, extracts the ridge by
        keeping only local maximum in each direction, and performs
        hysteresis filtering to keep only edges with high gradient
        magnitude.
    """
    # This code is loosely based on code by Stefan van der Waalt
    # Convert to floats to avoid overflows
    np_err = np.seterr(all='ignore')
    img = signal.wiener(image.astype(np.float))
    np.seterr(**np_err)
    # Where the noise variance is 0, Wiener can create nans
    img[np.isnan(img)] = image[np.isnan(img)]
    img /= img.max()
    grad_x = ndimage.sobel(img, mode='constant', axis=0)
    grad_y = ndimage.sobel(img, mode='constant', axis=1)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_angle = np.arctan2(grad_y, grad_x)
    # Scale the angles in the range [0, 2]
    grad_angle = (grad_angle + np.pi) / np.pi
    # Non-maximal suppression: an edge pixel is only good if its magnitude is
    # greater than its neighbors normal to the edge direction.
    thinner = np.zeros(grad_mag.shape, dtype=np.bool)
    for angle in np.arange(0, 2, .25):
        thinner = thinner | (
                (grad_mag > .85*ndimage.maximum_filter(grad_mag,
                                    footprint=_orientation_kernel(angle))) 
                & (((grad_angle - angle) % 2) < .75)
               )
    # Remove the edges next to the side of the image: they are not reliable
    thinner[0]     = 0
    thinner[-1]    = 0
    thinner[:, 0]  = 0
    thinner[:, -1] = 0

    thinned_grad = thinner * grad_mag
    # Hysteresis thresholding: find seeds above a high threshold, then
    # expand out until we go below the low threshold
    grad_values = thinned_grad[thinner]
    high = thinned_grad > _fast_abs_percentile(grad_values, 100*high_threshold)
    low = thinned_grad >  _fast_abs_percentile(grad_values, 100*low_threshold)
    edge_mask = ndimage.binary_dilation(high, structure=np.ones((3, 3)), 
                                        iterations=-1, mask=low)
    return grad_mag, edge_mask


def _edge_map(image):
    """ Return a maps of edges suitable for visualization.

        Parameters
        ==========
        image: 2D array
            The image that the edges are extracted from.

        Returns
        ========
        edge_mask: 2D masked array
            A mask of the edge as a masked array with parts without
            edges masked and the large extents detected with lower
            coefficients.
    """
    edge_mask = _edge_detect(image)[-1]
    edge_mask = edge_mask.astype(np.float)
    edge_mask = -np.sqrt(ndimage.distance_transform_cdt(edge_mask))
    edge_mask[edge_mask != 0] -= -.05+edge_mask.min()
    edge_mask = np.ma.masked_less(edge_mask, .01)
    return edge_mask


