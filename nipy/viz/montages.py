''' Routines for making image montage arrays from images and arrays

The module has routines for doing things like displaying 8 slices from
an image in a montage with 4 columns and 2 rows.

Inspired by ``plot_activations_simplepanel.py`` script by Benjamin
Thyreau
'''

import numpy as np

from ..algorithms.resample import resample_img2img


class MontageError(Exception):
    pass


def array_montage(arr, slicedef=None, n_columns=None, axis=-1):
    ''' 2D pixel montage array from 3D `arr` array

    Parameters
    ----------
    arr : array-like
       3 dimensional array representing 3 dimensional image
    slicedef : None or (sequence of int) or slice, optional
       slices to take over `axis` dimension of array.  If None, use all
       slices in volume.  If sequence, then take slices indicated by
       integers in sequence.  If slice, take slices over `axis`
       dimension of array.  For example, if ``axis==-1``, this would
       result in: ``arr[:,:,slice]``.
    n_columns : int, optional
       number of columns for the montage.  If ``nslices`` is the number
       of slices implied in `slicedef`, then the number of rows is given
       by ``np.ceil(nslices / n_columns)`` - that is, we round up the
       number of rows to include all the slices, and pad out any missing
       slices for the row with blank (0) slices.  By default we adjust
       `n_columns` for the montage to be nearly square
    axis : int, optional
       axis defining slices.  Default is -1 (the last).
    
    Returns
    -------
    montage : array
       2D pixel array with slices arranged in column major order.  So,
       with `n_columns` == 2, and 4 slices implied in `slicedef`, the
       slices will be arranged so that slice 1 will be to the left of
       slice 0, and slice 2 will be below slice 0.   To suit
       neuroimaging convention (first dimension shown left to right) we
       rotate each slice in the montage by 90 degrees counter clockwise
    '''
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise MontageError('Array should have 3 dimensions')
    if axis is None:
        raise MontageError('Axis cannot be none')
    # roll slicing axis to first for convenience
    arr = np.rollaxis(arr, axis)
    # select our slices
    if slicedef is None:
        slices = list(arr)
    else:
        try: # iterable 
            slicedef = list(slicedef)
        except TypeError: # we hope it does slicing
            slices = list(arr[slicedef])
        else: # iterable
            slices = [arr[sno] for sno in slicedef]
    n_slices = len(slices)
    if n_columns is None:
        n_columns = int(np.floor(np.sqrt(n_slices)))
    n_rows = int(np.ceil(n_slices / float(n_columns)))
    slice_shape = slices[0].shape[::-1] # reversed to reflect rotation
    blank_slice = np.zeros(slice_shape)
    rows = []
    for rn in range(n_rows):
        row = []
        for cn in range(n_columns):
            if slices:
                row.append(np.rot90(slices.pop(0)))
            else:
                row.append(blank_slice)
        rows.append(np.hstack(row))
    return np.vstack(rows)


def array_samples(arr, samples=16, n_columns=None, axis=-1):
    ''' Utility routine to plot a range of slices from an array-like

    
    This is more or less the algorithm from
    plot_activations_simplepanel.py

    Parameters
    ----------
    arr : array-like
       3 dimensional array
    samples : int, optional
       number of sample slices to take from array.  We take `samples`
       samples from the central 80% of the array
    n_columns : None or int, optional
       number of columns in the resulting montage.  If None, generate a
       montage that is close to square
    axis : int, optional
       Axis over which to show slices.  Default is -1 (last axis)
    '''
    arr = np.asarray(arr)
    slice_ax_len = arr.shape[axis]
    # select some slices through the image
    slicedef = list(np.round(
            np.linspace(0.2, 0.8, samples) * slice_ax_len).astype('i'))
    return array_montage(arr, slicedef, n_columns=n_columns, axis=axis)


def show_montage(montage, mpl_axis=None, **kwargs):
    ''' show `montage` using matplotlib axis `mpl_axis` or new axis

    Parameters
    ----------
    montage : array
       2D pixel array of image montage
    mpl_axis : None or matplotlib ``axis`` instance
       Axis onto which to draw the montage.  If None (the default),
       create a new figure and axes to draw the montage
    **kwargs :
       extra parameters for matplotlib ``imshow`` command

    Returns
    -------
    mpl_axis : matplotlib ``axis`` instance
       Same as input `mpl_axis` unless this was None, in which case we
       return the newly created axis instance. 
    '''
    import pylab as pl
    if mpl_axis is None:
        fig = pl.figure()
        mpl_axis = p.subplot((1,1,1))
    mpl_axis.imshow(montage, cmap='gray', origin='upper',
                    aspect='auto', **kwargs)
    mpl_axis.axis = 'off'
    return mpl_axis


def yoked_image_samples(img1, img2, samples=16, n_columns=None, axis=-1):
    ''' Show yoked montage of two NIPY images in the same figure

    Parameters
    ----------
    img1 : nipy ``Image``
       image for left hand subolot. We resample `img2` to match `img1`
    img2 : nipy ``Image``
       image for right hand subplot
    samples : int, optional
       number of sample slices to take from array.  We take `samples`
       samples from the central 80% of the array
    n_columns : None or int, optional
       number of columns in the resulting montage.  If None, generate a
       montage that is close to square
    axis : int, optional
       Axis over which to show slices.  Default is -1 (last axis)

    Returns
    -------
    mpl_axes : tuple
       The two matplotlib axis objects created for figure
    '''
    import pylab as pl
    img2_resample = resample_img2img(source=img2, target=img1)
    fig = pl.figure()
    ax1 = pl.subplot(1,2,1)
    montage1 = array_samples(img1, samples, n_columns, axis)
    show_montage(montage1, ax1)
    ax2 = pl.subplot(1,2,2)
    montage2 = array_samples(img2_resample, samples, n_columns, axis)
    show_montage(montage2, ax2)
    return (ax1, ax2)

    
