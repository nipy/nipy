''' Routines for making image montage arrays from images and arrays

The module has routines for doing things like displaying 8 slices from
an image in a montage with 4 columns and 2 rows.
'''

import numpy as np


class MontageError(Exception):
    pass


def array_montage(arr, slicedef=None, n_columns=None, axis=-1):
    ''' 2D pixel array from 3D `arr` array

    Parameters
    ----------
    arr : array-like
       3 dimensional array representing 3 dimensional image
    slicedef : None or (sequence of int) or slice, optional
       slices to take over `axis` dimension of array.  If None, use all
       slices in volume.  If sequence, then take slices indicated by
       integers in sequence.  If slice, and take slices over `axis`
       dimension of array.  For example, if ``axis==-1``, this would
       result in: ``arr[:,:,slice]``.
    n_columns : int, optional
       number of columns for the montage.  If ``nslices`` is the number
       of slices implied in `slicedef`, then the number of columns is
       given by ``np.ceil(nslices / n_columns)`` - that is, we round up
       the number of rows to include all the slices, and pad out any
       missing slices for the row with blank (0) slices.  By default we
       adjust `n_columns` for the montage to be nearly square
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
    # roll slicing axis to first for convenience
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise MontageError('Array should have 3 dimensions')
    arr = np.rollaxis(arr, axis)
    # select our slices
    if slicedef is None:
        slices = list(arr)
    else:
        try: # iterable 
            slicedef = list(slicedef)
        except TypeError: # we hope it's a slice
            slices = list(arr[slicedef])
        else: # iterable
            slices = [arr[sno] for sno in slicedef]
    n_slices = len(slices)
    if n_columns is None:
        n_columns = int(np.floor(np.sqrt(n_slices)))
    n_rows = int(np.ceil(n_slices / float(n_columns)))
    slice_shape = arr.shape[1:][::-1] # reflecting rot90
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


def show_array_samples(arr, samples=16, n_columns=None, axis=-1):
    ''' Utility routine to plot a range of slices from an array

    Uses pylab.imshow to display montage.

    Parameters
    ----------
    arr : array-like
       3 dimensional array
    samples : int, optional
       number of slices to take from array.  `samples` slices taken
       between 20% and top 20% of slices.
    n_columns : None or int, optional
       number of columns in the resulting montage.  If None, generate a
       montage that is close to square
    axis : int, optional
       Axis over which to show slices.  Default is -1 (last axis)
    '''
    import pylab as pl
    arr = np.asarray(arr)
    arr = np.rollaxis(arr, axis)
    # select some slices through the image
    slicedef = list(np.round(
            np.linspace(0.2, 0.8, samples) * arr.shape[0]).astype('i'))
    montage = array_montage(arr, slicedef, n_columns=n_columns, axis=0)
    # show in matplotlib
    pl.imshow(montage, cmap='gray', origin='upper', aspect='auto')
    
