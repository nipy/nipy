# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
''' Time series diagnostics

These started life as ``tsdiffana.m`` - see
http://imaging.mrc-cbu.cam.ac.uk/imaging/DataDiagnostics

Oliver Josephs (FIL) gave me (MB) the idea of time-point to time-point
subtraction as a diagnostic for motion and other sudden image changes.
'''

import numpy as np

from ...io.api import as_image

from ...core.reference.coordinate_map import (io_axis_indices, drop_io_dim, AxisError)


def time_slice_diffs(arr, time_axis=-1, slice_axis=None):
    ''' Time-point to time-point differences over volumes and slices

    We think of the passed array as an image.  The image has a "time"
    dimension given by `time_axis` and a "slice" dimension, given by
    `slice_axis`, and one or more other dimensions.  In the case of imaging
    there will usually be two more dimensions (the dimensions defining the size
    of an image slice). A single slice in the time dimension we call a "volume".
    A single entry in `arr` is a "voxel".  For example, if `time_axis` == 0,
    then ``v = arr[0]`` would be the first volume in the series.  The volume
    ``v`` above has ``v.size`` voxels.  If, in addition, `slice_axis` == 1, then
    for the volume ``v`` (above) ``s = v[0]`` would be a "slice", with
    ``s.size`` voxels. These are obviously terms from neuroimaging.

    Parameters
    ----------
    arr : array_like
       Array over which to calculate time and slice differences.  We'll
       call this array an 'image' in this doc.
    time_axis : int, optional
       axis of `arr` that varies over time. Default is last
    slice_axis : None or int, optional
       axis of `arr` that varies over image slice.  None gives last non-time
       axis.

    Returns
    -------
    results : dict

        ``T`` is the number of time points (``arr.shape[time_axis]``)

        ``S`` is the number of slices (``arr.shape[slice_axis]``)

        ``v`` is the shape of a volume (``rollimg(arr, time_axis)[0].shape``)

        ``d2[t]`` is the volume of squared differences between voxels at
        time point ``t`` and time point ``t+1``

        `results` has keys:

        * 'volume_mean_diff2' : (T-1,) array
           array containing the mean (over voxels in volume) of the
           squared difference from one time point to the next
        * 'slice_mean_diff2' : (T-1, S) array
           giving the mean (over voxels in slice) of the difference from
           one time point to the next, one value per slice, per
           timepoint
        * 'volume_means' : (T,) array
           mean over voxels for each volume ``vol[t] for t in 0:T``
        * 'slice_diff2_max_vol' : v[:] array
           volume, of same shape as input time point volumes, where each slice
           is is the slice from ``d2[t]`` for t in 0:T-1, that has the largest
           variance across ``t``.  Thus each slice in the volume may well result
           from a different difference time point.
        * 'diff2_mean_vol`` : v[:] array
           volume with the mean of ``d2[t]`` across t for t in 0:T-1.

    Raises
    ------
    ValueError : if `time_axis` refers to same axis as `slice_axis`
    '''
    arr = np.asarray(arr)
    ndim = arr.ndim
    # roll time axis to 0, slice axis to 1 for convenience
    if time_axis < 0:
        time_axis += ndim
    if slice_axis is None:
        slice_axis = ndim-2 if time_axis == ndim-1 else ndim-1
    elif slice_axis < 0:
        slice_axis += ndim
    if time_axis == slice_axis:
        raise ValueError('Time axis refers to same axis as slice axis')
    arr = np.rollaxis(arr, time_axis)
    # we may have changed the position of slice_axis
    if time_axis > slice_axis:
        slice_axis += 1
    arr = np.rollaxis(arr, slice_axis, 1)
    # shapes of things
    shape = arr.shape
    T = shape[0]
    S = shape[1]
    vol_shape = shape[1:]
    # loop over time points to save memory
    volds = np.empty((T-1,))
    sliceds = np.empty((T-1,S))
    means = np.empty((T,))
    diff_mean_vol = np.zeros(vol_shape)
    slice_diff_max_vol = np.zeros(vol_shape)
    slice_diff_maxes = np.zeros(S)
    last_tp = arr[0]
    means[0] = last_tp.mean()
    for dtpi in range(0,T-1):
        tp = arr[dtpi+1] # shape vol_shape
        means[dtpi+1] = tp.mean()
        dtp_diff2 = (tp - last_tp)**2
        diff_mean_vol += dtp_diff2
        sliceds[dtpi] = dtp_diff2.reshape(S, -1).mean(-1)
        # check whether we have found a highest-diff slice
        sdmx_higher = sliceds[dtpi] > slice_diff_maxes
        if any(sdmx_higher):
            slice_diff_maxes[sdmx_higher] = sliceds[dtpi][sdmx_higher]
            slice_diff_max_vol[sdmx_higher] = dtp_diff2[sdmx_higher]
        last_tp = tp
    volds = sliceds.mean(1)
    diff_mean_vol /= (T-1)
    # roll vol shapes back to match input
    diff_mean_vol = np.rollaxis(diff_mean_vol, 0, slice_axis)
    slice_diff_max_vol = np.rollaxis(slice_diff_max_vol, 0, slice_axis)
    return {'volume_mean_diff2': volds,
            'slice_mean_diff2': sliceds,
            'volume_means': means,
            'diff2_mean_vol': diff_mean_vol,
            'slice_diff2_max_vol': slice_diff_max_vol}


def time_slice_diffs_image(img, time_axis='t', slice_axis='slice'):
    """ Time-point to time-point differences over volumes and slices of image

    Parameters
    ----------
    img : Image
        The image on which to perform time-point differences
    time_axis : str or int, optional
        Axis indexing time-points. Default is 't'. If `time_axis` is an integer,
        gives the index of the input (domain) axis of `img`. If `time_axis` is a str,
        can be an input (domain) name, or an output (range) name, that maps to
        an input (domain) name.
    slice_axis : str or int, optional
        Axis indexing MRI slices. If `slice_axis` is an integer, gives the
        index of the input (domain) axis of `img`. If `slice_axis` is a str,
        can be an input (domain) name, or an output (range) name, that maps to
        an input (domain) name.

    Returns
    -------
    results : dict

        `arr` refers to the array as loaded from `img`

        ``T`` is the number of time points (``img.shape[time_axis]``)

        ``S`` is the number of slices (``img.shape[slice_axis]``)

        ``v`` is the shape of a volume (``rollimg(img, time_axis)[0].shape``)

        ``d2[t]`` is the volume of squared differences between voxels at
        time point ``t`` and time point ``t+1``

        `results` has keys:

        * 'volume_mean_diff2' : (T-1,) array
           array containing the mean (over voxels in volume) of the
           squared difference from one time point to the next
        * 'slice_mean_diff2' : (T-1, S) array
           giving the mean (over voxels in slice) of the difference from
           one time point to the next, one value per slice, per
           timepoint
        * 'volume_means' : (T,) array
           mean over voxels for each volume ``vol[t] for t in 0:T``
        * 'slice_diff2_max_vol' : v[:] image
           image volume, of same shape as input time point volumes, where each
           slice is is the slice from ``d2[t]`` for t in 0:T-1, that has the
           largest variance across ``t``.  Thus each slice in the volume may
           well result from a different difference time point.
        * 'diff2_mean_vol`` : v[:] image
           image volume with the mean of ``d2[t]`` across t for t in 0:T-1.
    """
    img = as_image(img)
    img_class = img.__class__
    time_in_ax, time_out_ax = io_axis_indices(img.coordmap, time_axis)
    if None in (time_in_ax, time_out_ax):
        raise AxisError('Cannot identify matching input output axes with "%s"'
                        % time_axis)
    slice_in_ax, slice_out_ax = io_axis_indices(img.coordmap, slice_axis)
    if None in (slice_in_ax, slice_out_ax):
        raise AxisError('Cannot identify matching input output axes with "%s"'
                        % slice_axis)
    vol_coordmap = drop_io_dim(img.coordmap, time_axis)
    results = time_slice_diffs(img.get_data(), time_in_ax, slice_in_ax)
    for key in ('slice_diff2_max_vol', 'diff2_mean_vol'):
        vol = img_class(results[key], vol_coordmap)
        results[key] = vol
    return results
