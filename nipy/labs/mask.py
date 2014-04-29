# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Utilities for extracting masks from EPI images and applying them to time
series.
"""

# Major scientific libraries imports
import numpy as np
from scipy import ndimage

# Neuroimaging libraries imports
from nibabel import load, nifti1, save

from ..io.nibcompat import get_header, get_affine, get_unscaled_data

###############################################################################
# Operating on connect component
###############################################################################

def largest_cc(mask):
    """ Return the largest connected component of a 3D mask array.

    Parameters
    -----------
    mask: 3D boolean array
          3D array indicating a mask.

    Returns
    --------
    mask: 3D boolean array
          3D array indicating a mask, with only one connected component.
    """
    # We use asarray to be able to work with masked arrays.
    mask = np.asarray(mask)
    labels, label_nb = ndimage.label(mask)
    if not label_nb:
        raise ValueError('No non-zero values: no connected components')
    if label_nb == 1:
        return mask.astype(np.bool)
    label_count = np.bincount(labels.ravel().astype(np.int))
    # discard 0 the 0 label
    label_count[0] = 0
    return labels == label_count.argmax()


def threshold_connect_components(map, threshold, copy=True):
    """ Given a map with some coefficients set to zero, segment the
        connect components with number of voxels smaller than the
        threshold and set them to 0.

    Parameters
    ----------
    map: ndarray,
         The spatial map to segment
    threshold: scalar,
               The minimum number of voxels to keep a cluster.
    copy: bool, optional
          If copy is false, the input array is modified inplace

    Returns
    -------
    map: ndarray,
         the map with connected components removed
    """
    labels, _ = ndimage.label(map)
    weights = np.bincount(labels.ravel())
    if copy:
        map = map.copy()
    for label, weight in enumerate(weights):
        if label == 0:
            continue
        if weight < threshold:
            map[labels == label] = 0
    return map


###############################################################################
# Utilities to calculate masks
###############################################################################

def compute_mask_files(input_filename, output_filename=None,
                        return_mean=False, m=0.2, M=0.9, cc=1,
                        exclude_zeros=False, opening=2):
    """
    Compute a mask file from fMRI nifti file(s)

    Compute and write the mask of an image based on the grey level
    This is based on an heuristic proposed by T.Nichols:
    find the least dense point of the histogram, between fractions
    m and M of the total image histogram.

    In case of failure, it is usually advisable to increase m.

    Parameters
    ----------
    input_filename : string
        nifti filename (4D) or list of filenames (3D).
    output_filename : string or None, optional
        path to save the output nifti image (if not None).
    return_mean : boolean, optional
        if True, and output_filename is None, return the mean image also, as
        a 3D array (2nd return argument).
    m : float, optional
        lower fraction of the histogram to be discarded.
    M: float, optional
        upper fraction of the histogram to be discarded.
    cc: boolean, optional
        if cc is True, only the largest connect component is kept.
    exclude_zeros: boolean, optional
        Consider zeros as missing values for the computation of the
        threshold. This option is useful if the images have been
        resliced with a large padding of zeros.
    opening: int, optional
        Size of the morphological opening performed as post-processing

    Returns
    -------
    mask : 3D boolean array
        The brain mask
    mean_image : 3d ndarray, optional
        The main of all the images used to estimate the mask. Only
        provided if `return_mean` is True.
    """
    if isinstance(input_filename, basestring):
        # One single filename or image
        nim = load(input_filename)  # load the image from the path
        vol_arr = get_unscaled_data(nim)
        header = get_header(nim)
        affine = get_affine(nim)
        if vol_arr.ndim == 4:
            if isinstance(vol_arr, np.memmap):
                # Get rid of memmapping: it is faster.
                mean_volume = np.array(vol_arr, copy=True).mean(axis=-1)
            else:
                mean_volume = vol_arr.mean(axis=-1)
            # Make a copy, to avoid holding a reference on the full array,
            # and thus polluting the memory.
            first_volume = vol_arr[..., 0].copy()
        elif vol_arr.ndim == 3:
            mean_volume = first_volume = vol_arr
        else:
            raise ValueError('Need 4D file for mask')
        del vol_arr
    else:
        # List of filenames
        if len(list(input_filename)) == 0:
            raise ValueError('input_filename should be a non-empty '
                'list of file names')
        # We have several images, we do mean on the fly,
        # to avoid loading all the data in the memory
        # We do not use the unscaled data here?:
        # if the scalefactor is being used to record real
        # differences in intensity over the run this would break
        for index, filename in enumerate(input_filename):
            nim = load(filename)
            if index == 0:
                first_volume = nim.get_data().squeeze()
                mean_volume = first_volume.copy().astype(np.float32)
                header = get_header(nim)
                affine = get_affine(nim)
            else:
                mean_volume += nim.get_data().squeeze()
        mean_volume /= float(len(list(input_filename)))
    del nim
    if np.isnan(mean_volume).any():
        tmp = mean_volume.copy()
        tmp[np.isnan(tmp)] = 0
        mean_volume = tmp

    mask = compute_mask(mean_volume, first_volume, m, M, cc,
                        opening=opening, exclude_zeros=exclude_zeros)

    if output_filename is not None:
        header['descrip'] = 'mask'
        output_image = nifti1.Nifti1Image(mask.astype(np.uint8),
                                            affine=affine,
                                            header=header)
        save(output_image, output_filename)
    if not return_mean:
        return mask
    else:
        return mask, mean_volume


def compute_mask(mean_volume, reference_volume=None, m=0.2, M=0.9,
                        cc=True, opening=2, exclude_zeros=False):
    """
    Compute a mask file from fMRI data in 3D or 4D ndarrays.

    Compute and write the mask of an image based on the grey level
    This is based on an heuristic proposed by T.Nichols:
    find the least dense point of the histogram, between fractions
    m and M of the total image histogram.

    In case of failure, it is usually advisable to increase m.

    Parameters
    ----------
    mean_volume : 3D ndarray
        mean EPI image, used to compute the threshold for the mask.
    reference_volume: 3D ndarray, optional
        reference volume used to compute the mask. If none is give, the
        mean volume is used.
    m : float, optional
        lower fraction of the histogram to be discarded.
    M: float, optional
        upper fraction of the histogram to be discarded.
    cc: boolean, optional
        if cc is True, only the largest connect component is kept.
    opening: int, optional
        if opening is larger than 0, an morphological opening is performed,
        to keep only large structures. This step is useful to remove parts of
        the skull that might have been included.
    exclude_zeros: boolean, optional
        Consider zeros as missing values for the computation of the
        threshold. This option is useful if the images have been
        resliced with a large padding of zeros.

    Returns
    -------
    mask : 3D boolean ndarray
        The brain mask
    """
    if reference_volume is None:
        reference_volume = mean_volume
    sorted_input = np.sort(mean_volume.reshape(-1))
    if exclude_zeros:
        sorted_input = sorted_input[sorted_input != 0]
    limiteinf = np.floor(m * len(sorted_input))
    limitesup = np.floor(M * len(sorted_input))

    delta = sorted_input[limiteinf + 1:limitesup + 1] \
            - sorted_input[limiteinf:limitesup]
    ia = delta.argmax()
    threshold = 0.5 * (sorted_input[ia + limiteinf]
                        + sorted_input[ia + limiteinf + 1])

    mask = (reference_volume >= threshold)

    if cc:
        mask = largest_cc(mask)

    if opening > 0:
        mask = ndimage.binary_opening(mask.astype(np.int),
                                        iterations=opening)
    return mask.astype(bool)


def compute_mask_sessions(session_images, m=0.2, M=0.9, cc=1, threshold=0.5,
                          exclude_zeros=False, return_mean=False, opening=2):
    """ Compute a common mask for several sessions of fMRI data.

        Uses the mask-finding algorithmes to extract masks for each
        session, and then keep only the main connected component of the
        a given fraction of the intersection of all the masks.


    Parameters
    ----------
    session_images : list of (list of strings) or nipy image objects
        A list of images/list of nifti filenames. Each inner list/image
        represents a session.
    m : float, optional
        lower fraction of the histogram to be discarded.
    M: float, optional
        upper fraction of the histogram to be discarded.
    cc: boolean, optional
        if cc is True, only the largest connect component is kept.
    threshold : float, optional
        the inter-session threshold: the fraction of the
        total number of session in for which a voxel must be in the
        mask to be kept in the common mask.
        threshold=1 corresponds to keeping the intersection of all
        masks, whereas threshold=0 is the union of all masks.
    exclude_zeros: boolean, optional
        Consider zeros as missing values for the computation of the
        threshold. This option is useful if the images have been
        resliced with a large padding of zeros.
    return_mean: boolean, optional
        if return_mean is True, the mean image accross subjects is
        returned.
    opening: int, optional,
             size of  the morphological opening

    Returns
    -------
    mask : 3D boolean ndarray
        The brain mask
    mean : 3D float array
        The mean image
    """
    mask, mean = None, None
    for index, session in enumerate(session_images):
        if hasattr(session, 'get_data'):
            mean = session.get_data()
            if mean.ndim > 3:
                mean = mean.mean(-1)
            this_mask = compute_mask(mean, None, m=m, M=M, cc=cc,
                        opening=opening, exclude_zeros=exclude_zeros)
            if return_mean:
                this_mask = this_mask, mean
        else:
            this_mask = compute_mask_files(
                session, m=m, M=M, cc=cc, exclude_zeros=exclude_zeros,
                return_mean=return_mean, opening=opening)
        if return_mean:
            this_mask, this_mean = this_mask
            if mean is None:
                mean = this_mean.astype(np.float)
            else:
                mean += this_mean
        this_mask = this_mask.astype(np.int8)
        if mask is None:
            mask = this_mask
        else:
            mask += this_mask
        # Free memory early
        del this_mask

    # Take the "half-intersection", i.e. all the voxels that fall within
    # 50% of the individual masks.
    mask = (mask > threshold * len(list(session_images)))

    if cc:
        # Select the largest connected component (each mask is
        # connect, but the half-interesection may not be):
        mask = largest_cc(mask)
    mask = mask.astype(np.bool)

    if return_mean:
        # Divide by the number of sessions
        mean /= len(session_images)
        return mask, mean

    return mask


def intersect_masks(input_masks, output_filename=None, threshold=0.5, cc=True):
    """
    Given a list of input mask images, generate the output image which
    is the the threshold-level intersection of the inputs

    Parameters
    ----------
    input_masks: list of strings or ndarrays
        paths of the input images nsubj set as len(input_mask_files), or
        individual masks.
    output_filename, string:
        Path of the output image, if None no file is saved.
    threshold: float within [0, 1[, optional
        gives the level of the intersection.
        threshold=1 corresponds to keeping the intersection of all
        masks, whereas threshold=0 is the union of all masks.
    cc: bool, optional
        If true, extract the main connected component

    Returns
    -------
    grp_mask, boolean array of shape the image shape
    """
    grp_mask = None
    if threshold > 1:
        raise ValueError('The threshold should be < 1')
    if threshold < 0:
        raise ValueError('The threshold should be > 0')
    threshold = min(threshold, 1 - 1.e-7)

    for this_mask in input_masks:
        if isinstance(this_mask, basestring):
            # We have a filename
            this_mask = load(this_mask).get_data()
        if grp_mask is None:
            grp_mask = this_mask.copy().astype(np.int)
        else:
            # If this_mask is floating point and grp_mask is integer, numpy 2
            # casting rules raise an error for in-place addition.
            # Hence we do it long-hand.
            # XXX should the masks be coerced to int before addition?
            grp_mask = grp_mask + this_mask

    grp_mask = grp_mask > (threshold * len(list(input_masks)))

    if np.any(grp_mask > 0) and cc:
        grp_mask = largest_cc(grp_mask)

    if output_filename is not None:
        if isinstance(input_masks[0], basestring):
            nim = load(input_masks[0])
            header = get_header(nim)
            affine = get_affine(nim)
        else:
            header = dict()
            affine = np.eye(4)
        header['descrip'] = 'mask image'
        output_image = nifti1.Nifti1Image(grp_mask.astype(np.uint8),
                                            affine=affine,
                                            header=header,
                                         )
        save(output_image, output_filename)

    return grp_mask > 0


###############################################################################
# Time series extraction
###############################################################################

def series_from_mask(filenames, mask, dtype=np.float32,
                     smooth=False, ensure_finite=True):
    """ Read the time series from the given sessions filenames, using the mask.

    Parameters
    -----------
    filenames: list of 3D nifti file names, or 4D nifti filename.
               Files are grouped by session.
    mask: 3d ndarray
          3D mask array: true where a voxel should be used.
    smooth: False or float, optional
            If smooth is not False, it gives the size, in voxel of the
            spatial smoothing to apply to the signal.
    ensure_finite: boolean, optional
            If ensure_finite is True, the non-finite values (NaNs and infs)
            found in the images will be replaced by zeros

    Returns
    --------
    session_series: ndarray
            3D array of time course: (session, voxel, time)
    header: header object
            The header of the first file.

    Notes
    -----
    When using smoothing, ensure_finite should be True: as elsewhere non
    finite values will spread accross the image.
    """
    assert len(filenames) != 0, (
        'filenames should be a file name or a list of file names, '
        '%s (type %s) was passed' % (filenames, type(filenames)))
    mask = mask.astype(np.bool)
    if smooth:
        # Convert from a sigma to a FWHM:
        smooth /= np.sqrt(8 * np.log(2))
    if isinstance(filenames, basestring):
        # We have a 4D nifti file
        data_file = load(filenames)
        header = get_header(data_file)
        series = data_file.get_data()
        if ensure_finite:
            # SPM tends to put NaNs in the data outside the brain
            series[np.logical_not(np.isfinite(series))] = 0
        series = series.astype(dtype)
        affine = get_affine(data_file)[:3, :3]
        del data_file
        if isinstance(series, np.memmap):
            series = np.asarray(series).copy()
        if smooth:
            vox_size = np.sqrt(np.sum(affine ** 2, axis=0))
            smooth_sigma = smooth / vox_size
            for this_volume in np.rollaxis(series, -1):
                this_volume[...] = ndimage.gaussian_filter(this_volume,
                                                        smooth_sigma)
        series = series[mask]
    else:
        nb_time_points = len(list(filenames))
        series = np.zeros((mask.sum(), nb_time_points), dtype=dtype)
        for index, filename in enumerate(filenames):
            data_file = load(filename)
            data = data_file.get_data()
            if ensure_finite:
                # SPM tends to put NaNs in the data outside the brain
                data[np.logical_not(np.isfinite(data))] = 0
            data = data.astype(dtype)
            if smooth is not False:
                affine = get_affine(data_file)[:3, :3]
                vox_size = np.sqrt(np.sum(affine ** 2, axis=0))
                smooth_sigma = smooth / vox_size
                data = ndimage.gaussian_filter(data, smooth_sigma)

            series[:, index] = data[mask]
            # Free memory early
            del data
            if index == 0:
                header = get_header(data_file)

    return series, header
