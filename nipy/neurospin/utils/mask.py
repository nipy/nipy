
# Standard libraries imports
import warnings

# Major scientific libraries imports
import numpy as np

# Neuroimaging libraries imports
from nipy.io.imageformats import load, nifti1, save

import nipy.neurospin.graph as fg


def _largest_cc(mask):
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
    xyz = np.array(np.where(mask))
    nbvox = mask.sum()
    g = fg.WeightedGraph(nbvox)
    g.from_3d_grid(xyz.T)
    u = g.main_cc()
    xyz = xyz[:,u]
    
    mask_cc = np.zeros(mask.shape, np.int8)
    mask_cc[tuple(xyz)] = 1
    return mask_cc


def compute_mask_files(input_filename, output_filename=None, 
                        return_mean=False, m=0.2, M=0.9, cc=1):
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

    Returns
    -------
    mask : 3D boolean array 
        The brain mask
    mean_image : 3d ndarray, optional
        The main of all the images used to estimate the mask. Only
        provided if `return_mean` is True.

    """
    if hasattr(input_filename, '__iter__'):
        if len(input_filename) == 0:
            raise ValueError('input_filename should be a non-empty '
                'list of file names')
        # We have several images, we do mean on the fly, 
        # to avoid loading all the data in the memory
        for index, filename in enumerate(input_filename):
            nim = load(filename)
            if index == 0:
                first_volume = nim.get_raw_data().squeeze()
                mean_volume = first_volume.copy().astype(np.float32)
                header = nim.get_header()
                affine = nim.get_affine()
            else:
                mean_volume += nim.get_raw_data().squeeze()
        mean_volume /= float(len(input_filename))
    else: 
        # one single filename
        nim = load(input_filename)
        header = nim.get_header()
        affine = nim.get_affine()
        data = nim.get_raw_data()
        # Make a copy, to avoid holding a reference on the full array,
        # and thus polluting the memory.
        first_volume = data[:,:,:,0].copy()
        mean_volume = data.mean(axis=3)
        del data
    del nim

    mask = compute_mask(mean_volume, first_volume, m, M, cc)
      
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
                                                cc=1):
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

    Returns
    -------
    mask : 3D boolean ndarray 
        The brain mask
    """
    if reference_volume is None:
        reference_volume = mean_volume
    inputVector = np.sort(mean_volume.reshape(-1))
    limiteinf = np.floor(m * len(inputVector))
    limitesup = np.floor(M * len(inputVector))#inputVector.argmax())

    delta = inputVector[limiteinf + 1:limitesup + 1] \
            - inputVector[limiteinf:limitesup]
    ia = delta.argmax()
    threshold = 0.5 * (inputVector[ia + limiteinf] 
                        + inputVector[ia + limiteinf  +1])
    
    mask = (reference_volume >= threshold)

    if cc:
        try:
            mask = _largest_cc(mask)
        except TypeError:
            """ The grid is probably too large, will just pass. """
            warnings.warn('Mask too large, cannot extract largest cc.')
    return mask.astype(bool)


def compute_mask_sessions(session_files, m=0.2, M=0.9, cc=1, threshold=0.5):
    """ Compute a common mask for several sessions of fMRI data.

        Uses the mask-finding algorithmes to extract masks for each
        session, and then keep only the main connected component of the
        a given fraction of the intersection of all the masks.

 
    Parameters
    ----------
    session_files : list of list of strings
        A list of list of nifti filenames. Each inner list
        represents a session.
    threshold : float, optional
        the inter-session threshold: the fraction of the
        total number of session in for which a voxel must be in the
        mask to be kept in the common mask.
        threshold=1 corresponds to keeping the intersection of all
        masks, whereas threshold=0 is the union of all masks.
    m : float, optional
        lower fraction of the histogram to be discarded.
    M: float, optional
        upper fraction of the histogram to be discarded.
    cc: boolean, optional
        if cc is True, only the largest connect component is kept.

    Returns
    -------
    mask : 3D boolean ndarray 
        The brain mask
    """
    mask = None
    for session in session_files:
        this_mask = compute_mask_files(session,
                                       m=m, M=M,
                                       cc=cc).astype(np.int8)
        if mask is None:
            mask = this_mask
        else:
            mask += this_mask
        # Free memory early
        del this_mask
        
    # Take the "half-intersection", i.e. all the voxels that fall within
    # 50% of the individual masks.
    mask = (mask > threshold*len(session_files))
   
    if cc:
        # Select the largest connected component (each mask is
        # connect, but the half-interesection may not be):
        try:
            mask = _largest_cc(mask)
        except TypeError:
            """ The grid is probably too large, will just pass. """
            warnings.warn('Mask too large, cannot extract largest cc.')

    # We need to convert to boolean, as the graph structure casts
    # in int8
    return mask.astype(np.bool)

def intersect_masks(input_mask_files, output_filename=None, 
                                        threshold=0.5, cc=True):
    """
    Given a list of input mask images, generate the output image which
    is the the threshold-level intersection of the inputs 

    
    Parameters
    ----------
    input_mask_files: list of strings or ndarrays
        paths of the input images nsubj set as len(input_mask_files), or
        individual masks.
    output_filename, string:
        Path of the output image, if None no file is saved.
    threshold: float within [0, 1], optional
        gives the level of the intersection.
    cc: bool, optional
        If true, extract the main connected component
        
    Returns
    -------
    gmask, boolean array of shape the image shape
    """  
    gmask = None 

    for filename in input_mask_files:
        nim = load(filename)
        if gmask is None:
            gmask = nim.get_data().copy() # !!!
        else:
            gmask += nim.get_data()  
    
    gmask = gmask>(threshold*len(input_mask_files))
    if np.any(gmask>0) and cc:
        gmask = _largest_cc(gmask)
    
    if output_filename is not None:
        header = nim.get_header()
        header['descrip'] = 'mask image'
        output_image = nifti1.Nifti1Image(gmask.astype(np.uint8),
                                            affine=nim.get_affine(),
                                            header=header,
                                         )
        output_image.save(output_filename)
    return gmask>0

################################################################################
# Legacy function calls.
################################################################################

def computeMaskIntra(inputFilename, outputFilename, m=0.2, M=0.9, cc=1):
    """ Depreciated, see compute_mask_intra.
    """
    print "here we are"
    return compute_mask_intra(inputFilename, outputFilename, 
                                    m=m, M=M, cc=cc)


def computeMaskIntraArray(volumeMean, firstVolume, m=0.2, M=0.9,cc=1):
    """ Depreciated, see compute_mask_intra.
    """
    warnings.warn(
            'Depreciated function name, please use compute_mask_intra_array',
            stacklevel=2)
    return compute_mask_intra_array(volumeMean, firstVolume, 
                                    m=m, M=M, cc=cc)


def compute_mask_intra(input_filename, output_filename=None, return_mean=False, 
                            m=0.2, M=0.9, cc=1):
    """
    See compute_mask_files.
    """
    warnings.warn('compute_mask_intra is depreciated, please use' 
                  ' compute_mask_files',
                  stacklevel=2)
    return compute_mask_files(input_filename=input_filename, 
                              output_filename=output_filename, 
                              return_mean=return_mean,
                              m=m, M=M, cc=cc)


def compute_mask_intra_array(volume_mean, reference_volume=None, m=0.2, M=0.9, 
                                                cc=True):
    """
    Depreciated, see compute_mask.
    """
    warnings.warn('Depreciated function name, please use compute_mask',
                        stacklevel=2)
    return compute_mask(volume_mean, 
                        reference_volume=reference_volume, m=m, M=M, cc=cc)


