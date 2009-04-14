
# Standard libraries imports
import warnings

# Major scientific libraries imports
from numpy import array, sort, floor, where, shape, sum, transpose, \
            zeros, int8, float32, uint8, bool

# Neuroimaging libraries imports
from nifti import NiftiImage
# In different versions of pynifti, this symbol lived in different places
try:
    from nifti.nifticlib import NIFTI_INTENT_LABEL
except ImportError:
    from nifti.clib import NIFTI_INTENT_LABEL


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
    xyz = array(where(mask))
    nbvox = sum(mask)
    g = fg.WeightedGraph(nbvox)
    g.from_3d_grid(transpose(xyz))
    u = g.main_cc()
    xyz = xyz[:,u]
    
    mask_cc = zeros(shape(mask), int8)
    mask_cc[tuple(xyz)] = 1
    return mask_cc


def compute_mask_intra(input_filename, output_filename=None, return_mean=False, 
                            copy_filename=None, m=0.2, M=0.9, cc=1):
    """
    See compute_mask_files.
    """
    return compute_mask_files(input_filename=input_filename, 
                              output_filename=output_filename, 
                            return_mean=return_mean,
                            copy_filename=copy_filename, m=m, 
                            M=M, cc=cc)


def compute_mask_files(input_filename, output_filename=None, return_mean=False, 
                            copy_filename=None, m=0.2, M=0.9, cc=1):
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
    copy_filename : string, optional
        optionally, a copy of the original data saved as a single-file 4D 
        nifti volume.
    m : float, optional
        lower fraction of the histogram to be discarded.
    M: float, optional
        upper fraction of the histogram to be discarded.
    cc: boolean, optional
        if cc is True, only the largest connect component is kept.

    Returns
    -------
    mask : nifti.NiftiImage object
        The brain mask
    mean_image : 3d ndarray, optional
        The main of all the images used to estimate the mask. Only
        provided if `return_mean` is True.

    """
    if hasattr(input_filename, '__iter__'):
        imgliste = [NiftiImage(x) for x in input_filename]
        volume = array([x.data.squeeze() for x in imgliste])
        volume = volume.squeeze()
    else: # one single filename
        imgliste = [NiftiImage(input_filename)]
        volume = imgliste[0].data

    volumeMean = volume.mean(0)
    firstVolume = volume[0]
    if copy_filename:
        # optionnaly write the volume as a 4D image
        NiftiImage(volume, imgliste[0].header).save(copy_filename)
    del volume
    
    dat = compute_mask_intra_array(volumeMean, firstVolume, m, M, cc)
    
    # header is auto-reupdated (number of dim, calmax.)
    outputImage = NiftiImage(dat.astype(uint8), imgliste[0].header) 
    # cosmetic updates
    outputImage.updateHeader({'intent_code': NIFTI_INTENT_LABEL, 
                              'intent_name': 'Intra Mask'})
    #outputImage.setPixDims(outputImage.voxdim + (0,))
    if output_filename is not None:
        outputImage.save(output_filename)
    if not return_mean:
        return outputImage
    else:
        return outputImage, volumeMean


def compute_mask_intra_array(volume_mean, reference_volume=None, m=0.2, M=0.9, 
                                                cc=True):
    """
    Depreciated, see compute_mask.
    """
    return compute_mask(volume_mean, 
            reference_volume=reference_volume, m=m, M=M, cc=cc)


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
    inputVector = sort(mean_volume.reshape(-1))
    limiteinf = floor(m * len(inputVector))
    limitesup = floor(M * len(inputVector))#inputVector.argmax())

    delta = inputVector[limiteinf + 1:limitesup + 1] \
            - inputVector[limiteinf:limitesup]
    ia = delta.argmax()
    threshold = 0.5 * (inputVector[ia + limiteinf] 
                        + inputVector[ia + limiteinf  +1])
    #print limitesup,limiteinf,reference_volume.max(),threshold
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
        # First compute the mean of the session
        session_mean = NiftiImage(session[0]).asarray().T.astype(float32)
        first_image = session_mean.copy()
        for filename in session[1:]:
            session_mean += NiftiImage(filename).asarray().T.astype(float32)
        session_mean /= float(len(session))

        this_mask = compute_mask_intra_array(session_mean, first_image, 
                                                m=m, M=M,
                                                cc=cc).astype(int8)
        if mask is None:
            mask = this_mask
        else:
            mask += this_mask
        # Free memory early
        del this_mask, first_image
        
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

    return mask


################################################################################
# Legacy function calls.
################################################################################

def computeMaskIntra(inputFilename, outputFilename, copyFilename=None, m=0.2, 
                        M=0.9,cc=1):
    """ Depreciated, see compute_mask_intra.
    """
    warnings.warn('Depreciated function name, please use compute_mask_intra',
                        stacklevel=2)
    print "here we are"
    return compute_mask_intra(inputFilename, outputFilename, 
                                    copy_filename=copyFilename,
                                    m=m, M=M, cc=cc)


def computeMaskIntraArray(volumeMean, firstVolume, m=0.2, M=0.9,cc=1):
    """ Depreciated, see compute_mask_intra.
    """
    warnings.warn(
            'Depreciated function name, please use compute_mask_intra_array',
            stacklevel=2)
    return compute_mask_intra_array(volumeMean, firstVolume, 
                                    m=m, M=M, cc=cc)


