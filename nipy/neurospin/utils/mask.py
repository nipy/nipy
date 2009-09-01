
# Standard libraries imports
import warnings

# Major scientific libraries imports
import numpy as np

# Neuroimaging libraries imports
import nifti
# In different versions of pynifti, this symbol lived in different places
try:
    from nifti.nifticlib import NIFTI_INTENT_LABEL
except ImportError:
    from nifti.clib import NIFTI_INTENT_LABEL


import nipy.neurospin.graph as fg

def load_nifti(filename):
    """ Load a nifti file, using memapping if possible.

       
    """
    try:
        nim = nifti.niftiimage.MemMappedNiftiImage(filename)
    except RuntimeError:
        "Memmapping is possible only for uncompressed files."
        nim = nifti.NiftiImage(filename)
    return nim


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


def compute_mask_files( input_filename, output_filename=None, 
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
    mask : nifti.NiftiImage object
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
            nim = load_nifti(filename)
            if index == 0:
                first_volume = nim.data.squeeze()
                mean_volume = first_volume.copy().astype(np.float32)
                header = nim.header
            else:
                mean_volume += nim.data.squeeze()
        mean_volume /= float(len(input_filename))
    else: 
        # one single filename
        nim = load_nifti(input_filename)
        header = nim.header
        first_volume = nim.data[0]
        mean_volume = nim.data.mean(axis=0)
    del nim

    dat = compute_mask(mean_volume, first_volume, m, M, cc)
    
    # header is auto-reupdated (number of dim, calmax.)
    output_image = nifti.NiftiImage(dat.astype(np.uint8), header) 
    # cosmetic updates
    output_image.updateHeader({'intent_code': NIFTI_INTENT_LABEL, 
                              'intent_name': 'Intra Mask'})
    #output_image.setPixDims(output_image.voxdim + (0,))
    if output_filename is not None:
        output_image.save(output_filename)
    if not return_mean:
        return output_image
    else:
        return output_image, mean_volume


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
                                       cc=cc).data.astype(np.int8)
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

    return mask

def intersect_masks(input_masks, output_filename, threshold, cc):
    """
    Given a list of input mask images, generate the output image which
    is the the threshold-level intersection of the inputs 

    
    Parameters
    ----------
    input_masks, list of strings, paths of the input images
                 nsubj set as len(input_masks)
    output_filename, string, path of the output image
    threshold: float, level of the intersection 
               must be within [0, 1]
    cc, bool additionally extract the main connected component
    """  
    nsubj = len(input_masks)
    
    nim = nifti.NiftiImage(inputs_masks[0])
    ref_dim = nim.getVolumeExtent()
    gmask = np.zeros(ref_dim)

    for s in range(nsubj):
        nim = nifti.NiftiImage(inputs_masks[s])
        gmask += nim.asarray()  
    
    gmask = gmask>(threshold*nsubj)     
    if (np.sum(gmask>0) & cc):
           gmask = _largest_cc(gmask)
    
    output_image = NiftiImage(gmask.astype(np.uint8), nim.header)
    output_image.description = 'mask image'
    output_image.save(output_filename)

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


