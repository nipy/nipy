# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from affine import Affine, Rigid, Similarity
from grid_transform import GridTransform
from iconic_registration import IconicRegistration
from spacetime_registration import FmriRealign4d 

import numpy as np 


transform_classes = {'affine': Affine, 'rigid': Rigid, 'similarity': Similarity}
                     
def register(source, 
             target, 
             similarity='cr',
             interp='pv',
             subsampling=None,
             search='affine',
             graduate_search=False,
             optimizer='powell'):
    
    """
    Three-dimensional affine image registration. 
    
    Parameters
    ----------
    source : nibabel-like image object 
       Source image 
    target : nibabel-like image 
       Target image array
    similarity : str or callable
       Cost-function for assessing image similarity.  If a string, one
       of 'cc', 'cr', 'crl1', 'mi', je', 'ce', 'nmi', 'smi'.  'cr'
       (correlation ratio) is the default. If a callable, it should
       take a two-dimensional array representing the image joint
       histogram as an input and return a float. See
       ``_registration.pyx``
    interp : str
       Interpolation method.  One of 'pv': Partial volume, 'tri':
       Trilinear, 'rand': Random interpolation.  See
       ``iconic.c``
    subsampling : None or sequence length 3
       subsampling of image in voxels, where None (default) results 
       in the subsampling to be automatically adjusted to roughly match
       a cubic grid of 64**3 voxels
    search : str or sequence 
       If a string, one of 'affine', 'rigid', 'similarity'; default 'affine'
       A sequence of strings can be provided to run a graduate search, e.g.
       by doing first 'rigid', then 'similarity', then 'affine'
    optimizer : str or sequence 
       If a string, one of 'simplex', 'powell', 'steepest', 'cg', 'bfgs'
       Alternatively, a sequence of such strings can be provided to
       run several optimizers sequentially. If bot `search` and
       `optimizer` are sequences, then the shorter is filled with its
       last value to match the longer. 

    Returns
    -------
    T : source-to-target affine transformation 
        Object that can be casted to a numpy array. 

    """
    R = IconicRegistration(source, target)
    if subsampling == None: 
        R.set_source_fov(fixed_npoints=64**3)
    else:
        R.set_source_fov(spacing=subsampling)
    R.similarity = similarity
    R.interp = interp

    if isinstance(search, basestring): 
        search = [search]
    if isinstance(optimizer, basestring):
        optimizer = [optimizer]
   
    T = None
    for i in range(max(len(search), len(optimizer))):
        search_ = search[min(i, len(search)-1)]
        optimizer_ = optimizer[min(i, len(optimizer)-1)]
        if T == None: 
            T = transform_classes[search_]()
        else: 
            T = transform_classes[search_](T.vec12)
        T = R.optimize(T, method=optimizer_)
    return T


"""
def transform(floating, T, reference=None, interp_order=3):

    # Convert assumed nibabel-like input images to local image class
    floating = Image(floating)
    if not reference == None: 
        reference = Image(reference)

    return asNifti1Image(floating.transform(np.asarray(T), grid_coords=False,
                                            reference=reference, interp_order=interp_order))

"""

