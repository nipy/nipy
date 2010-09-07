# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from affine import Affine, Rigid, Similarity, apply_affine
from grid_transform import GridTransform
from iconic_registration import IconicRegistration
from spacetime_registration import FmriRealign4d 
from _cubic_spline import cspline_transform, cspline_sample3d, cspline_resample3d

from nipy.core.image.affine_image import AffineImage

import numpy as np 
from scipy.ndimage import affine_transform, map_coordinates

_INTERP_ORDER = 3
                   
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
    if not subsampling == None: 
        R.focus(spacing=subsampling)
    R.similarity = similarity
    R.interp = interp

    if isinstance(search, basestring): 
        search = [search]
    if isinstance(optimizer, basestring):
        optimizer = [optimizer]
   
    transforms = {'affine': Affine, 'rigid': Rigid, 'similarity': Similarity}
    T = None
    for i in range(max(len(search), len(optimizer))):
        search_ = search[min(i, len(search)-1)]
        optimizer_ = optimizer[min(i, len(optimizer)-1)]
        if T == None: 
            T = transforms[search_]()
        else: 
            T = transforms[search_](T.vec12)
        T = R.optimize(T, method=optimizer_)
    return T



def resample(moving, transform, grid_coords=False, reference=None, 
             dtype=None, interp_order=_INTERP_ORDER):
    """
    Apply a transformation to the image considered as 'moving' to
    bring it into the same grid as a given 'reference' image. For
    technical reasons, the transformation is assumed to go from the
    'reference' to the 'moving'.

    This function uses scipy.ndimage except for the case
    `interp_order==3`, where a fast cubic spline implementation is
    used.
    
    Parameters
    ----------
    moving: nipy-like image
      Image to be resampled. 

    transform: nd array
      either a 4x4 matrix describing an affine transformation
      or a 3xN array describing voxelwise displacements of the
      reference grid points
    
    grid_coords : boolean
      True if the transform maps to grid coordinates, False if it maps
      to world coordinates
    
    reference: nipy-like image 
      Reference image, defaults to input. 
      
    interp_order: number 
      spline interpolation order, defaults to 3. 
    """
    if reference == None: 
        reference = moving
    shape = reference.shape
    data = moving.get_data()
    if dtype == None: 
        dtype = data.dtype
    t = np.asarray(transform)
    inv_affine = np.linalg.inv(moving.affine)

    # Case: affine transform
    if t.shape[-1] == 4: 
        if not grid_coords:
            t = np.dot(inv_affine, np.dot(t, reference.affine))
        if interp_order == 3: 
            output = cspline_resample3d(data, shape, t, dtype=dtype)
            output = output.astype(dtype)
        else: 
            output = np.zeros(shape, dtype=dtype)
            affine_transform(data, t[0:3,0:3], offset=t[0:3,3],
                             order=interp_order, cval=0, 
                             output_shape=shape, output=output)
    
    # Case: precomputed displacements
    else:
        if not grid_coords:
            t = apply_affine(inv_affine, t)
        coords = np.rollaxis(t, 3, 0)
        if interp_order == 3: 
            cbspline = cspline_transform(data)
            output = np.zeros(shape, dtype='double')
            output = cspline_sample3d(output, cbspline, *coords)
            output = output.astype(dtype)
        else: 
            output = map_coordinates(data, coords, order=interp_order, 
                                     cval=0, output=dtype)
    
    return AffineImage(output, reference.affine, 'ijk')


