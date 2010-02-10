import numpy as np 

from nipy.neurospin.image import transform_image, from_brifti, to_brifti

from iconic_registration import IconicRegistration
from affine import Affine 
from grid_transform import GridTransform

def register(source, 
             target, 
             similarity='cr',
             interp='pv',
             subsampling=None,
             normalize=None, 
             search='affine',
             graduate_search=False,
             optimizer='powell'):
    
    """
    Three-dimensional affine image registration. 
    
    Parameters
    ----------
    source : brifti-like image object 
       Source image 
    target : brifti-like image 
       Target image array
    similarity : str or callable
       Cost-function for assessing image similarity.  If a string, one
       of 'cc', 'cr', 'crl1', 'mi', je', 'ce', 'nmi', 'smi'.  'cr'
       (correlation ratio) is the default. If a callable, it should
       take a two-dimensional array representing the image joint
       histogram as an input and return a float. See
       ``registration_module.pyx``
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
       If a string, one of 'powell', 'simplex', 'conjugate_gradient'
       Alternatively, a sequence of such strings can be provided to
       run several optimizers sequentially. If bot `search` and
       `optimizer` are sequences, then the shorter is filled with its
       last value to match the longer. 

    Returns
    -------
    T : source-to-target affine transformation 
        Object that can be casted to a numpy array. 

    """
    regie = IconicRegistration(from_brifti(source), from_brifti(target))
    if subsampling == None: 
        regie.set_source_fov(fixed_npoints=64**3)
    else:
        regie.set_source_fov(spacing=subsampling)
    regie.similarity = similarity
    regie.interp = interp

    if isinstance(search, basestring): 
        search = [search]
    if isinstance(optimizer, basestring):
        optimizer = [optimizer]

    T = None
    for i in range(max(len(search), len(optimizer))):
        s = search[min(i, len(search)-1)]
        m = optimizer[min(i, len(optimizer)-1)]
        T = regie.optimize(method=m, search=s, start=T)
    return T


def transform(floating, T, reference=None, interp_order=3):

    # Convert assumed brifti-like input images to local image class
    floating = from_brifti(floating)
    if not reference == None: 
        reference = from_brifti(reference)

    # Switch on transformation type
    if isinstance(T, GridTransform): 
        if not T.shape == reference.shape: 
            raise ValueError('Wrong grid transformation shape')
        t = T()
        ttype = 'grid'
    else:
        t = np.asarray(T)
        ttype = 'affine'

    return to_brifti(transform_image(floating, t, ttype, grid_coords=False,
                                     reference=reference, interp_order=interp_order))



