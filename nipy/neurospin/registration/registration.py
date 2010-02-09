from nipy.neurospin.image import transform_image, from_brifti, to_brifti

from iconic_registration import IconicRegistration 


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
    optimizer : str
       One of 'powell', 'simplex', 'conjugate_gradient'
       
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

    T = None
    for s in search: 
        T = regie.optimize(method=optimizer, search=s, start=T)
    return T 


def transform(floating, T, reference=None, interp_order=3):
    floating = from_brifti(floating)
    if not reference == None: 
        reference = from_brifti(reference)
    return to_brifti(transform_image(floating, T, grid_coords=False,
                                     reference=reference, interp_order=interp_order))



