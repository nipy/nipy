import numpy as np

from neuroimaging.neurospin.register.iconic_matcher import IconicMatcher
from neuroimaging.neurospin.register.routines import cspline_resample

### FIXME LATER
from neuroimaging.neurospin import Image

def affine_register(source, 
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
    source : image object 
             Source image array 
    target : image object
             Target image array 
    
    Returns
    -------
    T : source-to-target affine transformation 
        Object that can be casted to a numpy array. 

    """
    
    matcher = IconicMatcher(source.get_data(), 
                            target.get_data(), 
                            source.get_affine(),
                            target.get_affine())
    if subsampling == None: 
        matcher.set_field_of_view(fixed_npoints=64**3)
    else:
        matcher.set_field_of_view(subsampling=subsampling)
    matcher.set_interpolation(method=interp)
    matcher.set_similarity(similarity=similarity, normalize=normalize)

    # Register
    print('Starting registration...')
    print('Similarity: %s' % matcher.similarity)
    print('Normalize: %s' % matcher.normalize) 
    print('Interpolation: %s' % matcher.interp)

    T = None
    if graduate_search or search=='rigid':
        T = matcher.optimize(method=optimizer, search='rigid')
    if graduate_search or search=='similarity':
        T = matcher.optimize(method=optimizer, search='similarity', start=T)
    if graduate_search or search=='affine':
        T = matcher.optimize(method=optimizer, search='affine', start=T)
    
    return T


def affine_resample(source, 
                    target, 
                    T, 
                    toresample='source', 
                    dtype=None, order=3, 
                    use_scipy=False): 
    """
    Image resampling using spline interpolation. 

    Parameters
    ----------
    source : image
    
    target : image

    T : source-to-target affine transform
    """
    Tv = np.dot(np.linalg.inv(target.get_affine()), np.dot(T, source.get_affine()))
    if use_scipy or not order==3: 
        use_scipy = True
        from scipy.ndimage import affine_transform 
    if toresample is 'target': 
        if not use_scipy:
            data = cspline_resample(target.get_data(), 
                                    source.get_shape(), 
                                    Tv, 
                                    dtype=dtype)
        else: 
            data = affine_transform(target.get_data(), 
                                  Tv[0:3,0:3], offset=Tv[0:3,3], 
                                  output_shape=source.get_shape(), 
                                  order=order)
        return Image(data, target.get_affine())
    else:
        if not use_scipy:
            data = cspline_resample(source.get_data(), 
                                    target.get_shape(), 
                                    np.linalg.inv(Tv), 
                                    dtype=dtype)
        else: 
            Tv = np.linalg.inv(Tv)
            data = affine_transform(source.get_data(), 
                                    Tv[0:3,0:3], offset=Tv[0:3,3], 
                                    output_shape=target.get_shape(), 
                                    order=order)
        return Image(data, source.get_affine())
