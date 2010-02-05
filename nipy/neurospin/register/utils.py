import numpy as np

CLAMP_DTYPE = 'short'

def clamp(source, th=0, mask=None, bins=256):
    """ 
    Define a mask as the intersection of an initial mask and those
    indices for which array values are above a given threshold. Then,
    clamp in-mask array values in the range [0..bins-1] and reset
    out-of-mask values to -1.
    
    Parameters
    ----------
    source : ndarray
      The input array

    th : number
      Low threshold

    mask : ndarray
      Mask 

    bins : number 
      Desired number of bins
    
    Returns
    -------
    y : ndarray
      Clamped array

    bins : number 
      Adjusted number of bins 

    """
 
    # Mask 
    if mask == None: 
        mask = np.ones(source.shape, dtype='bool')
    mask *= (source>=th)

    # Create output array to allow in-place operations
    y = np.zeros(source.shape, dtype=CLAMP_DTYPE)
    dmaxmax = 2**(8*y.dtype.itemsize-1)-1

    # Threshold
    dmax = bins-1 ## default output maximum value
    if dmax > dmaxmax: 
        raise ValueError('Excess number of bins')
    xmin = float(source[mask].min())
    xmax = float(source[mask].max())
    th = np.maximum(th, xmin)
    if th > xmax:
        th = xmin  
        print("Warning: Inconsistent threshold %f, ignored." % th) 
    
    # Input array dynamic
    d = xmax-th

    """
    If the image dynamic is small, no need for compression: just
    downshift image values and re-estimate the dynamic range (hence
    xmax is translated to xmax-tth casted to the appropriate
    dtype. Otherwise, compress after downshifting image values (values
    equal to the threshold are reset to zero).
    """
    y[mask==False] = -1
    if np.issubdtype(source.dtype, int) and d<=dmax:
        y[mask] = source[mask]-th
        bins = int(d)+1
    else: 
        a = dmax/d
        y[mask] = np.round(a*(source[mask]-th))
 
    return y, bins 


def subsample(source, npoints):
    """ Tune subsampling factors so that the number of voxels >0 in the
    output block matches a given number.
    
    Parameters
    ----------
    source : ndarray or sequence  
      Source image to subsample
    npoints : int
      Target number of voxels (negative values will be ignored)

    Returns
    -------
    sub_source: ndarray 
      Subsampled source 
    subsampling: ndarray 
      Subsampling factors
    actual_npoints: int
      Actual size of the subsampled array 

    """
    dims = source.shape
    actual_npoints = (source >= 0).sum()
    subsampling = np.ones(3, dtype='uint')
    sub_source = source

    while actual_npoints > npoints:
        # Subsample the direction with the highest number of samples
        ddims = dims/subsampling
        if ddims[0] >= ddims[1] and ddims[0] >= ddims[2]:
            dir = 0
        elif ddims[1] > ddims[0] and ddims[1] >= ddims[2]:
            dir = 1
        else:
            dir = 2
        subsampling[dir] += 1
        sub_source = source[::subsampling[0], ::subsampling[1], ::subsampling[2]]
        actual_npoints = (sub_source >= 0).sum()
            
    return sub_source, subsampling, actual_npoints
