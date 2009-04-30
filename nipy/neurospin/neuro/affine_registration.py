import time
import numpy as np 

import nipy.neurospin as fff2

def default_subsampling(dims):
    """  
    By default, subsampling factors are tuned so as to match
    approximately 64**3 voxels. 
    """
    NVOXELS_REF = 64**3 
    dims = np.asarray(dims)
    speedup_min = dims.prod()/NVOXELS_REF
    speedup = 1
    subsampling = np.ones(3)

    while speedup < speedup_min:
        # Subsample the direction with the highest number of samples
        ddims = dims/subsampling
        if ddims[0] >= ddims[1] and ddims[0] >= ddims[2]:
            dir = 0
        elif ddims[1] > ddims[0] and ddims[1] >= ddims[2]:
            dir = 1
        else:
            dir = 2
        subsampling[dir] += 1
        speedup = subsampling.prod()
            
    print('Using default subsampling %s' % subsampling)
    return subsampling


def affine_registration(source, target, 
                        similarity='correlation ratio', 
                        interp='partial volume', 
                        subsampling=None,
                        normalize=None, 
                        search='affine',
                        graduate_search=False,
                        optimizer='powell',
                        resample=True):

    if not isinstance(source, fff2.neuro.image) or not isinstance(target, fff2.neuro.image): 
        raise ValueError, 'Incorrect input image objects'

    if subsampling==None: 
        subsampling = default_subsampling(source.array.shape)

    matcher = fff2.registration.iconic(source, target)
    matcher.set(subsampling=subsampling, interp=interp, similarity=similarity, normalize=normalize)

    # Register
    print('Starting registration...')
    print('Similarity: %s' % matcher.similarity)
    print('Normalize: %s' % matcher.normalize) 
    print('Interpolation: %s' % matcher.interp)
    tic = time.time()

    t1 = t2 = None
    if graduate_search or search=='rigid':
        T, t1 = matcher.optimize(method=optimizer, search='rigid 3D')
    if graduate_search or search=='similarity':
        T, t2 = matcher.optimize(method=optimizer, search='similarity 3D', start=t1)
    if graduate_search or search=='affine':
        T, t3 = matcher.optimize(method=optimizer, search='affine 3D', start=t2)

    toc = time.time()
    print('  Registration time: %f sec' % (toc-tic))
    
    # Resample source image
    if resample: 
        print('Resampling source image...')
        tic = time.time()
        source_rsp = fff2.neuro.image(target)
        source_rsp.set_array(matcher.resample(T))
        toc = time.time()
        print('  Resampling time: %f sec' % (toc-tic))
        return T, source_rsp 
    else:
        return T

    
