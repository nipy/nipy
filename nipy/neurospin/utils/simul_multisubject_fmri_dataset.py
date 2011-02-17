# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This module conatins a function to produce a dataset which simulates
a collection of 2D images This dataset is saved as a 3D image
(each slice being a subject) and a 3D array

example of use: surrogate_2d_dataset(nbsubj=1,fid="/tmp/toto.dat",verbose=1)

todo: rewrite it as a class

Author : Bertrand Thirion, 2008-2009
"""

import numpy as np
import scipy.ndimage as nd

# definition of the maxima at the group level
pos = np.array([[6 ,  7],
                [10, 10],
                [15, 10]])
ampli = np.array([3, 4, 4])

def _cone2d(shape, ij, pos, ampli, width):
    """
    Define a cone of the proposed grid
    """
    temp = np.zeros(shape)
    pos = np.reshape(pos,(1,2))
    dist = np.sqrt(np.sum((ij-pos)**2, axis=1))
    codi = (width-dist)*(dist < width)/width
    temp[ij[:,0],ij[:,1]] = codi*ampli
    return temp

def _cone3d(shape, ij, pos, ampli, width):
    """
    Define a cone of the proposed grid
    """
    temp = np.zeros(shape)
    pos = np.reshape(pos,(1,3))
    dist = np.sqrt(np.sum((ij-pos)**2, axis=1))
    codi = (width-dist)*(dist < width)/width
    temp[ij[:,0],ij[:,1],ij[:,2]] = codi*ampli
    return temp


def surrogate_2d_dataset(nbsubj=10, dimx=30, dimy=30, sk=1.0, 
                         noise_level=1.0, pos=pos, ampli=ampli,
                         spatial_jitter=1.0, signal_jitter=1.0,
                         width=5.0, width_jitter=0,
                         out_text_file=None, out_image_file=None, 
                         verbose=False, seed=False):
    """
    Create surrogate (simulated) 2D activation data with spatial noise.

    Parameters
    -----------
    nbsubj: integer, optionnal
        The number of subjects, ie the number of different maps
        generated.
    dimx: integer, optionnal
        The x size of the array returned.
    dimy: integer
        The y size of the array returned.
    sk: float, optionnal
        Amount of spatial noise smoothness.
    noise_level: float, optionnal
        Amplitude of the spatial noise.
        amplitude=noise_level)
    pos: 2D ndarray of integers, optionnal
        x, y positions of the various simulated activations.
    ampli: 1D ndarray of floats, optionnal
        Respective amplitude of each activation
    spatial_jitter: float, optionnal
        Random spatial jitter added to the position of each activation,
        in pixel.
    signal_jitter: float, optionnal
        Random amplitude fluctuation for each activation, added to the 
        amplitude specified by ampli
    width: float or ndarray, optionnal
        Width of the activations
    width_jitter: float
        Relative width jitter of the blobs
    out_text_file: string or None, optionnal
        If not None, the resulting array is saved as a text file with the
        given file name
    out_image_file: string or None, optionnal
        If not None, the resulting is saved as a nifti file with the
        given file name.
    verbose: boolean, optionnal
        If verbose is true, the data for the last subject is plotted as
        a 2D image.
    seed=False:  int, optionnal
        If seed is not False, the random number generator is initialized
        at a certain value

    Returns
    -------
    dataset: 3D ndarray
        The surrogate activation map, with dimensions (nbsubj, dimx, dimy)
    """
    if seed:
        nr = np.random.RandomState([seed])
    else:
        import numpy.random as nr
    
    shape = (dimx, dimy)
    ij = np.array(np.where(np.ones(shape))).T
    dataset = []

    for s in range(nbsubj):
        # make the signal
        data = np.zeros(shape)
        lpos = pos + spatial_jitter*nr.randn(1, 2)
        lampli = ampli + signal_jitter*nr.randn(np.size(ampli))
        this_width = width * (1 - width_jitter*nr.randn(np.size(ampli)))
        for k in range(np.size(lampli)):
            data = np.maximum(data,
                             _cone2d(shape, ij, lpos[k], lampli[k], 
                                     this_width[k]))
    
        # make some noise
        noise = nr.randn(dimx,dimy)

        # smooth the noise
        noise = nd.gaussian_filter(noise, sk)
        noise = np.reshape(noise, (-1, 1))

        noise *= noise_level/np.std(noise)

        #make the mixture
        data += np.reshape(noise, shape)

        dataset.append(data)

    if verbose:
        import matplotlib.pylab as mp
        mp.figure()
        mp.imshow(data, interpolation='nearest')
        mp.colorbar()

    dataset = np.array(dataset)

    if out_text_file is not None: 
        dataset.tofile(out_text_file)

    if out_image_file is not None:
        from nibabel import save, Nifti1Image 
        save(Nifti1Image( dataset, np.eye(4)), out_image_file)

    return dataset


def surrogate_3d_dataset(nbsubj=1, shape=(20,20,20), mask=None,
                            sk=1.0, noise_level=1.0, pos=None, ampli=None,
                            spatial_jitter=1.0, signal_jitter=1.0,
                            width=5.0, out_text_file=None, out_image_file=None, 
                            verbose=False, seed=False):
    """
    Create surrogate (simulated) 3D activation data with spatial noise.

    Parameters
    -----------
    nbsubj: integer, optionnal
        The number of subjects, ie the number of different maps
        generated.
    shape=(20,20,20): tuple of integers,
         the shape of each image
    mask=None: brifti image instance,
        referential- and mask- defining image (overrides shape)
    sk: float, optionnal
        Amount of spatial noise smoothness.
    noise_level: float, optionnal
        Amplitude of the spatial noise.
        amplitude=noise_level)
    pos: 2D ndarray of integers, optionnal
        x, y positions of the various simulated activations.
    ampli: 1D ndarray of floats, optionnal
        Respective amplitude of each activation
    spatial_jitter: float, optionnal
        Random spatial jitter added to the position of each activation,
        in pixel.
    signal_jitter: float, optionnal
        Random amplitude fluctuation for each activation, added to the 
        amplitude specified by ampli
    width: float or ndarray, optionnal
        Width of the activations
    out_text_file: string or None, optionnal
        If not None, the resulting array is saved as a text file with the
        given file name
    out_image_file: string or None, optionnal
        If not None, the resulting is saved as a nifti file with the
        given file name.
    verbose: boolean, optionnal
        If verbose is true, the data for the last subject is plotted as
        a 2D image.
    seed=False:  int, optionnal
        If seed is not False, the random number generator is initialized
        at a certain value

    Returns
    -------
    dataset: 3D ndarray
        The surrogate activation map, with dimensions (nbsubj, dimx, dimy, dimz)
    """
    if seed:
        nr = np.random.RandomState([seed])
    else:
        import numpy.random as nr

    if mask is not None:
        shape = mask.get_shape()
        mask_data = mask.get_data()
    else:
        mask_data = np.ones(shape)
    
    ijk = np.array(np.where(mask_data)).T
    dataset = []

    # make the signal
    for s in range(nbsubj):
        data = np.zeros(shape)
        if pos !=None:
            if len(pos)!=len(ampli):
                raise ValueError, 'ampli and pos do not have the same len'
            lpos = pos + spatial_jitter*nr.randn(1, 3)
            lampli = ampli + signal_jitter*nr.randn(np.size(ampli))
        for k in range(np.size(lampli)):
            data = np.maximum(data,_cone3d(shape, ijk, lpos[k], lampli[k],
                                           width))
    
        # make some noise
        noise = nr.randn(shape[0], shape[1], shape[2])
    
        # smooth the noise
        noise = nd.gaussian_filter(noise, sk)
        #noise = np.reshape(noise, (-1, 1))
        noise *= noise_level/np.std(noise)
        
        #make the mixture
        data += noise
    
        dataset.append(data)

    if verbose:
        import matplotlib.pylab as mp
        mp.figure()
        mp.imshow(data, interpolation='nearest')
        mp.colorbar()

    dataset = np.array(dataset)

    if out_text_file is not None: 
        dataset.tofile(out_text_file)

    if out_image_file is not None:
        from nibabel import save, Nifti1Image 
        save(Nifti1Image( dataset, np.eye(4)), out_image_file)

    return dataset

def surrogate_4d_dataset(shape=(20,20,20), mask=None, n_scans=1, n_sess=1,
                         dmtx=None, sk=1.0, noise_level=1.0,  signal_level=1.0,
                         out_image_file=None, verbose=False, seed=False):
    """
    Create surrogate (simulated) 3D activation data with spatial noise.

    Parameters
    -----------
    shape=(20,20,20): tuple of integers,
         the shape of each image
    mask=None: brifti image instance,
        referential- and mask- defining image (overrides shape)
    n_scans: int, optional,
        number of scans to be simlulated
        overrided by the design matrix
    n_sess: int, optional,
        the number of simulated sessions
    dmtx: array of shape(n_scans, n_rows),
        the design matrix
    sk: float, optionnal
        Amount of spatial noise smoothness.
    noise_level: float, optionnal
        Amplitude of the spatial noise.
        amplitude=noise_level)
    signal_level: float, optional,
        Amplitude of the signal
    out_image_file: string or None, optionnal
        If not None, the resulting is saved as a nifti file with the
        given file name.
    verbose: boolean, optionnal
        If verbose is true, the data for the last subject is plotted as
        a 2D image.
    seed=False:  int, optionnal
        If seed is not False, the random number generator is initialized
        at a certain value

    Returns
    -------
    dataset: a list of n_sess ndarray of shape 
             (shape[0], shape[1], shape[2], n_scans)
             The surrogate activation map
    """
    if seed:
        nr = np.random.RandomState([seed])
    else:
        import numpy.random as nr

    if mask is not None:
        shape = mask.get_shape()
        affine = mask.get_affine()
        mask_data = mask.get_data().astype('bool')
    else:
        affine = np.eye(4)
        mask_data = np.ones(shape).astype('bool')

    if dmtx is not None:
        n_scans = dmtx.shape[0]
    
    shape_4d = tuple((shape[0], shape[1], shape[2], n_scans))
    
    output_images = []
    if dmtx is not None:
        beta = []
        for r in range(dmtx.shape[1]):
            betar = nd.gaussian_filter(nr.randn(*shape), sk)
            betar /= np.std(betar)
            beta.append(signal_level*betar)
        beta = np.rollaxis(np.array(beta), 0, 4)
    
    for ns in range(n_sess):
        data = np.zeros(shape_4d)
        
        # make the signal
        if dmtx is not None:
            data[mask_data,:] += np.dot(beta[mask_data], dmtx.T) 
            #for r in range(dmtx.shape[1]):
            #    #data[mask_data,:] += np.outer(beta[mask_data], dmtx[:,r]) 
                

        for s in range(n_scans):
            # make some noise
            noise = nr.randn(*shape)
    
            # smooth the noise
            noise = nd.gaussian_filter(noise, sk)
            noise *= noise_level/np.std(noise)
        
            # make the mixture
            data[:,:,:,s] += noise
            data[:,:,:,s] += 100*mask_data
            
        from nibabel import save, Nifti1Image 
        wim = Nifti1Image( data, affine)
        output_images.append(wim)
        if out_image_file is not None:
            save(wim, out_image_file[s])

    return output_images
