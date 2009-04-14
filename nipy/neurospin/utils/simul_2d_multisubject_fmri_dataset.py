"""
This module conatins a function to produce a dataset which simulates
a collection of 2D images This dataset is saved as a 3D nifti image
(each slice being a subject) and a 3D array

example of use: make_surrogate_array(nbsubj=1,fid="/tmp/toto.dat",verbose=1)

todo: rewrite it as a class

Author : Bertrand Thirion, 2008-2009
"""

import numpy as np
from numpy.random import randn
import scipy.ndimage as nd


# definition of the maxima at the group level
pos = np.array([[6 ,  7],
                [10, 10],
                [15, 10]])
ampli = np.array([3, 4, 4])

def cone(shape, ij, pos, ampli, width):
    """
    Define a cone of the proposed grid
    """
    temp = np.zeros(shape)
    pos = np.reshape(pos,(1,2))
    dist = np.sqrt(np.sum((ij-pos)**2, axis=1))
    codi = (width-dist)*(dist < width)/width
    temp[ij[:,0],ij[:,1]] = codi*ampli
    return temp


def make_surrogate_array(nbsubj=10, dimx=30, dimy=30, sk=1.0, 
                         noise_level=1.0, pos=pos, ampli=ampli,
                         spatial_jitter=1.0, signal_jitter=1.0,
                         width=5.0, out_text_file=None, out_niftifile=None, 
                         verbose=False):
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
    out_text_file: string or None, optionnal
        If not None, the resulting array is saved as a text file with the
        given file name
    out_niftifile: string or None, optionnal
        If not None, the resulting is saved as a nifti file with the
        given file name.
    verbose: boolean, optionnal
        If verbose is true, the data for the last subject is plotted as
        a 2D image.

    Returns
    -------
    dataset: 3D ndarray
        The surrogate activation map, with dimensions (nbsubj, dimx, dimy)
    """
    shape = (dimx, dimy)
    ij = np.transpose(np.where(np.ones(shape)))
    dataset = []

    for s in range(nbsubj):
        # make the signal
        data = np.zeros(shape)
        lpos = pos + spatial_jitter*randn(1, 2)
        lampli = ampli + signal_jitter*randn(np.size(ampli))
        for k in range(np.size(lampli)):
            data = np.maximum(data,
                                cone(shape, ij, lpos[k], lampli[k], width))
    
        # make some noise
        noise = randn(dimx,dimy)

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

    if out_niftifile is not None:
        import nifti
        nifti.NiftiImage(dataset).save(out_niftifile)

    return dataset

