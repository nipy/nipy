import numpy as N

from neuroimaging.core.api import ImageList

from neuroimaging.core.image.iterators import SliceIterator
from neuroimaging.core.reference.grid import SamplingGrid
from neuroimaging.core.reference.coordinate_system import VoxelCoordinateSystem
from neuroimaging.core.reference.mapping import Affine

# this is unnecessary, i think
from neuroimaging.modalities.fmri.iterators import FmriParcelIterator, \
     FmriSliceParcelIterator

class FmriImage(ImageList):
    """
    TODO: change hte name of FmriImage -- maybe FmriImageList
    """

    def __init__(self, images=None, TR=None, slicetimes=None):
        """
        A lightweight implementation of an fMRI image as in ImageList
        
        Parameters
        ----------
        images: a sliceable object whose items are meant to be images,
                this is checked by asserting that each has a `grid` attribute
        TR:     time between frames in fMRI acquisition
        slicetimes: ndarray specifying offset for each slice of each frame

        >>> from numpy import asarray
        >>> from neuroimaging.testing funcfile
        >>> from neuroimaging.modalities.fmri.api import FmriImageList
        >>> from neuroimaging.modalities.fmri.api import load_fmri
        >>> from neuroimaging.modalities.core.api import load_image
        
        >>> # fmrilist and ilist represent the same data

        >>> fmrilist = load_fmri(funcfile)
        >>> funcim = load_image(funcfile)
        >>> ilist = FmriImageList(funcim)
        >>> print ilist[2:5]
        
        >>> print ilist[2]
        
        >>> print asarray(ilist).shape
        >>> print asarray(ilist[4]).shape

        See Also
        --------
        neuroimaging.core.image_list.ImageList

        """

        ImageList.__init__(self, images=images)
        self.TR = TR
        self.slicetimes = slicetimes

    def __getitem__(self, index):
        FmriImage(images=self.list[index], TR=self.TR,
                  slicetimes=self.slicetimes)

def parcel_iterator(img, parcelmap, parcelseq=None, mode='r'):
    """
    Parameters

    ----------
    parcelmap : ``[int]``
        This is an int array of the same shape as self.
        The different values of the array define different regions.
        For example, all the 0s define a region, all the 1s define
        another region, etc.           
    parcelseq : ``[int]`` or ``[(int, int, ...)]``
        This is an array of integers or tuples of integers, which
        define the order to iterate over the regions. Each element of
        the array can consist of one or more different integers. The
        union of the regions defined in parcelmap by these values is
        the region taken at each iteration. If parcelseq is None then
        the iterator will go through one region for each number in
        parcelmap.
    mode : ``string``
        The mode to run the iterator in.
        'r' - read-only (default)
        'w' - read-write                
    
    """
    return FmriParcelIterator(img, parcelmap, parcelseq, mode=mode)

def slice_parcel_iterator(img, parcelmap, parcelseq=None, mode='r'):
    """
    Parameters
    ----------
    parcelmap : ``[int]``
        This is an int array of the same shape as self.
        The different values of the array define different regions.
        For example, all the 0s define a region, all the 1s define
        another region, etc.           
    parcelseq : ``[int]`` or ``[(int, int, ...)]``
        This is an array of integers or tuples of integers, which
        define the order to iterate over the regions. Each element of
        the array can consist of one or more different integers. The
        union of the regions defined in parcelmap by these values is
        the region taken at each iteration. If parcelseq is None then
        the iterator will go through one region for each number in
        parcelmap.                
    mode : ``string``
        The mode to run the iterator in.
        'r' - read-only (default)
        'w' - read-write
    
    """
    
    return FmriSliceParcelIterator(img, parcelmap, parcelseq, mode=mode)

def fromimage(fourdimage, TR=None, slicetimes=None):
    """Create an FmriImage from a 4D Image.

    Load an image from the file specified by ``url`` and ``datasource``.

    Note this assumes that the 4d Affine mapping is such that it
    can be made into a list of 3d Affine mappings

    Parameters
    ----------
    fourdimage: a 4D Image 
    TR:     time between frames in fMRI acquisition
    slicetimes: ndarray specifying offset for each slice of each frame


    """
    images = []
    if not isinstance(fourdimage.grid.mapping, Affine):
        raise ValueError, 'fourdimage must have an Affine mapping'
    
    for im in [fourdimage[i] for i in fourdimage.shape]:
        g = im.grid
        ia = g.grid.input_coords.axes()[1:]
        ic = VoxelCoordinateSystem("voxel", ia)
        t = im.grid.mapping.transform[1:]
        a = Affine(t)
        newg = SamplingGrid(a, ic, g.output_coords)
        images.append(N.asarray(im), newg)

    return FmriImage(images=images, TR=TR, slicetimes=slicetimes)
