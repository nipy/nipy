# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Conversion mechansims for IO and interaction between volumetric datasets 
and other type of neuroimaging data.
"""
import os

import numpy as np

from nipy.io import imageformats
from nipy.io.imageformats.spatialimages import SpatialImage

from .volumes.volume_img import VolumeImg

def as_volume_img(obj, copy=True, squeeze=True, world_space=None):
    """ Convert the input to a VolumeImg.

        Parameters
        ----------
        obj : filename, pynifti or brifti object, or volume dataset.
            Input object, in any form that can be converted to a
            VolumeImg. This includes Nifti filenames, pynifti or brifti
            objects, or other volumetric dataset objects.
        copy: boolean, optional
            If copy is True, the data and affine arrays are copied, 
            elsewhere a view is taken.
        squeeze: boolean, optional
            If squeeze is True, the data array is squeeze on for
            dimensions above 3.
        world_space: string or None, optional
            An optional specification of the world space, to override
            that given by the image.

        Returns
        -------
        volume_img: VolumeImg object
            A VolumeImg object containing the data. The metadata is
            kept as much as possible in the metadata attribute.

        Notes
        ------
        The world space might not be correctly defined by the input
        object (in particular, when loading data from disk). In this
        case, you can correct it manually using the world_space keyword
        argument.

        For pynifti objects, the data is transposed.
    """
    if hasattr(obj, 'as_volume_img'):
        obj = obj.as_volume_img()
        if copy:
            obj = obj.__copy__()
        return obj

    elif isinstance(obj, basestring):
        if not os.path.exists(obj):
            raise ValueError("The file '%s' cannot be found" % obj)
        obj = imageformats.load(obj)
        copy = False
    
    if isinstance(obj, SpatialImage):
        data   = obj.get_data()
        affine = obj.get_affine()
        header = dict(obj.get_header())
        if obj._files:
            header['filename'] = obj._files['image']
    elif hasattr(obj, 'data') and hasattr(obj, 'sform') and \
                                            hasattr(obj, 'getVolumeExtent'):
        # Duck-types to a pynifti object
        data     = obj.data.T
        affine   = obj.sform
        header   = obj.header
        filename = obj.getFilename()
        if filename != '':
            header['filename'] = filename

    if world_space is not None:
        " Ugly if statement to use elif after "
    elif header.get('sform_code', 0) == 4:
        world_space = 'mni152'
    elif 'filename' in header:
        world_space = header['filename']
    else:
        # Cheap way to construct an object-specific hash
        world_space = str(id(obj))

    data    = np.asanyarray(data)
    affine  = np.asanyarray(affine)
    if copy:
        data    = data.copy()
        affine  = affine.copy()

    if squeeze:
        # Squeeze the dimensions above 3
        shape = [val for index, val in enumerate(data.shape)
                     if val !=1 or index < 3]
        data = np.reshape(data, shape)
    
    return VolumeImg(data, affine, world_space, metadata=header)


def save(filename, obj):
    """ Save an nipy image object to a file.
    """
    obj = as_volume_img(obj, copy=False)
    imageformats.save(imageformats.Nifti1Image(obj.get_data(), 
                      obj.affine), filename)

