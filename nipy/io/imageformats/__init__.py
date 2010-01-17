#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""This module provides Python bindings to the NIfTI data format.

The NiBabel module is a Python interface to the NIfTI I/O libraries. Using
NiBabel, one can easily read and write NIfTI and ANALYZE images from within
Python. The :class:`~nifti.image.NiftiImage` class provides pythonic
access to the full header information and for a maximum of interoperability the
image data is made available via NumPy arrays.

===============================
 nibabel python implementation
===============================

Quickstart::

   import nipy.io.imageformats

   img1 = nifti.load('my_file.nii')
   img2 = nifti.load('other_file.nii.gz')
   img3 = nifti.load('spm_file.img')

   data = img1.get_data()
   affine = img1.get_affine()

   print img1

   nifti.save(img1, 'my_file_copy.nii.gz')

   new_image = nifti.Nifti1Image(data, affine)
   nifti.save(new_image, 'new_image.nii.gz')
"""

__docformat__ = 'restructuredtext'

# expose the two main classes
#from nifti.image import NiftiImage, MemMappedNiftiImage

# canonical version string
__version__ = '0.20090303.1'


# module imports
from nipy.io.imageformats import analyze as ana
from nipy.io.imageformats import spm99analyze as spm99
from nipy.io.imageformats import spm2analyze as spm2
from nipy.io.imageformats import nifti1 as ni1
from nipy.io.imageformats import minc
# object imports
from nipy.io.imageformats.loadsave import load, save
from nipy.io.imageformats.analyze import AnalyzeHeader, AnalyzeImage
from nipy.io.imageformats.spm99analyze import Spm99AnalyzeHeader, Spm99AnalyzeImage
from nipy.io.imageformats.spm2analyze import Spm2AnalyzeHeader, Spm2AnalyzeImage
from nipy.io.imageformats.nifti1 import Nifti1Header, Nifti1Image
from nipy.io.imageformats.minc import MincHeader, MincImage
from nipy.io.imageformats.funcs import squeeze_image, concat_images, four_to_three
from nipy.io.imageformats.affines import io_orientation
