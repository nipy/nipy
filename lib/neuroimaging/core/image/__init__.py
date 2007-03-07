"""
The L{Image<image.Image>} class provides the interface which should be used
by users at the application level. It is build on top of a
L{BaseImage<image.base_image.BaseImage>} object (self._source) which handles
the actual representation of the data. A base image provides a grid,
a data type and the data itself, while the main L{Image<image.Image>} class
builds on top of these.

A L{BaseImage<image.base_image.BaseImage>} object can be created from an
ndarray (L{ArrayImage<image.base_image.ArrayImage>})
or from a file (L{Format<neuroimaging.data_io.formats.Format>}). 

Class structure::

   Application Level
 ----------------------
        Image
          |
          o
          |
      BaseImage
          |
          |
      ------------
      |          |
   Formats   ArrayImage
      |
   Binary   
      |
   ------------------
   |        |       |
 Nifti   Analyze  ECAT
"""
__docformat__ = 'restructuredtext'

import image, base_image, roi, iterators

def test(level=1, verbosity=1, flags=[]):
    from neuroimaging.utils.test_decorators import set_flags
    set_flags(flags)
    from numpy.testing import NumpyTest
    return NumpyTest().test(level, verbosity)


