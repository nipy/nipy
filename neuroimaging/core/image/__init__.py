"""
The L{Image<image.Image>} class provides the interface which should be used
by users at the application level. The image provides a grid,
and the data itself.


Class structure::

   Application Level

TODO: I think this graph is unnecessary and wrong after removing
      BaseImage, JT

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

import image, roi, generators

def test(level=1, verbosity=1, flags=[]):
    from neuroimaging.utils.testutils import set_flags
    set_flags(flags)
    from numpy.testing import NumpyTest
    return NumpyTest().test(level, verbosity)


