"""
The L{Image<image.Image>} class provides the interface which should be used
by users at the application level. The image provides a coordinate map,
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

from nipy.testing import Tester
test = Tester().test
bench = Tester().bench

