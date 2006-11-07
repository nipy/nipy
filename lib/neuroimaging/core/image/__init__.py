"""
The Image class provides the interface which should be used
by users at the application level. It is build on top of a
BaseImage object (self._source) which handles the actual
representation of the data. A base image provides a grid,
a data type and the data itself, while the main Image class
builds on top of these.

A BaseImage object can be created from an ndarray (ArrayImage)
or from a file (Formats). 

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



