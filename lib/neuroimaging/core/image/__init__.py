"""
The Image class provides the interface which should be used
by users at the application level. It is build onto of a
BaseImage object (self._source) which handles the actual
representation of the data. A base image provides a grid,
a data type and the data itself, while the main Image class
builds on top of these.

A BaseImage object can be created from an ndarray (ArrayImage)
or from a file (Formats). 


TODO: The Formats class does not currently subclass from BaseImage,
or even provide the correct interface, so this needs to be fixed
to fall in line with the class structure shown here.

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



