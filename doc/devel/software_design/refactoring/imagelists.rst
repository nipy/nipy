========================
 Refactoring imagelists
========================

Usecases for ImageList
======================

Thus far only used in anger in :mod:`nipy.modalities.fmri.fmristat.models`

From that file, an object ``obj`` of class :class:`FmriImageList` must:

* return 4D array from ``np.asarray(obj)``, such that the first axis
  (axis 0) is the axis over which the model is applied
* be indexable such that ``obj[0]`` returns an Image instance, with
  valid ``shape`` and ``coordmap`` attributes for a time-point 3D volume
  in the 4D time-series.
* have an attribute ``volume_start_times`` giving times of the start of
  each of the volumes in the 4D time series.
* Return the number of volumes in the time-series from ``len(obj)``

Output of ``grin ImageList``
============================

::
  ./nipy/core/api.py:
     14 : from nipy.core.image.image_list import ImageList
  ./nipy/core/image/image_list.py:
      8 : class ImageList(object):
     24 :         >>> from nipy.core.api import Image, ImageList
     27 :         >>> ilist = ImageList(funcim)
     30 :         Slicing an ImageList returns a new ImageList
     32 :         >>> isinstance(sublist, ImageList)
     35 :         Indexing an ImageList returns a new Image
     40 :         >>> isinstance(newimg, ImageList)
     82 :             return ImageList(images=self.list[index])
     86 :         Return another ImageList instance consisting with
     90 :         return ImageList(images=self.list[i:j])
     99 :         >>> from nipy.core.api import ImageList
    102 :         >>> ilist = ImageList(funcim)
  ./nipy/core/image/tests/test_image_list.py:
      7 : from nipy.core.image.image_list import ImageList
      9 : from nipy.modalities.fmri.api import FmriImageList, fromimage
     22 :     fl = ImageList([f.frame(i) for i in range(f.shape[0])])
     30 : ## from nipy.core.image.image_list import ImageList
     34 : ## ilist = ImageList(funcim)
  ./nipy/modalities/fmri/api.py:
      1 : from fmri import FmriImageList, fmri_generator, fromimage
  ./nipy/modalities/fmri/fmri.py:
      3 : from nipy.core.api import ImageList, Image, \
      6 : class FmriImageList(ImageList):
     16 :         A lightweight implementation of an fMRI image as in ImageList
     31 :         nipy.core.image_list.ImageList
     35 :         >>> from nipy.modalities.fmri.api import FmriImageList, fromimage
     42 :         >>> ilist = FmriImageList(funcim)
     50 :         ImageList.__init__(self, images=images)
     66 :         else return an FmriImageList with images=self.list[index].
     72 :             return FmriImageList(images=self.list[index], 
     93 :     Note that if data is an FmriImageList instance, there is more 
    107 :     """Create an FmriImageList from a 4D Image.
    142 :     return FmriImageList(images=images, 
  ./nipy/modalities/fmri/fmristat/tests/test_utils.py:
     12 : from nipy.modalities.fmri.api import FmriImageList
     51 :     #     test_fmri.nii.gz) and the calling convention for FmriImageList
     56 :         self.img = FmriImageList("test_fmri.hdr", datasource=repository, volume_start_times=volume_start_times,
  ./nipy/modalities/fmri/fmristat/model.py:
     26 : from nipy.modalities.fmri.api import FmriImageList, fmri_generator
    109 :     fmri_image : `FmriImageList`
    196 :     fmri_image : `FmriImageList`
    318 :     image: FmriImageList 
    336 :     if isinstance(fmri_image, FmriImageList):
    353 :         raise ValueError, "expecting FmriImageList or 4d Image"
  ./nipy/modalities/fmri/tests/test_iterators.py:
      7 : from nipy.modalities.fmri.api import FmriImageList
     18 :         self.img = FmriImageList(im)
     84 :         tmp = FmriImageList(self.img[:] * 1., self.img.coordmap)
  ./doc/devel/planning/TODO.rst:
     49 :   ImageList and FmriImageList.
    147 : * FmriImageList.emptycopy() - Is there a better way to do this?
