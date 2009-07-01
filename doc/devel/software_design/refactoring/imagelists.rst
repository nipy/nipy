========================
 Refactoring imagelists
========================

Usecases for ImageList
======================

Thus far only used in anger in
:mod:`nipy.modalities.fmri.fmristat.model`, similarly in
:mod:`nipy.modalities.fmri.spm.model`.

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
     17 :     f = ImageList.from_image(ff)
     19 :     fl = ImageList([f.frame(i) for i in range(f.shape[0])])
     27 : ## from nipy.core.image.image_list import ImageList
     31 : ## ilist = ImageList(funcim)
  ./nipy/modalities/fmri/api.py:
      1 : from fmri import FmriImageList, fmri_generator
  ./nipy/modalities/fmri/fmri.py:
      3 : from nipy.core.api import ImageList, Image, \
      6 : class FmriImageList(ImageList):
     16 :         A lightweight implementation of an fMRI image as in ImageList
     31 :         nipy.core.image_list.ImageList
     38 :         >>> fmrilist = FmriImageList.from_image(funcim)
     39 :         >>> ilist = FmriImageList(funcim)
     47 :         ImageList.__init__(self, images=images)
     63 :         else return an FmriImageList with images=self.list[index].
     69 :             return FmriImageList(images=self.list[index], 
     84 :         """Create an FmriImageList from a 4D Image.
    141 :     If data is an ``FmriImageList`` instance, there is more overhead
  ./nipy/modalities/fmri/spm/model.py:
     60 :     fmri_image : `FmriImageList`
  ./nipy/modalities/fmri/fmristat/tests/test_utils.py:
     12 : from nipy.modalities.fmri.api import FmriImageList
     51 :     #     test_fmri.nii.gz) and the calling convention for FmriImageList
     56 :         self.img = FmriImageList("test_fmri.hdr", datasource=repository, volume_start_times=volume_start_times,
  ./nipy/modalities/fmri/fmristat/tests/test_model.py:
      9 : from nipy.modalities.fmri.api import FmriImageList
     48 :         fmriims = FmriImageList.from_image(funcim, volume_start_times=2.)
  ./nipy/modalities/fmri/fmristat/tests/test_iterables.py:
      7 : from nipy.modalities.fmri.api import FmriImageList, fmri_generator
     51 :         self.fi = FmriImageList.from_image(load_image(funcfile))
     54 :         # array from FmriImageList
  ./nipy/modalities/fmri/fmristat/model.py:
     20 : from nipy.modalities.fmri.api import FmriImageList, fmri_generator
     97 :     fmri_image : `FmriImageList` or 4D image
    188 :     fmri_image : `FmriImageList`
    268 :     fmri_image : ``FmriImageList``
    306 :     fmri_image : ``FmriImageList``
    326 :     fmri_image : ``FmriImageList`` or 4D image
    351 :     fmri_image : ``FmriImageList`` or 4D image
    352 :        If ``FmriImageList``, needs attributes ``volume_start_times``,
    365 :     if isinstance(fmri_image, FmriImageList):
    382 :         raise ValueError, "expecting FmriImageList or 4d Image"
  ./nipy/modalities/fmri/tests/test_iterators.py:
      7 : from nipy.modalities.fmri.api import FmriImageList
     18 :         self.img = FmriImageList(im)
     84 :         tmp = FmriImageList(self.img[:] * 1., self.img.coordmap)
  ./nipy/modalities/fmri/tests/test_pca.py:
      3 : from nipy.modalities.fmri.api import FmriImageList
     13 :         self.fmridata = FmriImageList.from_image(self.img)
  ./nipy/modalities/fmri/tests/test_fmri.py:
      9 : from nipy.modalities.fmri.api import fmri_generator, FmriImageList
     29 :     test = FmriImageList.from_image(load_image(fname))
  ./doc/devel/planning/TODO.rst:
     49 :   ImageList and FmriImageList.
    147 : * FmriImageList.emptycopy() - Is there a better way to do this?
