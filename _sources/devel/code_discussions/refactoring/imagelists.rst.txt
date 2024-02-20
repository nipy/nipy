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
