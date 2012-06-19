# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Intensity-based image registration
"""

from sys import maxint

import numpy as np

from ...core.image.image_spaces import (make_xyz_image,
                                        as_xyz_image,
                                        xyz_affine)

from .optimizer import configure_optimizer
from .affine import inverse_affine, subgrid_affine, affine_transforms
from .chain_transform import ChainTransform
from .similarity_measures import similarity_measures as _sms
from ._registration import _joint_histogram


# Module globals
VERBOSE = True  # enables online print statements
OPTIMIZER = 'powell'
XTOL = 1e-2
FTOL = 1e-2
GTOL = 1e-3
MAXITER = 25
MAXFUN = None
CLAMP_DTYPE = 'short'  # do not edit
BINS = 256
SIMILARITY = 'crl1'
INTERP = 'pv'
NPOINTS = 64 ** 3

# Dictionary of interpolation methods (partial volume, trilinear,
# random)
interp_methods = {'pv': 0, 'tri': 1, 'rand': -1}


class HistogramRegistration(object):
    """
    A class to reprensent a generic intensity-based image registration
    algorithm.
    """
    def __init__(self, from_img, to_img,
                 from_bins=BINS, to_bins=None,
                 from_mask=None, to_mask=None,
                 similarity=SIMILARITY, interp=INTERP,
                 **kwargs):
        """
        Creates a new histogram registration object.

        Parameters
        ----------
        from_img : nipy-like image
          `From` image
        to_img : nipy-like image
          `To` image
        from_bins : integer
          Number of histogram bins to represent the `from` image
        to_bins : integer
          Number of histogram bins to represent the `to` image
        from_mask : nipy image-like
          Mask to apply to the `from` image
        to_mask : nipy image-like
          Mask to apply to the `to` image
        similarity : str or callable
          Cost-function for assessing image similarity. If a string,
          one of 'cc': correlation coefficient, 'cr': correlation
          ratio, 'crl1': L1-norm based correlation ratio, 'mi': mutual
          information, 'nmi': normalized mutual information, 'slr':
          supervised log-likelihood ratio. If a callable, it should
          take a two-dimensional array representing the image joint
          histogram as an input and return a float.
       interp : str
         Interpolation method.  One of 'pv': Partial volume, 'tri':
         Trilinear, 'rand': Random interpolation.  See ``joint_histogram.c``
        """
        # Function assumes xyx_affine for inputs
        from_img = as_xyz_image(from_img)
        to_img = as_xyz_image(to_img)
        if not from_mask is None:
            from_mask = as_xyz_image(from_mask)
        if not to_mask is None:
            to_mask = as_xyz_image(to_mask)

        # Binning sizes
        if to_bins == None:
            to_bins = from_bins

        # Clamping of the `from` image. The number of bins may be
        # overriden if unnecessarily large.
        mask = None
        if not from_mask is None:
            mask = from_mask.get_data()
        data, from_bins = clamp(from_img.get_data(), bins=from_bins, mask=mask)
        self._from_img = make_xyz_image(data, xyz_affine(from_img), 'scanner')
        # Set the subsampling.  This also sets the _from_data and _vox_coords
        # attributes
        self.subsample()

        # Clamping of the `to` image including padding with -1
        mask = None
        if not to_mask is None:
            mask = to_mask.get_data()
        data, to_bins = clamp(to_img.get_data(), bins=to_bins, mask=mask)
        self._to_data = -np.ones(np.array(to_img.shape) + 2, dtype=CLAMP_DTYPE)
        self._to_data[1:-1, 1:-1, 1:-1] = data
        self._to_inv_affine = inverse_affine(xyz_affine(to_img))

        # Joint histogram: must be double contiguous as it will be
        # passed to C routines which assume so
        self._joint_hist = np.zeros([from_bins, to_bins], dtype='double')

        # Set default registration parameters
        self._set_interp(interp)
        self._set_similarity(similarity, **kwargs)

    def _get_interp(self):
        return interp_methods.keys()\
            [interp_methods.values().index(self._interp)]

    def _set_interp(self, interp):
        self._interp = interp_methods[interp]

    interp = property(_get_interp, _set_interp)

    def subsample(self, spacing=None, corner=[0, 0, 0], size=None,
                  npoints=NPOINTS):
        """
        Defines a subset of the `from` image to restrict joint
        histogram computation.

        Parameters
        ----------
        spacing : sequence (3,) of positive integers
          Subsampling of image in voxels, where None (default) results
          in the subsampling to be automatically adjusted to roughly
          match a cubic grid with `npoints` voxels
        corner : sequence (3,) of positive integers
          Bounding box origin in voxel coordinates
        size : sequence (3,) of positive integers
          Desired bounding box size
        npoints : positive integer
          Desired number of voxels in the bounding box. If a `spacing`
          argument is provided, then `npoints` is ignored.
        """
        if spacing == None:
            spacing = [1, 1, 1]
        else:
            npoints = None
        if size == None:
            size = self._from_img.shape
        slicer = lambda: tuple([slice(corner[i],
                                      size[i] + corner[i],
                                      spacing[i]) for i in range(3)])
        fov_data = self._from_img.get_data()[slicer()]
        # Adjust spacing to match desired field of view size
        if npoints:
            spacing = ideal_spacing(fov_data, npoints=npoints)
            fov_data = self._from_img.get_data()[slicer()]
        self._from_data = fov_data
        self._from_npoints = (fov_data >= 0).sum()
        self._from_affine = subgrid_affine(xyz_affine(self._from_img),
                                           slicer())
        # We cache the voxel coordinates of the clamped image
        self._vox_coords =\
            np.indices(self._from_data.shape).transpose((1, 2, 3, 0))

    def _set_similarity(self, similarity='cr', **kwargs):
        if similarity in _sms:
            self._similarity = similarity
            self._similarity_call =\
                _sms[similarity](self._joint_hist.shape, **kwargs)
        else:
            if not hasattr(similarity, '__call__'):
                raise ValueError('similarity should be callable')
            self._similarity = 'custom'
            self._similarity_call = similarity

    def _get_similarity(self):
        return self._similarity

    similarity = property(_get_similarity, _set_similarity)

    def eval(self, T):
        """
        Evaluate similarity function given a world-to-world transform.

        Parameters
        ----------
        T : Transform
            Transform object implementing ``apply`` method
        """
        Tv = ChainTransform(T, pre=self._from_affine, post=self._to_inv_affine)
        return self._eval(Tv)

    def _eval(self, Tv):
        """
        Evaluate similarity function given a voxel-to-voxel transform.

        Parameters
        ----------
        Tv : Transform
             Transform object implementing ``apply`` method
             Should map voxel space to voxel space
        """
        # trans_vox_coords needs be C-contiguous
        trans_vox_coords = Tv.apply(self._vox_coords)
        interp = self._interp
        if self._interp < 0:
            interp = - np.random.randint(maxint)
        _joint_histogram(self._joint_hist,
                         self._from_data.flat,  # array iterator
                         self._to_data,
                         trans_vox_coords,
                         interp)
        return self._similarity_call(self._joint_hist)

    def optimize(self, T, optimizer=OPTIMIZER, **kwargs):
        """ Optimize transform `T` with respect to similarity measure.

        The input object `T` will change as a result of the optimization.

        Parameters
        ----------
        T : object or str
          An object representing a transformation that should
          implement ``apply`` method and ``param`` attribute or
          property. If a string, one of 'rigid', 'similarity', or
          'affine'. The corresponding transformation class is then
          initialized by default.
        optimizer : str
          Name of optimization function (one of 'powell', 'steepest',
          'cg', 'bfgs', 'simplex')
        **kwargs : dict
          keyword arguments to pass to optimizer
        """
        # Replace T if a string is passed
        if T in affine_transforms:
            T = affine_transforms[T]()

        # Pull callback out of keyword arguments, if present
        callback = kwargs.pop('callback', None)

        # Create transform chain object with T generating params
        Tv = ChainTransform(T, pre=self._from_affine, post=self._to_inv_affine)
        tc0 = Tv.param

        # Cost function to minimize
        def cost(tc):
            # This is where the similarity function is calculcated
            Tv.param = tc
            return -self._eval(Tv)

        # Callback during optimization
        if callback == None and VERBOSE:

            def callback(tc):
                Tv.param = tc
                print(Tv.optimizable)
                print(str(self.similarity) + ' = %s' % self._eval(Tv))
                print('')

        # Switching to the appropriate optimizer
        if VERBOSE:
            print('Initial guess...')
            print(Tv.optimizable)

        kwargs.setdefault('xtol', XTOL)
        kwargs.setdefault('ftol', FTOL)
        kwargs.setdefault('gtol', GTOL)
        kwargs.setdefault('maxiter', MAXITER)
        kwargs.setdefault('maxfun', MAXFUN)

        fmin, args, kwargs = configure_optimizer(optimizer,
                                                 fprime=None,
                                                 fhess=None,
                                                 **kwargs)

        # Output
        if VERBOSE:
            print ('Optimizing using %s' % fmin.__name__)
        kwargs['callback'] = callback
        Tv.param = fmin(cost, tc0, *args, **kwargs)
        return Tv.optimizable

    def explore(self, T0, *args):
        """
        Evaluate the similarity at the transformations specified by
        sequences of parameter values.

        For instance:

        explore(T0, (0, [-1,0,1]), (4, [-2.,2]))
        """
        nparams = T0.param.size
        deltas = [[0] for i in range(nparams)]
        for a in args:
            deltas[a[0]] = a[1]
        grids = np.mgrid[[slice(0, len(d)) for d in deltas]]
        ntrials = np.prod(grids.shape[1:])
        Deltas = [np.asarray(deltas[i])[grids[i, :]].ravel()\
                      for i in range(nparams)]
        simis = np.zeros(ntrials)
        params = np.zeros([nparams, ntrials])

        Tv = ChainTransform(T0, pre=self._from_affine,
                            post=self._to_inv_affine)
        param0 = Tv.param
        for i in range(ntrials):
            param = param0 + np.array([D[i] for D in Deltas])
            Tv.param = param
            simis[i] = self._eval(Tv)
            params[:, i] = param

        return simis, params


def _clamp(x, y, bins=BINS, mask=None):

    # Threshold
    dmaxmax = 2 ** (8 * y.dtype.itemsize - 1) - 1
    dmax = bins - 1  # default output maximum value
    if dmax > dmaxmax:
        raise ValueError('Excess number of bins')
    xmin = float(x.min())
    xmax = float(x.max())
    d = xmax - xmin

    """
    If the image dynamic is small, no need for compression: just
    downshift image values and re-estimate the dynamic range (hence
    xmax is translated to xmax-tth casted to the appropriate
    dtype. Otherwise, compress after downshifting image values (values
    equal to the threshold are reset to zero).
    """
    if issubclass(x.dtype.type, np.integer) and d <= dmax:
        y[:] = x - xmin
        bins = int(d) + 1
    else:
        a = dmax / d
        y[:] = np.round(a * (x - xmin))

    return y, bins


def clamp(x, bins=BINS, mask=None):
    """
    Clamp array values that fall within a given mask in the range
    [0..bins-1] and reset masked values to -1.

    Parameters
    ----------
    x : ndarray
      The input array

    bins : number
      Desired number of bins

    mask : ndarray, tuple or slice
      Anything such that x[mask] is an array.

    Returns
    -------
    y : ndarray
      Clamped array, masked items are assigned -1

    bins : number
      Adjusted number of bins

    """
    if bins > np.iinfo(np.short).max:
        raise ValueError('Too large a bin size')
    y = -np.ones(x.shape, dtype=CLAMP_DTYPE)
    if mask == None:
        y, bins = _clamp(x, y, bins)
    else:
        ym = y[mask]
        xm = x[mask]
        ym, bins = _clamp(xm, ym, bins)
        y[mask] = ym
    return y, bins


def ideal_spacing(data, npoints):
    """
    Tune spacing factors so that the number of voxels in the
    output block matches a given number.

    Parameters
    ----------
    data : ndarray or sequence
      Data image to subsample

    npoints : number
      Target number of voxels (negative values will be ignored)

    Returns
    -------
    spacing: ndarray
      Spacing factors

    """
    dims = data.shape
    actual_npoints = (data >= 0).sum()
    spacing = np.ones(3, dtype='uint')

    while actual_npoints > npoints:

        # Subsample the direction with the highest number of samples
        ddims = dims / spacing
        if ddims[0] >= ddims[1] and ddims[0] >= ddims[2]:
            dir = 0
        elif ddims[1] > ddims[0] and ddims[1] >= ddims[2]:
            dir = 1
        else:
            dir = 2
        spacing[dir] += 1
        subdata = data[::spacing[0], ::spacing[1], ::spacing[2]]
        actual_npoints = (subdata >= 0).sum()

    return spacing
