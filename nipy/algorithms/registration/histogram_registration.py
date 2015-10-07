# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Intensity-based image registration
"""
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import scipy.ndimage as nd

from ...core.image.image_spaces import (make_xyz_image,
                                        as_xyz_image,
                                        xyz_affine)

from .optimizer import configure_optimizer
from .affine import inverse_affine, subgrid_affine, affine_transforms
from .chain_transform import ChainTransform
from .similarity_measures import similarity_measures as _sms
from ._registration import _joint_histogram

MAX_INT = np.iinfo(np.intp).max

# Module globals
VERBOSE = True  # enables online print statements
OPTIMIZER = 'powell'
XTOL = 1e-2
FTOL = 1e-2
GTOL = 1e-3
MAXITER = 25
MAXFUN = None
CLAMP_DTYPE = 'short'  # do not edit
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
                 from_bins=256, to_bins=None,
                 from_mask=None, to_mask=None,
                 similarity='crl1', interp='pv',
                 smooth=0, renormalize=False, dist=None):
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
        from_mask : array-like
          Mask to apply to the `from` image
        to_mask : array-like
          Mask to apply to the `to` image
        similarity : str or callable
          Cost-function for assessing image similarity. If a string,
          one of 'cc': correlation coefficient, 'cr': correlation
          ratio, 'crl1': L1-norm based correlation ratio, 'mi': mutual
          information, 'nmi': normalized mutual information, 'slr':
          supervised log-likelihood ratio. If a callable, it should
          take a two-dimensional array representing the image joint
          histogram as an input and return a float.
       dist: None or array-like
          Joint intensity probability distribution model for use with the 
          'slr' measure. Should be of shape (from_bins, to_bins).
       interp : str
         Interpolation method.  One of 'pv': Partial volume, 'tri':
         Trilinear, 'rand': Random interpolation.  See ``joint_histogram.c``
       smooth : float
         Standard deviation in millimeters of an isotropic Gaussian
         kernel used to smooth the `To` image. If 0, no smoothing is
         applied.
        """
        # Function assumes xyx_affine for inputs
        from_img = as_xyz_image(from_img)
        to_img = as_xyz_image(to_img)

        # Binning sizes
        if to_bins is None:
            to_bins = from_bins

        # Clamping of the `from` image. The number of bins may be
        # overriden if unnecessarily large.
        data, from_bins_adjusted = clamp(from_img.get_data(), from_bins,
                                         mask=from_mask)
        if not similarity == 'slr':
            from_bins = from_bins_adjusted
        self._from_img = make_xyz_image(data, xyz_affine(from_img), 'scanner')
        # Set field of view in the `from` image with potential
        # subsampling for faster similarity evaluation. This also sets
        # the _from_data and _vox_coords attributes
        if from_mask is None:
            self.subsample(npoints=NPOINTS)
        else:
            corner, size = smallest_bounding_box(from_mask)
            self.set_fov(corner=corner, size=size, npoints=NPOINTS)

        # Clamping of the `to` image including padding with -1
        self._smooth = float(smooth)
        if self._smooth < 0:
            raise ValueError('smoothing kernel cannot have negative scale')
        elif self._smooth > 0:
            data = smooth_image(to_img.get_data(), xyz_affine(to_img),
                                self._smooth)
        else:
            data = to_img.get_data()
        data, to_bins_adjusted = clamp(data, to_bins, mask=to_mask)
        if not similarity == 'slr':
            to_bins = to_bins_adjusted
        self._to_data = -np.ones(np.array(to_img.shape) + 2, dtype=CLAMP_DTYPE)
        self._to_data[1:-1, 1:-1, 1:-1] = data
        self._to_inv_affine = inverse_affine(xyz_affine(to_img))

        # Joint histogram: must be double contiguous as it will be
        # passed to C routines which assume so
        self._joint_hist = np.zeros([from_bins, to_bins], dtype='double')

        # Set default registration parameters
        self._set_interp(interp)
        self._set_similarity(similarity, renormalize=renormalize, dist=dist)

    def _get_interp(self):
        return list(interp_methods.keys())[\
            list(interp_methods.values()).index(self._interp)]

    def _set_interp(self, interp):
        self._interp = interp_methods[interp]

    interp = property(_get_interp, _set_interp)

    def set_fov(self, spacing=None, corner=(0, 0, 0), size=None,
                npoints=None):
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
        if spacing is None and npoints is None:
            spacing = [1, 1, 1]
        if size is None:
            size = self._from_img.shape
        slicer = lambda c, s, sp:\
            tuple([slice(c[i], s[i] + c[i], sp[i]) for i in range(3)])
        # Adjust spacing to match desired field of view size
        if spacing is not None:
            fov_data = self._from_img.get_data()[slicer(corner, size, spacing)]
        else:
            fov_data = self._from_img.get_data()[
                slicer(corner, size, [1, 1, 1])]
            spacing = ideal_spacing(fov_data, npoints=npoints)
            fov_data = self._from_img.get_data()[slicer(corner, size, spacing)]
        self._from_data = fov_data
        self._from_npoints = (fov_data >= 0).sum()
        self._from_affine = subgrid_affine(xyz_affine(self._from_img),
                                           slicer(corner, size, spacing))
        # We cache the voxel coordinates of the clamped image
        self._vox_coords =\
            np.indices(self._from_data.shape).transpose((1, 2, 3, 0))

    def subsample(self, spacing=None, npoints=None):
        self.set_fov(spacing=spacing, npoints=npoints)

    def _set_similarity(self, similarity, renormalize=False, dist=None):
        if similarity in _sms:
            if similarity == 'slr':
                if dist is None:
                    raise ValueError('slr measure requires a joint intensity distribution model, '
                                     'see `dist` argument of HistogramRegistration')
                if dist.shape != self._joint_hist.shape:
                    raise ValueError('Wrong shape for the `dist` argument')
            self._similarity = similarity
            self._similarity_call =\
                _sms[similarity](self._joint_hist.shape, renormalize, dist)
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

    def eval_gradient(self, T, epsilon=1e-1):
        """
        Evaluate the gradient of the similarity function wrt
        transformation parameters.

        The gradient is approximated using central finite differences
        at the transformation specified by `T`. The input
        transformation object `T` is modified in place unless it has a
        ``copy`` method.

        Parameters
        ----------
        T : Transform
            Transform object implementing ``apply`` method
        epsilon : float
            Step size for finite differences in units of the
            transformation parameters

        Returns
        -------
        g : ndarray
            Similarity gradient estimate
        """
        param0 = T.param.copy()
        if hasattr(T, 'copy'):
            T = T.copy()

        def simi(param):
            T.param = param
            return self.eval(T)

        return approx_gradient(simi, param0, epsilon)

    def eval_hessian(self, T, epsilon=1e-1, diag=False):
        """
        Evaluate the Hessian of the similarity function wrt
        transformation parameters.

        The Hessian or its diagonal is approximated at the
        transformation specified by `T` using central finite
        differences. The input transformation object `T` is modified
        in place unless it has a ``copy`` method.

        Parameters
        ----------
        T : Transform
            Transform object implementing ``apply`` method
        epsilon : float
            Step size for finite differences in units of the
            transformation parameters
        diag : bool
            If True, approximate the Hessian by a diagonal matrix.

        Returns
        -------
        H : ndarray
            Similarity Hessian matrix estimate
        """
        param0 = T.param.copy()
        if hasattr(T, 'copy'):
            T = T.copy()

        def simi(param):
            T.param = param
            return self.eval(T)

        if diag:
            return np.diag(approx_hessian_diag(simi, param0, epsilon))
        else:
            return approx_hessian(simi, param0, epsilon)

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
            interp = - np.random.randint(MAX_INT)
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

        Returns
        -------
        T : object
          Locally optimal transformation
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
        if callback is None and VERBOSE:

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
            print('Optimizing using %s' % fmin.__name__)
        kwargs['callback'] = callback
        Tv.param = fmin(cost, tc0, *args, **kwargs)
        return Tv.optimizable

    def explore(self, T, *args):
        """
        Evaluate the similarity at the transformations specified by
        sequences of parameter values.

        For instance:

        s, p = explore(T, (0, [-1,0,1]), (4, [-2.,2]))

        Parameters
        ----------
        T : object
          Transformation around which the similarity function is to be
          evaluated. It is modified in place unless it has a ``copy``
          method.
        args : tuple
          Each element of `args` is a sequence of two elements, where
          the first element specifies a transformation parameter axis
          and the second element gives the successive parameter values
          to evaluate along that axis.

        Returns
        -------
        s : ndarray
          Array of similarity values
        p : ndarray
          Corresponding array of evaluated transformation parameters
        """
        nparams = T.param.size
        if hasattr(T, 'copy'):
            T = T.copy()
        deltas = [[0] for i in range(nparams)]
        for a in args:
            deltas[a[0]] = a[1]
        grids = np.mgrid[[slice(0, len(d)) for d in deltas]]
        ntrials = np.prod(grids.shape[1:])
        Deltas = [np.asarray(deltas[i])[grids[i, :]].ravel()\
                      for i in range(nparams)]
        simis = np.zeros(ntrials)
        params = np.zeros([nparams, ntrials])

        Tv = ChainTransform(T, pre=self._from_affine,
                            post=self._to_inv_affine)
        param0 = Tv.param
        for i in range(ntrials):
            param = param0 + np.array([D[i] for D in Deltas])
            Tv.param = param
            simis[i] = self._eval(Tv)
            params[:, i] = param

        return simis, params


def _clamp(x, y, bins):

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


def clamp(x, bins, mask=None):
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
    if mask is None:
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


def smallest_bounding_box(msk):
    """
    Extract the smallest bounding box from a mask

    Parameters
    ----------
    msk : ndarray
      Array of boolean

    Returns
    -------
    corner: ndarray
      3-dimensional coordinates of bounding box corner
    size: ndarray
      3-dimensional size of bounding box
    """
    x, y, z = np.where(msk > 0)
    corner = np.array([x.min(), y.min(), z.min()])
    size = np.array([x.max() + 1, y.max() + 1, z.max() + 1])
    return corner, size


def approx_gradient(f, x, epsilon):
    """
    Approximate the gradient of a function using central finite
    differences

    Parameters
    ----------
    f: callable
      The function to differentiate
    x: ndarray
      Point where the function gradient is to be evaluated
    epsilon: float
      Stepsize for finite differences

    Returns
    -------
    g: ndarray
      Function gradient at `x`
    """
    n = len(x)
    g = np.zeros(n)
    ei = np.zeros(n)
    for i in range(n):
        ei[i] = .5 * epsilon
        g[i] = (f(x + ei) - f(x - ei)) / epsilon
        ei[i] = 0
    return g


def approx_hessian_diag(f, x, epsilon):
    """
    Approximate the Hessian diagonal of a function using central
    finite differences

    Parameters
    ----------
    f: callable
      The function to differentiate
    x: ndarray
      Point where the Hessian is to be evaluated
    epsilon: float
      Stepsize for finite differences

    Returns
    -------
    h: ndarray
      Diagonal of the Hessian at `x`
    """
    n = len(x)
    h = np.zeros(n)
    ei = np.zeros(n)
    fx = f(x)
    for i in range(n):
        ei[i] = epsilon
        h[i] = (f(x + ei) + f(x - ei) - 2 * fx) / (epsilon ** 2)
        ei[i] = 0
    return h


def approx_hessian(f, x, epsilon):
    """
    Approximate the full Hessian matrix of a function using central
    finite differences

    Parameters
    ----------
    f: callable
      The function to differentiate
    x: ndarray
      Point where the Hessian is to be evaluated
    epsilon: float
      Stepsize for finite differences

    Returns
    -------
    H: ndarray
      Hessian matrix at `x`
    """
    n = len(x)
    H = np.zeros((n, n))
    ei = np.zeros(n)
    for i in range(n):
        ei[i] = .5 * epsilon
        g1 = approx_gradient(f, x + ei, epsilon)
        g2 = approx_gradient(f, x - ei, epsilon)
        H[i, :] = (g1 - g2) / epsilon
        ei[i] = 0
    return H


def smooth_image(data, affine, sigma):
    """
    Smooth an image by an isotropic Gaussian filter

    Parameters
    ----------
    data: ndarray
      Image data array
    affine: ndarray
      Image affine transform
    sigma: float
      Filter standard deviation in mm

    Returns
    -------
    sdata: ndarray
      Smoothed data array
    """
    sigma_vox = sigma / np.sqrt(np.sum(affine[0:3, 0:3] ** 2, 0))
    return nd.gaussian_filter(data, sigma_vox)
