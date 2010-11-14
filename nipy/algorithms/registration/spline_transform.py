''' Spline transformation classes '''

import numpy as np

from .transform import Transform

class SplineTransform(Transform):
    def __init__(self, control_points, kernel, param=None):
        """ Create spline transform

        Parameters
        ----------
        control_points : (K,3) array-like
            spline control points
        kernel : callable
            kernel G where G(x - x_k) for point x and control point k is the
            basis for the parameters
        param : None or (K,3) array, optional
            coefficients for each control point.  None gives array of zeros as
            starting default
        """
        self.control_points = np.asarray(control_points)
        K = control_points.shape[0]
        self.kernel = kernel
        if param is None:
            self.param = np.zeros((K,3))
        else:
            self.param = np.asarray(param)
        self._cache = {'last_pts': None, 'Gik': None}

    def apply(self, pts):
        # logic for applying transform to points
        if pts is self._cache['last_pts']:
            Gik = self._cache['Gik']
        else: # calculate and cache Gik matrix
            # Gik is an N by K matrix where G is the kernel, i indexes input
            # points, k indexes control points, and Gik[i, k] is G(x_i - x_k)
            Gik = None
            self._cache['last_pts'] = pts
            self._cache['Gik'] = Gik
        return np.dot(Gik, self.param)
