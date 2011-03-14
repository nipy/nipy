""" Chain transforms """

from .affine import Affine


class ChainTransform(object):
    def __init__(self, optimizable, pre=None, post=None):
        """ Create chain transform instance

        Parameters
        ----------
        optimizable : array or Transform
            Transform that we are optimizing.  If this is an array, then assume
            it's an affine matrix.
        pre : None or array or Transform, optional
            If not None, a transform that should be applied to points before
            applying the `optimizable` transform.  If an array, then assume it's
            an affine matrix.
        post : None or Transform, optional
            If not None, a transform that should be applied to points after
            applying any `pre` transform, and then the `optimizable`
            transform.  If an array, assume it's an affine matrix
        """
        if not hasattr(optimizable, 'param'):
            raise ValueError('Input transform should be optimizable')
        if not hasattr(optimizable, 'apply'):
            optimizable = Affine(optimizable)
        if not hasattr(pre, 'apply'):
            pre = Affine(pre)
        if not hasattr(post, 'apply'):
            post = Affine(post)
        self.optimizable = optimizable
        self.pre = pre
        self.post = post

    def apply(self, pts):
        """ Apply full transformation to points `pts`

        If there are N points, then `pts` will be N by 3

        Parameters
        ----------
        pts : array-like
            array of points

        Returns
        -------
        transformed_pts : array
            N by 3 array of transformed points
        """
        composed = self.post.compose(self.optimizable.compose(self.pre))
        return composed.apply(pts)

    def _set_param(self, param):
        self.optimizable.param = param
    def _get_param(self):
        return self.optimizable.param
    param = property(_get_param, _set_param, None, 'get/set param')
