""" Chain transforms """

from .affine import Affine


class ChainTransform(object):
    def __init__(self, optimizable, pre=None, post=None):
        """ Create chain transform instance

        Parameters
        ----------
        optimizable : Transform
            Transform that we are optimizing
        pre : None or Transform, optional
            If not None, a transform that should be applied to points before
            applying the `optimizable` transform
        post : None or Transform, optional
            If not None, a transform that should be applied to points after
            applying any `pre` transform, and then the `optimizable`
            transform.
        """
        self.optimizable = optimizable
        if pre is None:
            pre = Affine()
        if post is None:
            post = Affine()
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
