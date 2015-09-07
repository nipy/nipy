""" Generic transform class

This implementation specifies an API.  We've done our best to avoid checking
instances, so any class implementing this API should be valid in the places
(like registration routines) that use transforms.  If that isn't true, it's a
bug.
"""

class Transform(object):
    """ A default transformation class

    This class specifies the tiny API.  That is, the class should implement:

    * obj.param - the transformation exposed as a set of parameters. Changing
      param should change the transformation
    * obj.apply(pts) - accepts (N,3) array-like of points in 3 dimensions,
      returns an (N, 3) array of transformed points
    * obj.compose(xform) - accepts another object implementing ``apply``, and
      returns a new transformation object, where the resulting transformation is
      the composition of the ``obj`` transform onto the ``xform`` transform.
    """
    def __init__(self, func):
        self.func = func

    def apply(self, pts):
        return self.func(pts)

    def compose(self, other):
        return Transform(
            lambda pts : self.apply(other.apply(pts)))

    @property
    def param(self):
        raise AttributeError('No param for generic transform')
