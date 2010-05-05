# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
The AffineTransform class
"""

import numpy as np

from .transform import Transform
from .affine_utils import apply_affine

################################################################################
# Class `AffineTransform`
################################################################################
class AffineTransform(Transform):
    """
    A transformation from an input 3D space to an output 3D space defined
    by an affine matrix.

    It is defined by the affine matrix , and the name of the input and output 
    spaces.
    """

    # The coordinate mapping from input space to output space
    affine = None

    def __init__(self, input_space, output_space, affine):
        """ Create a new affine transform object.

            Parameters
            ----------

            input_space: string
                Name of the input space
            output_space: string
                Name of the output space
            affine: 4x4 ndarray 
                Affine matrix giving the coordinate mapping between the 
                input and output space. 
        """
        assert hasattr(affine, '__array__'), \
            'affine argument should be an array-like'
        self.affine       = affine 
        self.input_space  = input_space
        self.output_space = output_space

    #-------------------------------------------------------------------------
    # Transform Interface
    #-------------------------------------------------------------------------

    def composed_with(self, transform):
        """ Returns a new transform obtained by composing this transform
            with the one provided.

            Parameters
            -----------
            transform: nipy.core.transforms.transform object
                The transform to compose with.
        """
        if not isinstance(transform, AffineTransform):
            return super(AffineTransform, self).composed_with(transform)
        self._check_composition(transform)
        new_affine = np.dot(transform.affine, self.affine)
        return AffineTransform(self.input_space, 
                               transform.output_space,
                               new_affine, 
                               )


    def get_inverse(self):
        """ Return the inverse transform.
        """
        return AffineTransform(self.output_space,
                               self.input_space,
                               np.linalg.inv(self.affine),
                               )

    def inverse_mapping(self, x, y, z):
        """ Transform the given coordinate from output space to input space.

            Parameters
            ----------
            x: number or ndarray
                The x coordinates
            y: number or ndarray
                The y coordinates
            z: number or ndarray
                The z coordinates
        """
        return apply_affine(x, y, z, np.linalg.inv(self.affine))


    def mapping(self, x, y, z):
        """ Transform the given coordinate from input space to output space.

            Parameters
            ----------
            x: number or ndarray
                The x coordinates
            y: number or ndarray
                The y coordinates
            z: number or ndarray
                The z coordinates
        """
        return apply_affine(x, y, z, self.affine)


    #---------------------------------------------------------------------------
    # Private methods
    #---------------------------------------------------------------------------

    def __repr__(self):
        representation = \
                '%s(\n  affine=%s,\n  input_space=%s,\n  output_space=%s)' % (
                self.__class__.__name__,
                '\n         '.join(repr(self.affine).split('\n')),
                self.input_space,
                self.output_space, 
                )
        return representation


    def __copy__(self):
        """ Copy the transform 
        """
        return self.__class__(affine=self.affine,
                              input_space=self.input_space,
                              output_space=self.output_space)


    def __deepcopy__(self, option):
        """ Copy the Image and the arrays and metadata it contains.
        """
        return self.__class__(affine=self.affine.copy(),
                              input_space=self.input_space,
                              output_space=self.output_space)


    def __eq__(self, other):
        return (isinstance(other, self.__class__)
                and self.input_space == other.input_space
                and self.output_space == other.output_space
                and np.allclose(self.affine, other.affine))



