# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
The base Transform class.

This class defines the Transform interface and can be subclassed to
define more clever composition logic.
"""

################################################################################
class CompositionError(Exception):
    """ The Exception raised when composing transforms with non matching
        respective input and output word spaces.
    """
    pass



################################################################################
# Class `Transform`
################################################################################
class Transform(object):
    """
    A transform is a representation of a transformation from one 3D space to
    another. It is composed of a coordinate mapping, or its inverse, as well 
    as the name of the input and output spaces.

    The Transform class is the base class for transformations and defines 
    the transform object API.
    """

    # The name of the input space
    input_space     = ''

    # The name of the output space
    output_space    = ''

    # The coordinate mapping from input space to output space
    mapping = None
    
    # The inverse coordinate mapping from output space to input space
    inverse_mapping = None

    def __init__(self, input_space, output_space, mapping=None, 
                       inverse_mapping=None):
        """ Create a new transform object.

            Parameters
            ----------

            mapping: callable f(x, y, z)
                Callable mapping coordinates from the input space to
                the output space. It should take 3 numbers or arrays, 
                and return 3 numbers or arrays of the same shape.
            inverse_mapping: callable f(x, y, z)
                Callable mapping coordinates from the output space to
                the input space. It should take 3 numbers or arrays, 
                and return 3 numbers or arrays of the same shape.
            input_space: string
                Name of the input space
            output_space: string
                Name of the output space

            Notes
            ------

            You need to supply either the mapping or the inverse mapping.
        """
        if inverse_mapping is None and mapping is None:
            raise ValueError(
                    'You need to supply either the coordinate mapping or '
                    'the inverse coordinate mapping'
                )
        if mapping is not None:
            assert callable(mapping), \
                'The mapping argument of a Transform must be callable'
        if inverse_mapping is not None:
            assert callable(inverse_mapping), \
                'The inverse_mapping argument of a Transform must be callable'
        self.mapping         = mapping
        self.inverse_mapping = inverse_mapping
        self.input_space     = input_space
        self.output_space    = output_space

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
        self._check_composition(transform)
        # We don't want to keep references on the transforms, in the
        # closure of the new mapping so we extract their mapping
        # outside of the definition of the new mapping
        first_mapping  = self.mapping
        second_mapping = transform.mapping
        if first_mapping is not None and second_mapping is not None:
            def new_mapping(x, y, z):
                """ Coordinate mapping from %s to %s.
                """ % (self.input_space, transform.output_space)
                return second_mapping(*first_mapping(x, y, z))
        else:
            new_mapping = None
        
        first_inverse_mapping  = self.inverse_mapping
        second_inverse_mapping = transform.inverse_mapping
        if ( first_inverse_mapping is not None 
             and second_inverse_mapping is not None):
            def new_inverse_mapping(x, y, z):
                """ Coordinate mapping from %s to %s.
                """ % (transform.output_space, self.input_space)
                return first_inverse_mapping(*second_inverse_mapping(x, y, z))
        else:
            new_inverse_mapping = None
 
        if new_mapping is None and new_inverse_mapping is None:
            raise CompositionError(
                """Composing two transforms with no chainable mapping:
                %s
                and
                %s"""
                % (self, transform)
                )

        return Transform(self.input_space, 
                         transform.output_space,
                         mapping=new_mapping,
                         inverse_mapping=new_inverse_mapping,
                         )


    def get_inverse(self):
        """ Return the inverse transform.
        """
        return self.__class__(
                                input_space     = self.output_space,
                                output_space    = self.input_space,
                                mapping         = self.inverse_mapping,
                                inverse_mapping = self.mapping,
                             )

    #-------------------------------------------------------------------------
    # Private methods
    #-------------------------------------------------------------------------
    def _check_composition(self, transform):
        """ Check that the given transform can be composed with this
            one.
        """
        if not transform.input_space == self.output_space:
            raise CompositionError("The input space of the "
                "second transform (%s) does not match the input space "
                "of first transform (%s)" % 
                    (transform.input_space, self.output_space)
                )


    def __repr__(self):
        representation = \
                '%s(\n  input_space=%s,\n  output_space=%s,\n  mapping=%s,\n  inverse_mapping=%s)' % (
                self.__class__.__name__,
                                    self.input_space,
                                    self.output_space, 
                '\n         '.join(repr(self.mapping).split('\n')),
                '\n         '.join(repr(self.inverse_mapping).split('\n')),
                )
        return representation


    def __copy__(self):
        """ Copy the transform 
        """
        return self.__class__(input_space=self.input_space,
                              output_space=self.output_space,
                              mapping=self.mapping, 
                              inverse_mapping=self.inverse_mapping)


    def __eq__(self, other):
        return (isinstance(other, self.__class__)
                and self.input_space == other.input_space
                and self.output_space == other.output_space
                and self.mapping == other.mapping
                and self.inverse_mapping == other.inverse_mapping
                )



