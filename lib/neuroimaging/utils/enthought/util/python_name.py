#-----------------------------------------------------------------------------
#
#  Copyright (c) 2006 by Enthought, Inc.
#  All rights reserved.
#
#  Author: Dave Peterson <dpeterson@neuroimaging.utils.enthought.com>
#
#-----------------------------------------------------------------------------

""" Provides the capability to format a string to a valid python name.
"""

# Standard library imports.
import keyword

# Major packages.

# Enthought library imports

# Application specific imports.

# Local imports.


def python_name(name):
    """ Attempt to make a valid Python identifier out of a name.
    """

    if len(name) > 0:
        # Replace spaces with underscores.
        name = name.replace(' ', '_').lower()

        # If the name is a Python keyword then prefix it with an
        # underscore.
        if keyword.iskeyword(name):
            name = '_' + name

        # If the name starts with a digit then prefix it with an
        # underscore.
        if name[0].isdigit():
            name = '_' + name

    return name


#### EOF #####################################################################

