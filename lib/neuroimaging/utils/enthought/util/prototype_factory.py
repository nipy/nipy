#------------------------------------------------------------------------------
# Copyright (c) 2005, Enthought, Inc.
# All rights reserved.
# 
# This software is provided without warranty under the terms of the BSD
# license included in enthought/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
# Thanks for using Enthought open source!
# 
# Author: Enthought, Inc.
# Description: <Enthought util package component>
#------------------------------------------------------------------------------
""" A generic prototype factory. """


# Enthought library imports.
from neuroimaging.utils.enthought.traits import AnyValue

# Local imports.
from factory import Factory


class PrototypeFactory(Factory):
    """ A generic prototype factory. """

    __traits__ = {
        # The object that we will create clones of.
        'prototype' : AnyValue,
    }
    
    ###########################################################################
    # object interface.
    ###########################################################################

    def __init__(self, prototype):
        """ Creates a new factory for clones of the specified prototype. """

        # Base-class constructor.
        #
        # fixme: Do we really need the factory to contain the class?!?
        Factory.__init__(self, prototype.__class__)

        # The object that we will create clones of.
        self.prototype = prototype
        
        return

    ###########################################################################
    # 'Factory' interface.
    ###########################################################################

    def generate(self):
        """ Creates a clone of our prototype. """
        
        return self.prototype.clone()

#### EOF ######################################################################
