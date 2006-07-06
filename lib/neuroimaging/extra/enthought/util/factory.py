#------------------------------------------------------------------------------
# Copyright (c) 2005, Enthought, Inc.
# All rights reserved.
# 
# This software is provided without warranty under the terms of the BSD
# license included in enthought/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.neuroimaging.extra.enthought.com/licenses/BSD.txt
# Thanks for using Enthought open source!
# 
# Author: Enthought, Inc.
# Description: <Enthought util package component>
#------------------------------------------------------------------------------
""" A generic instance factory. """


# Enthought library imports.
from neuroimaging.extra.enthought.traits import Any, Dict, HasTraits, List


class _Factory(HasTraits):
    """ A generic instance.

    A factory is a factory that creates instances of a specified class.
    
    """

    # The class that we are a factory for.
    klass = Any
        
    # Any non-keyword arguments to pass to our klass constructor.
    args = List()

    # Any keyword arguments to pass to our klass constructor.
    kw = Dict()

    
    ###########################################################################
    # object interface.
    ###########################################################################

    def __call__(self, *args, **kw):
        """ Creates an instance of the 'klass'.

        This allows the caller to create an instance using a different set
        of arguments to those passed into the factory constructor, and in
        turn makes the factory behave like a normal class (which is simply
        a factory after all).

        """

        return self._create_instance(self.klass, args, kw)

    ###########################################################################
    # 'Factory' interface.
    ###########################################################################

    def generate(self):
        """ Creates an instance of this 'klass'. """
        
        return self._create_instance(self.klass, self._get_args(),
                                     self._get_kw())
    
    ###########################################################################
    # Protected interface.
    ###########################################################################
    def _get_args(self):
        return self.args

    def _get_kw(self):
        return self.kw

    ###########################################################################
    # Private interface.
    ###########################################################################

    def _create_instance(self, klass, args, kw):
        """ Creates an instance of 'klass'. """

        return klass(*args, **kw)

# make clones of arguments. this is a temporary fix for Converge ticket #676.
# fixme: this is a bit hacky. the correct thing to do is to
#        figure out an intelligent way to intelligently indicate when we
#        should be cloning these arguments, but for now we're going to do it
#        any time we can.
class Factory(_Factory):

    def _get_args(self):

        args = []
        for arg in self.args:
            if hasattr(arg, 'clone'):
                args.append(arg.clone())
            else:
                args.append(arg)

        return args

    def _get_kw(self):
        kw = {}

        for key, value in self.kw.items():
            if hasattr(value, 'clone'):
                kw[key] = value.clone()
            else:
                kw[key] = value
        
        return kw
        
#### EOF ######################################################################
