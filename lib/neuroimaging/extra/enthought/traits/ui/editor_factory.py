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
# Author: David C. Morrill
# Date: 10/07/2004
# Description: Define the abstract EditorFactory class used to represent a
#              factory for creating the Editor objects used in a traits-based
#              user interface.
#  Symbols defined: EditorFactory
#------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from neuroimaging.extra.enthought.traits \
    import HasPrivateTraits, Callable, Str, true, false

#-------------------------------------------------------------------------------
#  'EditorFactory' abstract base class:
#-------------------------------------------------------------------------------

class EditorFactory ( HasPrivateTraits ):
    
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------
    
    # Function to use for formatting:
    format_func = Callable
    
    # Format string to use for formatting (used if 'format_func' is not set):
    format_str = Str
    
    # Is the editor being used to create table grid cells?
    is_grid_cell = false
    
    # Are created editors initially enabled?
    enabled = true

    #---------------------------------------------------------------------------
    #  Initializes the object:
    #---------------------------------------------------------------------------
    
    def __init__ ( self, *args, **traits ):
        """ Initializes the object.
        """
        HasPrivateTraits.__init__( self, **traits )
        self.init( *args )
        
    #---------------------------------------------------------------------------
    #  Performs any initialization needed after all constructor traits have 
    #  been set:
    #---------------------------------------------------------------------------
     
    def init ( self ):
        """ Performs any initialization needed after all constructor traits 
            have been set.
        """
        pass
    
    #---------------------------------------------------------------------------
    #  'Editor' factory methods:
    #---------------------------------------------------------------------------
    
    def simple_editor ( self, ui, object, trait_name, description, parent ):
        raise NotImplementedError
    
    def custom_editor ( self, ui, object, trait_name, description, parent ):
        raise NotImplementedError
    
    def text_editor ( self, ui, object, trait_name, description, parent ):
        raise NotImplementedError
    
    def readonly_editor ( self, ui, object, trait_name, description, parent ):
        raise NotImplementedError

