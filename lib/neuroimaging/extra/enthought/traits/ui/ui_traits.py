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
# Date: 10/14/2004
# Description: Definition of common traits used within the traits.ui package.
#             
# Symbols defined:
#
#------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from neuroimaging.extra.enthought.traits \
    import Trait, TraitPrefixList, Delegate, Str, Instance, List, Enum, Any

#-------------------------------------------------------------------------------
#  Trait definitions:
#-------------------------------------------------------------------------------

# User interface element styles:
style_trait = Trait( 'simple',
                     TraitPrefixList( 'simple', 'custom', 'text', 'readonly' ),
                     cols = 4 )
                     
# The default object being edited trait:                     
object_trait = Str( 'object' )                     

# The default dock style to use:
dock_style_trait = Enum( 'fixed', 'horizontal', 'vertical', 'tab',
                         desc = "the default docking style to use" )
                         
# The default notebook tab image to use:                         
image_trait = Instance( 'neuroimaging.extra.enthought.pyface.image_resource.ImageResource',
                        desc = 'the image to be displayed on notebook tabs' )
                     
# The category of elements dragged out of the view:
export_trait = Str( desc = 'the category of elements dragged out of the view' )

# Delegate a trait value to the object's 'container':                      
container_delegate = Delegate( 'container' )

# An external help context identifier:
help_id_trait = Str( desc = "the external help context identifier" )                     

# The set of buttons to add to the view:   
a_button = Trait( '', Str, Instance( 'neuroimaging.extra.enthought.traits.ui.menu.Action' ) )
buttons_trait = List( a_button,
                      desc = 'the action buttons to add to the bottom of '
                             'the view' )

# View trait specified by name or instance:
AView = Any
#AView = Trait( '', Str, Instance( 'neuroimaging.extra.enthought.traits.ui.View' ) )

#-------------------------------------------------------------------------------
#  Other definitions:
#-------------------------------------------------------------------------------

SequenceTypes = ( tuple, list )
