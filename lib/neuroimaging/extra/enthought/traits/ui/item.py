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
# Description: Define the Item class used to represent a single item within a
#              traits-based user interface.
#  Symbols defined: Item
#------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

import re

from string \
    import find, rfind
    
from neuroimaging.extra.enthought.traits \
    import Instance, Str, Int, Range, false, Callable, Delegate
    
from neuroimaging.extra.enthought.traits.trait_base \
    import user_name_for
    
from view_element \
    import ViewSubElement
    
from ui_traits \
    import container_delegate
    
from editor_factory \
    import EditorFactory

#-------------------------------------------------------------------------------
#  Constants:
#-------------------------------------------------------------------------------

# Pattern of all digits:    
all_digits = re.compile( r'\d+' )

# Pattern for finding size infomation imbedded in an item description:
size_pat = re.compile( r"^(.*)<(.*)>(.*)$", re.MULTILINE | re.DOTALL )

# Pattern for finding tooltip infomation imbedded in an item description:
tooltip_pat = re.compile( r"^(.*)`(.*)`(.*)$", re.MULTILINE | re.DOTALL )

#-------------------------------------------------------------------------------
#  Trait definitions:
#-------------------------------------------------------------------------------

# EditorFactory reference trait:
ItemEditor = Instance( EditorFactory, allow_none = True )

# Amount of padding to add around item:
Padding = Range( -15, 15, 0, desc = 'amount of padding to add around item' )

#-------------------------------------------------------------------------------
#  'Item' class:
#-------------------------------------------------------------------------------

class Item ( ViewSubElement ):
    """ An element in a trait-based user interface.
    """
    
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------
   
    # Name of the item:
    id = Str
    
    # User interface label for the item:
    label = Str
    
    # Name of the trait the item is editing:
    name = Str
    
    # Help text describing purpose of item:
    help = Str
    
    # Object the item is editing:
    object = container_delegate
    
    # Presentation style for the item:
    style = container_delegate
    
    # Docking style for the item:
    dock = container_delegate
    
    # Image to display on notebook tabs:
    image = container_delegate
    
    # Category of elements dragged from view:
    export = container_delegate
    
    # Is label added?
    show_label = Delegate( 'container', 'show_labels' )
    
    # Editor to use for the item:
    editor = ItemEditor
    
    # Should the item use extra space?
    resizable = false
    
    # Should the item use extra space along its Group's layout orientation?
    springy = false
    
    # Should the item be emphasized?
    emphasized = false
    
    # Should the item receive focus initially?
    has_focus = false
    
    # Pre-condition for defining the item:
    defined_when = Str
    
    # Pre-condition for showing the item:
    visible_when = Str
    
    # Pre-condition for enabling the item:
    enabled_when = Str
    
    # Amount of padding to add around item:
    padding = Padding
    
    # Tooltip to display over item:
    tooltip = Str
    
    # Function to use for formatting:
    format_func = Callable
    
    # Format string to use for formatting:
    format_str = Str
    
    # Requested width  of editor (in pixels):
    width = Int( -1 )
    
    # Requested height of editor (in pixels):
    height = Int( -1 ) 
   
    #---------------------------------------------------------------------------
    #  Initialize the object:
    #---------------------------------------------------------------------------
   
    def __init__ ( self, value = None, **traits ):
        super( ViewSubElement, self ).__init__( **traits )
        if value is None:
            return
        if not type( value ) is str:
            raise TypeError, ("The argument to Item must be a string of the "
                 "form: {id:}{object.}{name}{[label]}`tooltip`{#^}{$|@|*|~|;style}")
        value, empty = self._parse_label( value )
        if empty:
            self.show_label = False
        value = self._parse_style( value )
        value = self._parse_size(  value )
        value = self._parse_tooltip( value )
        value = self._option( value, '#',  'resizable',  True )
        value = self._option( value, '^',  'emphasized', True )
        value = self._split( 'id',     value, ':', find,  0, 1 )
        value = self._split( 'object', value, '.', find,  0, 1 )
        if value != '':
            self.name = value
            
    #---------------------------------------------------------------------------
    #  Returns whether or not the object is replacable by an Include object:
    #---------------------------------------------------------------------------
            
    def is_includable ( self ):
        """ Returns a boolean indicating whether the object is replacable by an 
        Include object.
        """
        return (self.id != '')
        
    #---------------------------------------------------------------------------
    #  Returns whether or not the Item represents a spacer or separator:
    #---------------------------------------------------------------------------
        
    def is_spacer ( self ):
        name = self.name.strip()
        return ((name == '') or (name == '_') or 
                (all_digits.match( name ) is not None))
        
    #---------------------------------------------------------------------------
    #  Gets the help text associated with the Item in a specified UI:
    #---------------------------------------------------------------------------
        
    def get_help ( self, ui ):
        """ Gets the help text associated with the Item in a specified UI.
        """
        # Return 'None' if the Item is a separator or spacer:
        if self.is_spacer():
            return None
           
        # Otherwise, it must be a trait Item:
        if self.help != '':
            return self.help
        return ui.context[ self.object ].base_trait( self.name ).get_help()

    #---------------------------------------------------------------------------
    #  Gets the label to use for a specified Item in a specified UI:
    #---------------------------------------------------------------------------
        
    def get_label ( self, ui ):
        """ Gets the label to use for a specified Item.
        """
        # Return 'None' if the Item is a separator or spacer:
        if self.is_spacer():
            return None
            
        label = self.label
        if label != '':
            return label
            
        name   = self.name
        object = ui.context[ self.object ]
        trait  = object.base_trait( name )
        label  = user_name_for( name )
        tlabel = trait.label
        if tlabel is None:
            return label
        if type( tlabel ) is str:
            if tlabel[0:3] == '...':
                return label + tlabel[3:]
            if tlabel[-3:] == '...':
                return tlabel[:-3] + label
            if self.label != '':
                return self.label
            return tlabel
        return tlabel( object, name, label )
        
    #---------------------------------------------------------------------------
    #  Returns an id used to identify the item:  
    #---------------------------------------------------------------------------
                
    def get_id ( self ):
        """ Returns an id used to identify the item.
        """
        if self.id != '':
            return self.id
            
        return self.name
        
    #---------------------------------------------------------------------------
    #  Parses a '<width,height>' value from the string definition:
    #---------------------------------------------------------------------------
        
    def _parse_size ( self, value ):
        """ Parses a '<width,height>' value from the string definition.
        """
        match = size_pat.match( value )
        if match is not None:
            data  = match.group( 2 )
            value = match.group( 1 ) + match.group( 3 )
            col   = data.find( ',' )
            if col < 0:
                self._set_int( 'width', data ) 
            else:
                self._set_int( 'width',  data[ : col ] )
                self._set_int( 'height', data[ col + 1: ] )
        return value
        
    #---------------------------------------------------------------------------
    #  Parses a '`tooltip`' value from the string definition:
    #---------------------------------------------------------------------------
        
    def _parse_tooltip ( self, value ):
        """ Parses a '`tooltip`' value from the string definition.
        """
        match = tooltip_pat.match( value )
        if match is not None:
            self.tooltip = match.group( 2 )
            value        = match.group( 1 ) + match.group( 3 )
        return value
        
    #---------------------------------------------------------------------------
    #  Sets a specified trait to a specified string converted to an integer:
    #---------------------------------------------------------------------------
                
    def _set_int ( self, name, value ):
        """ Sets a specified trait to a specified string converted to an
            integer.
        """
        value = value.strip()
        if value != '':
            setattr( self, name, int( value ) )
            
    #---------------------------------------------------------------------------
    #  Returns a 'pretty print' version of the Item:
    #---------------------------------------------------------------------------
            
    def __repr__ ( self ):
        """ Returns a 'pretty print' version of the Item.
        """
        return '"%s%s%s%s%s"' % ( self._repr_value( self.id, '', ':' ), 
                                  self._repr_value( self.object, '', '.', 
                                                    'object' ), 
                                  self._repr_value( self.name ),
                                  self._repr_value( self.label,'=' ),
                                  self._repr_value( self.style, ';', '', 
                                                    'simple' ) )

#-------------------------------------------------------------------------------
#  'Label' class:
#-------------------------------------------------------------------------------

class Label ( Item ):

    #---------------------------------------------------------------------------
    #  Initializes the object:
    #---------------------------------------------------------------------------

    def __init__ ( self, label ):
        super( Label, self ).__init__( label = label )
        
