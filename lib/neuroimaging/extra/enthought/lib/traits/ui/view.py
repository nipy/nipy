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
# Author: David C. Morrill
# Date: 10/07/2004
# Description: Define the View class used to represent the structural content of
#              a traits-based user interface.
#
#  Symbols defined: View
#
#------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from neuroimaging.extra.enthought.traits \
    import Trait, TraitPrefixList, TraitError, Str, Float, Bool, Instance, \
           List, Any, Callable, Event, Enum
           
from view_element \
    import ViewElement, ViewSubElement
    
from ui \
    import UI
    
from ui_traits \
    import SequenceTypes, object_trait, style_trait, dock_style_trait, \
           image_trait, export_trait, help_id_trait, buttons_trait
    
from handler \
    import Handler, default_handler
    
from group \
    import Group
    
from item \
    import Item
    
from include \
    import Include

#-------------------------------------------------------------------------------
#  Trait definitions:
#-------------------------------------------------------------------------------

# Name of the view trait:
id_trait = Str( desc = 'the name of the view' )

# Contents of the view trait (i.e. a single Group object):
content_trait = Instance( Group,
                          desc = 'the content of the view' )

# The menu bar for the view:
#menubar_trait = Instance( 'neuroimaging.extra.enthought.pyface.action.MenuBarManager',
#                          desc = 'the menu bar for the view' )

# The tool bar for the view:
#toolbar_trait = Instance( 'neuroimaging.extra.enthought.pyface.action.ToolBarManager',
#                          desc = 'the tool bar for the view' )

# An optional model/view factory for converting the model into a viewable
# 'model_view' object:
model_view_trait = Callable( desc = 'the factory function for converting a' 
                                    'model into a model/view object' )
                    
# Reference to a Handler object trait:
handler_trait = Any( desc = 'the handler for the view' )

# Dialog window title trait:
title_trait = Str( desc = 'the window title for the view' )

# Dialog window icon trait:
#icon_trait = Instance( 'neuroimaging.extra.enthought.pyface.image_resource.ImageResource',
#                     desc = 'the ImageResource of the icon file for the view' )

# User interface kind trait:
kind_trait = Trait( 'live', 
                    TraitPrefixList( 'panel', 'subpanel', 
                                     'modal', 'nonmodal',
                                     'livemodal', 'live', 'wizard' ), 
                    desc = 'the kind of view window to create',
                    cols = 4 )
           
# Optional window button traits:                    
apply_trait  = Bool( True,
                     desc = "whether to add an 'Apply' button to the view" )
                    
revert_trait = Bool( True,
                     desc = "whether to add a 'Revert' button to the view" )
                    
undo_trait   = Bool( True,
                 desc = "whether to add 'Undo' and 'Redo' buttons to the view" )
          
ok_trait     = Bool( True,
                     desc = "whether to add an 'OK' button to the view" )
          
cancel_trait = Bool( True,
                     desc = "whether to add a 'Cancel' button to the view" )
          
help_trait   = Bool( True,
                     desc = "whether to add a 'Help' button to the view" )
                     
on_apply_trait = Callable( desc = 'the routine to call when modal changes are '
                                  'applied or reverted' )
                     
# Is dialog window resizable trait:
resizable_trait = Bool( False,
                        desc = 'whether dialog can be resized or not' )
                     
# Is view scrollable:
scrollable_trait = Bool( False,
                         desc = 'whether view should be scrollable or not' )

# The valid categories of imported elements that can be dragged into the view:
imports_trait = List( Str, desc = 'the categories of elements that can be '
                                  'dragged into the view' )

# The view position and size traits:                    
width_trait  = Float( -1E6,
                      desc = 'the width of the view window' )
height_trait = Float( -1E6,
                      desc = 'the height of the view window' )
x_trait      = Float( -1E6,
                      desc = 'the x coordinate of the view window' )
y_trait      = Float( -1E6,
                      desc = 'the y coordinate of the view window' )
                      
# The result that should be returned if the user clicks the window/dialog close
# button/icon:
close_result_trait = Enum( None, True, False,
                         desc = 'the result to return when the user clicks the '
                                'window or dialog close button or icon' )
                    
#-------------------------------------------------------------------------------
#  'View' class:
#-------------------------------------------------------------------------------

class View ( ViewElement ):
    """ A traits-based user interface for one or more objects.
    
    The attributes of the View object determine the contents and layout of
    an attribute-editing window. A View object contains a set of `Group`, 
    `Item`, and `Include` objects. A View object can be an attribute of an
    object derived from `HasTraits`, or a standalone object.
    """
    
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    # The name of the view:
    id = id_trait
    
    # The top-level Group object for the view:
    content = content_trait
    
    # The menu bar for the view:
    menubar = Any
    
    # The menu bar for the view:
    toolbar = Any
    
    # List of button actions to add to view:
    buttons = buttons_trait
    
    # The menu bar for the view:
#   menubar = menubar_trait

    # The tool bar for the view:
#   toolbar = toolbar_trait

    # The Handler object for handling events:
    handler = handler_trait 
    
    # The factory function for converting a model into a model/view object:
    model_view = model_view_trait
    
    # The modal/wizard dialog window title:
    title = title_trait
    
    # The name of the icon to display in the dialog window title bar:
    icon = Any
    
    # The kind of user interface to create:
    kind = kind_trait
    
    # The default object being edited:
    object = object_trait
    
    # The style of user interface to create:
    style = style_trait
    
    # The default docking style to use:
    dock = dock_style_trait
    
    # The image to display on notebook tabs:
    image = image_trait
    
    # Called when modal changes applied/reverted
    on_apply = on_apply_trait
    
    # Should an Apply button be added?  (deprecated)
    apply = apply_trait
    
    # Should a Revert button be added?  (deprecated)
    revert = revert_trait
    
    # Should Undo/Redo buttons be added?  (deprecated)
    undo = undo_trait
    
    # Should an OK button be added?  (deprecated)
    ok = ok_trait
    
    # Should a Cancel button be added?  (deprecated)
    cancel = cancel_trait
    
    # Should dialog be resizable?
    resizable = resizable_trait
    
    # Should the view be scrollable?
    scrollable = scrollable_trait
    
    # The category of exported elements:
    export = export_trait
    
    # The valid categories of imported elements:
    imports = imports_trait
    
    # Should a Help button be added?
    help = help_trait
    
    # External help context identifier:
    help_id = help_id_trait
    
    # Requested view window x coordinate:
    x = x_trait
    
    # Requested view window y coordinate:
    y = y_trait
    
    # Requested view window width:
    width = width_trait
    
    # Requested view window height:
    height = height_trait
    
    # Class of dropped objects that can be added:
    drop_class = Any
    
    # View has been updated event:
    updated = Event
    
    # What result should be returned if the user clicks the window/dialog close
    # button/icon?
    close_result = close_result_trait
    
    # View has been updated event
    #updated   = Event( Bool )  
    
    # Note: Group objects delegate their 'object' and 'style' traits to the View
        
    #---------------------------------------------------------------------------
    #  Initializes the object:
    #---------------------------------------------------------------------------

    def __init__ ( self, *values, **traits ):
        """ Initializes the object.
        """
        ViewElement.__init__( self, **traits )
        content = []
        accum   = []
        for value in values:
            if isinstance( value, ViewSubElement ):
                content.append( value )
            elif type( value ) in SequenceTypes:
                content.append( Group( *value ) )
            else:
                content.append( Item( value ) )
        
        # If 'content' trait was specified, add it to the end of the content:
        if self.content is not None:
            content.append( self.content )
            
        # If there are any 'Item' objects in the content, wrap the content in a
        # Group:
        for item in content:
            if isinstance( item, Item ):
                content = [ Group( *content ) ]
                break
                
        # Wrap all of the content up into a Group and save it as our content:
        self.content = Group( container = self, *content )

    #---------------------------------------------------------------------------
    #  Creates a UI user interface object:
    #---------------------------------------------------------------------------
    
    def ui ( self, context, parent        = None, kind       = None, 
                            view_elements = None, handler    = None,
                            id            = '',   scrollable = None,
                            args          = None ):
        """ Creates a `UI` object.
        
        Parameters
        ----------
        context : object or dictionary of objects 
            The object or objects to be edited
        parent : window component 
            The window parent of the View object's window
        kind : string
            The kind of window to create. The values in the following list are
            valid. If *kind* is unspecified or **None**, the **kind** attribute 
            of the current object is used.

                'panel'
                    An embeddable panel. This type of window is intended to
                    to be used as part of a larger interface.
                'modal'
                    A modal dialog box that operates on a clone of the object
                    until the user clicks commits the change
                'nonmodal'
                    A nonmodal dialog box that operates on a clone of the object
                    until the user clicks commits the change
                'live'
                    A nonmodal dialog box that immediately updates the object
                'livemodal'
                    A modal dialog box that immediately updates the object
                'wizard'
                    A wizard modal dialog box. A wizard contains a sequence of 
                    pages, which can be accessed by click "Next" and "Back" 
                    buttons. Changes to attribute values are only applied if the
                    user clicks the "Finish" button on the last page.
                    
        view_elements : `ViewElements` object 
            The set of Group, Item, and Include objects contained in the view
        handler : `Handler` object
            A handler for the UI object
        
        """
        handler = handler or self.handler or default_handler()
        if not isinstance( handler, Handler ):
            handler = handler()
        if args is not None:
            handler.set( **args )
        
        if not isinstance( context, dict ):
            context = context.trait_context()
            
        context.setdefault( 'handler', handler )
                        
        if self.model_view is not None:
            context[ 'object' ] = self.model_view( context[ 'object' ] )
            
        self_id = self.id
        if self_id != '':
            if id != '':
                id = '%s:%s' % ( self_id, id )
            else:
                id = self_id
                
        if scrollable is None:
            scrollable = self.scrollable
            
        ui = UI( view          = self,
                 context       = context,
                 handler       = handler,
                 view_elements = view_elements,
                 title         = self.title,
                 id            = id,
                 scrollable    = scrollable )
                 
        if kind is None:
            kind = self.kind
        
        ui.ui( parent, kind )
        
        return ui
    
    #---------------------------------------------------------------------------
    #  Replaces any items which have an 'id' with an Include object with the 
    #  same 'id', and puts the object with the 'id' into the specified 
    #  ViewElements object: 
    #---------------------------------------------------------------------------
    
    def replace_include ( self, view_elements ):
        """ Replaces any items which have an 'id' with an Include object with 
            the same 'id', and puts the object with the 'id' into the specified 
            ViewElements object.
        """
        if self.content is not None:
            self.content.replace_include( view_elements )
        
    #---------------------------------------------------------------------------
    #  Returns a 'pretty print' version of the View:
    #---------------------------------------------------------------------------
            
    def __repr__ ( self ):
        """ Returns a 'pretty print' version of the View.
        """
        if self.content is None:
            return '()'
        return "( %s )" %  ', '.join( 
               [ item.__repr__() for item in self.content.content ] )
        
