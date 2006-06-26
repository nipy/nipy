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
# Description: Define the abstract Editor class used to represent an object
#              trait editing control in a traits-based user interface.
#
#  Symbols defined: Editor 
#------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from neuroimaging.extra.enthought.traits \
    import Trait, HasPrivateTraits, ReadOnly, Any, Property, Undefined, true, \
           false, TraitError, Str
           
from editor_factory \
    import EditorFactory
    
from undo \
    import UndoItem

#-------------------------------------------------------------------------------
#  Trait definitions:
#-------------------------------------------------------------------------------

# Reference to an EditorFactory object:
factory_trait = Trait( EditorFactory )

#-------------------------------------------------------------------------------
#  'Editor' abstract base class:
#-------------------------------------------------------------------------------

class Editor ( HasPrivateTraits ):
    
    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------
    
    # The UI (user interface) this editor is part of:
    ui = ReadOnly
    
    # The object this editor is editing:
    object = ReadOnly
    
    # The name of the trait this editor is editing:
    name = ReadOnly
    
    # Original value of object.name:
    old_value = ReadOnly
    
    # Text description of the object trait being edited:
    description = ReadOnly
    
    # The Item object used to create this editor:
    item = ReadOnly
    
    # Context name of object editor is editing:
    object_name = Str( 'object' )
    
    # The GUI widget defined by this editor:
    control = Any
    
    # The GUI label (if any) defined by this editor:
    label_control = Any
    
    # Is the underlying GUI widget enabled?
    enabled = true
    
    # Is the underlying GUI widget visible?
    visible = true
    
    # Is the underlying GUI widget scrollable?
    scrollable = false
    
    # The EditorFactory used to create this editor:
    factory = factory_trait
    
    # Is the editor updating the object.name value?
    updating = false
    
    # Current value for object.name:
    value = Property
    
    # Current value of object trait as a string:
    str_value = Property
    
    #---------------------------------------------------------------------------
    #  Initializes the object:
    #---------------------------------------------------------------------------
    
    def __init__ ( self, parent, **traits ):
        """ Initializes the object.
        """
        HasPrivateTraits.__init__( self, **traits )
        try:
            self.old_value = getattr( self.object, self.name )
        except AttributeError:
            # Getting the attribute will fail for 'Event' traits:
            self.old_value = Undefined
            
    #---------------------------------------------------------------------------
    #  Finishes editor set-up:  
    #---------------------------------------------------------------------------
                        
    def prepare ( self, parent ):
        """ Finishes editor set-up.
        """
        self.object.on_trait_change( self._update_editor, self.name )
        self.init( parent )
        self.update_editor()
        
    #---------------------------------------------------------------------------
    #  Finishes initializing the editor by creating the underlying toolkit
    #  widget:
    #---------------------------------------------------------------------------
        
    def init ( self, parent ):
        """ Finishes initializing the editor by creating the underlying toolkit
            widget.
        """
        raise NotImplementedError
        
    #---------------------------------------------------------------------------
    #  Disposes of the contents of an editor:    
    #---------------------------------------------------------------------------
                
    def dispose ( self ):
        """ Disposes of the contents of an editor.
        """
        self.object.on_trait_change( self._update_editor, self.name, 
                                     remove = True )
        if self._user_from is not None:
            for name, value in self._user_from.items():
                user_object, user_name, editor_name, is_list = value
                user_object.on_trait_change( self._user_trait_modified,
                                             user_name, remove = True )
                if is_list:
                    user_object.on_trait_change( self._user_list_modified,
                                           user_name + '_items', remove = True )
                    
        if self._user_to is not None:
            for name, value in self._user_to.items():
                user_object, user_name, editor_name, is_list = value
                self.on_trait_change( self._editor_trait_modified,
                                      editor_name, remove = True )
                if is_list:
                    self.on_trait_change( self._editor_list_modified,
                                      editor_name + '_items', remove = True )
       
    #---------------------------------------------------------------------------
    #  Gets/Sets the associated object trait's value:
    #---------------------------------------------------------------------------
    
    def _get_value ( self ):
        return getattr( self.object, self.name )
        
    def _set_value ( self, value ):
        self.ui.do_undoable( self.__set_value, value )
    
    def __set_value ( self, value ):  
        self._no_update = True
        try:
            try:
                handler  = self.ui.handler
                obj_name = self.object_name
                method   = (getattr( handler, '%s_%s_setattr' % ( obj_name, 
                                              self.name ), None ) or 
                            getattr( handler, '%s_setattr' % obj_name, None ) or
                            getattr( handler, 'setattr' ))
                method( self.ui.info, self.object, self.name, value )
            except TraitError, excp:
                self.error( excp )
                raise
        finally:
            self._no_update = False
        
    #---------------------------------------------------------------------------
    #  Returns the text representation of a specified object trait value:
    #---------------------------------------------------------------------------
  
    def string_value ( self, value ):
        """ Returns the text representation of a specified object trait value.

            If format_func is set on the factory then we call that function to 
            do the formatting.  If format_str is set on the factory then we use 
            that string for formatting. Otherwise we call str.
        """
        factory = self.factory
        if factory.format_func is not None:
            return factory.format_func( value )
            
        if factory.format_str != '':
            return factory.format_str % value
            
        return str( value )
        
    #---------------------------------------------------------------------------
    #  Returns the text representation of the object trait:
    #---------------------------------------------------------------------------
  
    def _get_str_value ( self ):
        """ Returns the text representation of the object trait.
        """
        return self.string_value( getattr( self.object, self.name ) )
  
    #---------------------------------------------------------------------------
    #  Returns the text representation of a specified value:
    #---------------------------------------------------------------------------
  
    def _str ( self, value ):
        """ Returns the text representation of a specified value.
        """
        return str( value )
        
    #---------------------------------------------------------------------------
    #  Handles an error that occurs while setting the object's trait value:
    #
    #  (Should normally be overridden in a subclass)
    #---------------------------------------------------------------------------
        
    def error ( self, excp ):
        """ Handles an error that occurs while setting the object's trait value.
        """
        pass
        
    #---------------------------------------------------------------------------
    #  Performs updates when the object trait changes:
    #---------------------------------------------------------------------------
        
    def _update_editor ( self, object, name, old_value, new_value ):
        """ Performs updates when the object trait changes.
        """
        # Exit immediately if the change was caused by the editor itself:
        if self._no_update:
            return
            
        # If the editor has gone away for some reason, disconnect and exit:
        if self.control is None:
            object.on_trait_change( self._update_editor, name, remove = True )
            return
            
        # Log the change that was made (as long as it is not for an event):
        if object.base_trait( name ).type != 'event':
            self.log_change( self.get_undo_item, object, name, 
                                                 old_value, new_value )
                    
        # Update the editor control to reflect the current object state:                    
        self.update_editor()
        
    #---------------------------------------------------------------------------
    #  Logs a change made in the editor:    
    #---------------------------------------------------------------------------
                
    def log_change ( self, undo_factory, *undo_args ):
        """ Logs a change made in the editor.
        """
        # Indicate that the contents of the user interface have been changed:
        ui          = self.ui
        ui.modified = True
        
        # Create an undo history entry if we are maintaining a history:
        undoable = ui._undoable
        if undoable >= 0:
            history = ui.history
            if history is not None:
                item = undo_factory( *undo_args )
                if item is not None:
                    if undoable == history.now: 
                        # Create a new undo transaction:
                        history.add( item )
                    else:
                        # Extend the most recent undo transaction:
                        history.extend( item )
        
    #---------------------------------------------------------------------------
    #  Updates the editor when the object trait changes external to the editor:
    #
    #  (Should normally be overridden in a subclass)
    #---------------------------------------------------------------------------
        
    def update_editor ( self ):
        """ Updates the editor when the object trait changes external to the 
            editor.
        """
        pass
        
    #---------------------------------------------------------------------------
    #  Creates an undo history entry:   
    #
    #  (Can be overridden in a subclass for special value types)
    #---------------------------------------------------------------------------
           
    def get_undo_item ( self, object, name, old_value, new_value ):
        """ Creates an undo history entry.
        """
        return UndoItem( object    = object,
                         name      = name,
                         old_value = old_value,
                         new_value = new_value ) 
                         
    #---------------------------------------------------------------------------
    #  Sets/Unsets synchronization between an editor trait and a user object 
    #  trait:  
    #---------------------------------------------------------------------------
    
    def sync_value ( self, user_name, editor_name, 
                           mode = 'both', is_list = False, remove = False ):
        """ Sets/Unsets synchronization between an editor trait and a user 
            object trait.
        """
        if user_name != '':
            object_name = 'object'
            col         = user_name.find( '.' )
            if col >= 0:
                object_name = user_name[ : col ]
                user_name   = user_name[ col + 1: ]
            user_object = self.ui.context[ object_name ]
            value       = ( user_object, user_name, editor_name, is_list )
            
            if mode in ( 'from', 'both' ):
                user_object.on_trait_change( self._user_trait_modified, 
                                             user_name )
                if is_list:
                    user_object.on_trait_change( self._user_list_modified,
                                                 user_name + '_items' )
                if self._user_to is None:
                    self._user_to = {}
                self._user_to[ user_name ] = value
                if mode == 'from':
                    setattr( self, editor_name,
                             getattr( user_object, user_name ) )
                             
            if mode in ( 'to', 'both' ):
                self.on_trait_change( self._editor_trait_modified, editor_name )
                if is_list:
                    self.on_trait_change( self._editor_list_modified,
                                          editor_name + '_items' )
                if self._user_from is None:
                    self._user_from = {}
                self._user_from[ editor_name ] = value
                if mode == 'to':
                    setattr( user_object, user_name,
                             getattr( self, editor_name ) )
                
    def _user_trait_modified ( self, object, name, old, new ):
        if not self._no_update:
            user_object, user_name, editor_name, is_list = self._user_to[ name ]
            self._no_update = True
            try:
                setattr( self, editor_name, new )
            finally:
                self._no_update = False
                
    def _user_list_modified ( self, object, name, old, event ):
        if not self._no_update:
            user_object, user_name, editor_name, is_list = self._user_to[ name ]
            n = event.index
            self._no_update = True
            try:
                getattr( self, editor_name )[ n: n + len( event.removed ) ] = \
                                                                    event.added
            finally:
                self._no_update = False
                
    def _editor_trait_modified ( self, object, name, old, new ):
        if not self._no_update:
            user_object, user_name, editor_name, is_list = self._user_from[name]
            self._no_update = True
            try:
                setattr( user_object, user_name, new )
            finally:
                self._no_update = False
                
    def _editor_list_modified ( self, object, name, old, event ):
        if not self._no_update:
            user_object, user_name, editor_name, is_list = self._user_from[name]
            n = event.index
            self._no_update = True
            try:
                getattr( user_object, user_name )[ 
                                     n: n + len( event.removed )] = event.added
            finally:                                                                    
                self._no_update = False                                                                    
        
#-- UI preference save/restore interface ---------------------------------------

    #---------------------------------------------------------------------------
    #  Restores any saved user preference information associated with the 
    #  editor:
    #---------------------------------------------------------------------------
            
    def restore_prefs ( self, prefs ):
        """ Restores any saved user preference information associated with the 
            editor.
        """
        pass
            
    #---------------------------------------------------------------------------
    #  Returns any user preference information associated with the editor:
    #---------------------------------------------------------------------------
            
    def save_prefs ( self ):
        """ Returns any user preference information associated with the editor.
        """
        return None
        
#-- End UI preference save/restore interface -----------------------------------                         

