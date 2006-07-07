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
# Date: 12/19/2004
# Description: Defines the standard menu bar for use with Traits UI windows and
#              panels.
#------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from enthought.traits \
    import Str

# Import and rename the needed PyFace elements:
from enthought.pyface.action \
    import ToolBarManager as ToolBar
    
from enthought.pyface.action \
    import MenuBarManager as MenuBar
    
from enthought.pyface.action \
    import MenuManager as Menu
    
from enthought.pyface.action \
    import Group as ActionGroup
    
from enthought.pyface.action \
    import Action as PyFaceAction

#-------------------------------------------------------------------------------
#  'Action' class (extends the core pyface Action class):  
#-------------------------------------------------------------------------------

class Action ( PyFaceAction ):
    
    #---------------------------------------------------------------------------
    #  Trait definitions:  
    #---------------------------------------------------------------------------
        
    # Value of the expression determines when item is visible:
    visible_when = Str
        
    # Value of the expression determines when item is enabled:
    enabled_when = Str
    
    # Value of the expression determines when item is checked:
    checked_when = Str
    
    # Value of the expression determines if the item should be defined:
    defined_when = Str

    # The method to call when the action is performed:
    action = Str

#-------------------------------------------------------------------------------
#  Standard actions and menu bar definitions:
#-------------------------------------------------------------------------------

# Menu separator:
Separator = ActionGroup

# The standard 'close window' action:
CloseAction = Action(
    name   = 'Close',
    action = '_on_close'
)

# The standard 'undo last change' action:
UndoAction = Action(
    name         = 'Undo',
    action       = '_on_undo',
    defined_when = 'ui.history is not None',
    enabled_when = 'ui.history.can_undo'
)

# The standard 'redo last undo' action:
RedoAction = Action(
    name         = 'Redo',
    action       = '_on_redo',
    defined_when = 'ui.history is not None',
    enabled_when = 'ui.history.can_redo'
)

# The standard 'Revert all changes' action:
RevertAction = Action(
    name         = 'Revert',
    action       = '_on_revert',
    defined_when = 'ui.history is not None',
    enabled_when = 'ui.history.can_undo'
)

# The standard 'Show help' action:
HelpAction = Action(
    name   = 'Help',
    action = 'show_help'
)

# The standard Trait's UI menu bar:
StandardMenuBar = MenuBar(
    Menu( CloseAction,
          name = 'File' ),
    Menu( UndoAction,
          RedoAction,
          RevertAction,
          name = 'Edit' ),
    Menu( HelpAction,
          name = 'Help' )
)

#-------------------------------------------------------------------------------
#  Standard buttons (i.e. actions):  
#-------------------------------------------------------------------------------

NoButton     = Action( name = '' )
UndoButton   = Action( name = 'Undo' )
RevertButton = Action( name = 'Revert' )
ApplyButton  = Action( name = 'Apply' )
OKButton     = Action( name = 'OK' )
CancelButton = Action( name = 'Cancel' )
HelpButton   = Action( name = 'Help' )

OKCancelButtons = [ OKButton, CancelButton ]
ModalButtons = [ ApplyButton, RevertButton, OKButton, CancelButton, HelpButton ]
LiveButtons  = [ UndoButton,  RevertButton, OKButton, CancelButton, HelpButton ]
NoButtons    = [ NoButton ]


