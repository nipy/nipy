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
# Date: 07/01/2005
# Description: Define the filter object used to filter items displayed in a
#              table editor.
#
#  Symbols defined: TableFilter
#
#------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from neuroimaging.extra.enthought.traits \
    import HasPrivateTraits, Str, Any, Instance, Trait, List, Property, Event, \
           Expression, Enum, false, true

from neuroimaging.extra.enthought.traits.ui \
    import View, Group, Item, Include, Handler, EnumEditor, EditorFactory

from neuroimaging.extra.enthought.traits.ui.menu \
    import Action

from neuroimaging.extra.enthought.traits.ui.table_column \
    import ObjectColumn

#-------------------------------------------------------------------------------
#  Trait definitions:
#-------------------------------------------------------------------------------

GenericTableFilterRuleOperation = Trait( '=', {
    '=':           'eq',
    '<>':          'ne',
    '<':           'lt',
    '<=':          'le',
    '>':           'gt',
    '>=':          'ge',
    'contains':    'contains',
    'starts with': 'starts_with',
    'ends with':   'ends_with'
} )

#-------------------------------------------------------------------------------
#  'TableFilter' class:
#-------------------------------------------------------------------------------

class TableFilter ( HasPrivateTraits ):

    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    # UI name of this filter (so user can identify it in the UI):
    name = Str( 'Default filter' )

    # Default name that can be automatically overridden:
    _name = Str( 'Default filter' )

    # Is the filter a template (i.e. non-deletable, non-editable)?
    template = false

    #---------------------------------------------------------------------------
    #  Class constants:
    #---------------------------------------------------------------------------

    # Traits ignored by the 'anytrait' handler:
    ignored_traits = [ '_name', 'template' ]

    #---------------------------------------------------------------------------
    #  Traits view definitions:
    #---------------------------------------------------------------------------

    traits_view     = View( 'name{Filter name}', '_', Include( 'filter_view' ),
                            title  = 'Edit filter',
                            width  = 0.2,
                            buttons = ['OK',
                                       'Cancel',
                                       Action( name='Help',
                                               action = 'show_help',
                                               defined_when =
                                               'ui.view_elements.content["filter_view"].help_id != ""',
                                       )],
                            )

    searchable_view = View( [
        [ Include( 'search_view' ), '|[]' ],
        [ 'handler.status~', '|[]<>' ],
        [ 'handler.find_next`Find the next matching item`',
          'handler.find_previous`Find the previous matching item`',
          'handler.select`Select all matching items`',
          'handler.OK`Exit search`', '-<>'   ],
        '|<>' ],
        title  = 'Search for',
        kind   = 'livemodal',
        undo   = False,
        revert = False,
        ok     = False,
        cancel = False,
        help   = False,
        width  = 0.25 )

    search_view = Group( Include( 'filter_view' ) )

    filter_view = Group()

    #---------------------------------------------------------------------------
    #  Returns whether a specified object meets the filter/search criteria:
    #  (Should normally be overridden)
    #---------------------------------------------------------------------------

    def filter ( self, object ):
        """ Returns whether a specified object meets the filter/search criteria.
        """
        return True

    #---------------------------------------------------------------------------
    #  Returns a user readable description of what kind of object will
    #  satisfy the filter:
    #  (Should normally be overridden):
    #---------------------------------------------------------------------------

    def description ( self ):
        """ Returns a user readable description of what kind of object will
            satisfy the filter.
        """
        return 'All items'

    #---------------------------------------------------------------------------
    #  Edits the contents of the filter:
    #---------------------------------------------------------------------------

    def edit ( self, object ):
        """ Edits the contents of the filter.

            Note: The 'object' is a sample object for the table the filter will
                  be applied to. It is supplied in case the filter needs to
                  extract data, or meta-data, from the object. If the table is
                  empty, the 'object' will be 'None'.
        """
        return self.edit_traits( kind = 'livemodal' )

    #---------------------------------------------------------------------------
    #  'object' interface:
    #---------------------------------------------------------------------------

    def __str__ ( self ):
        return self.name

    #---------------------------------------------------------------------------
    #  Event handlers:
    #---------------------------------------------------------------------------

    def _anytrait_changed ( self, name, old, new ):
        if ((name not in self.ignored_traits) and
            ((self.name == self._name) or (self.name == ''))):
            self.name = self._name = self.description()

#-------------------------------------------------------------------------------
#  'EvalTableFilter' class:
#-------------------------------------------------------------------------------

class EvalTableFilter ( TableFilter ):

    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    # Override the standard 'name' trait:
    name = 'Default evaluation filter'

    # Python expression which will be applied to each table item:
    expression = Expression

    #---------------------------------------------------------------------------
    #  Traits view definitions:
    #---------------------------------------------------------------------------

    filter_view = Group( 'expression' )

    #---------------------------------------------------------------------------
    #  Returns whether a specified object meets the filter/search criteria:
    #  (Should normally be overridden)
    #---------------------------------------------------------------------------

    def filter ( self, object ):
        """ Returns whether a specified object meets the filter/search criteria.
        """
        if self._traits is None:
            self._traits = object.trait_names()
        try:
            return eval( self.expression_, globals(),
                         object.get( *self._traits ) )
        except:
            return False

    #---------------------------------------------------------------------------
    #  Returns a user readable description of what kind of object will
    #  satisfy the filter:
    #  (Should normally be overridden):
    #---------------------------------------------------------------------------

    def description ( self ):
        """ Returns a user readable description of what kind of object will
            satisfy the filter.
        """
        return self.expression

#-------------------------------------------------------------------------------
#  'GenericTableFilterRule' class:
#-------------------------------------------------------------------------------

class GenericTableFilterRule ( HasPrivateTraits ):

    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    # Filter this rule is part of:
    filter = Instance( 'RuleTableFilter' )

    # Is the rule enabled?
    enabled = false

    # Is this rule an 'and' rule or an 'or' rule?
    and_or = Enum( 'and', 'or' )

    # EnumEditor used to edit the 'name' trait:
    name_editor = Instance( EditorFactory )

    # Name of the object trait this rule applies to:
    name = Str

    # Operation to be applied in the rule:
    operation = GenericTableFilterRuleOperation

    # Editor used to edit the 'value' trait:
    value_editor = Instance( EditorFactory )

    # Value to use in the operation when applying the rule to an object:
    value = Any

    #---------------------------------------------------------------------------
    #  Class constants:
    #---------------------------------------------------------------------------

    # Traits ignored by the 'anytrait' handler:
    ignored_traits = [ 'filter', 'name_editor', 'value_editor' ]

    #---------------------------------------------------------------------------
    #  Initializes the object:
    #---------------------------------------------------------------------------

    def __init__ ( self, **traits ):
        super( GenericTableFilterRule, self ).__init__( **traits )
        if self.name == '':
            names = self.filter._trait_values.keys()
            if len( names ) > 0:
                names.sort()
                self.name = names[0]

    #---------------------------------------------------------------------------
    #  Handles the value of the 'name' trait changing:
    #---------------------------------------------------------------------------

    def _name_changed ( self, name ):
        """ Handles the value of the 'name' trait changing.
        """
        filter            = self.filter
        self.value        = filter._trait_values.get( name )
        self.value_editor = filter._object.base_trait( name ).get_editor()

    #---------------------------------------------------------------------------
    #  Event handlers:
    #---------------------------------------------------------------------------

    def _anytrait_changed ( self, name, old, new ):
        if (name not in self.ignored_traits) and (self.filter is not None):
            self.filter.modified = True
            if name != 'enabled':
                self.enabled = True

    #---------------------------------------------------------------------------
    #  Returns a description of the filter:
    #---------------------------------------------------------------------------

    def description ( self ):
        """ Returns a description of the filter.
        """
        return '%s %s %s' % ( self.name, self.operation, self.value )

    #---------------------------------------------------------------------------
    #  Returns whether the rule is true for a specified object:
    #---------------------------------------------------------------------------

    def is_true ( self, object ):
        """ Returns whether the rule is true for a specified object.
        """
        try:
            return getattr( self, self.operation_ )(
                            getattr( object, self.name ), self.value )
        except:
            return False

    #---------------------------------------------------------------------------
    #  Implemenations of the various rule operations:
    #---------------------------------------------------------------------------

    def eq ( self, value1, value2 ):
        return (value1 == value2)

    def ne ( self, value1, value2 ):
        return (value1 != value2)

    def lt ( self, value1, value2 ):
        return (value1 < value2)

    def le ( self, value1, value2 ):
        return (value1 <= value2)

    def gt ( self, value1, value2 ):
        return (value1 > value2)

    def ge ( self, value1, value2 ):
        return (value1 >= value2)

    def contains ( self, value1, value2 ):
        return (value1.lower().find( value2.lower() ) >= 0)

    def starts_with ( self, value1, value2 ):
        return (value1[ : len( value2 ) ].lower() == value2.lower())

    def ends_with ( self, value1, value2 ):
        return (value1[ -len( value2 ): ].lower() == value2.lower())

#-------------------------------------------------------------------------------
#  'GenericTableFilterRuleEnabledColumn' class:
#-------------------------------------------------------------------------------

class GenericTableFilterRuleEnabledColumn ( ObjectColumn ):

    #---------------------------------------------------------------------------
    #  Returns the value of the column for a specified object:
    #---------------------------------------------------------------------------

    def get_value ( self, object ):
        """ Returns the traits editor of the column for a specified object.
        """
        return [ '', '==>' ][ object.enabled ]

#-------------------------------------------------------------------------------
#  'GenericTableFilterRuleAndOrColumn' class:
#-------------------------------------------------------------------------------

class GenericTableFilterRuleAndOrColumn ( ObjectColumn ):

    #---------------------------------------------------------------------------
    #  Returns the value of the column for a specified object:
    #---------------------------------------------------------------------------

    def get_value ( self, object ):
        """ Returns the traits editor of the column for a specified object.
        """
        if object.and_or == 'or':
            return 'or'
        return ''

#-------------------------------------------------------------------------------
#  'GenericTableFilterRuleNameColumn' class:
#-------------------------------------------------------------------------------

class GenericTableFilterRuleNameColumn ( ObjectColumn ):

    #---------------------------------------------------------------------------
    #  Returns the traits editor of the column for a specified object:
    #---------------------------------------------------------------------------

    def get_editor ( self, object ):
        """ Returns the traits editor of the column for a specified object.
        """
        return object.name_editor

#-------------------------------------------------------------------------------
#  'GenericTableFilterRuleValueColumn' class:
#-------------------------------------------------------------------------------

class GenericTableFilterRuleValueColumn ( ObjectColumn ):

    #---------------------------------------------------------------------------
    #  Returns the traits editor of the column for a specified object:
    #---------------------------------------------------------------------------

    def get_editor ( self, object ):
        """ Returns the traits editor of the column for a specified object.
        """
        return object.value_editor

#-------------------------------------------------------------------------------
#  Defines the columns to display in the generic filter rule table:
#-------------------------------------------------------------------------------

generic_table_filter_rule_columns = [
    GenericTableFilterRuleAndOrColumn( name = 'and_or', label = 'or' ),
    GenericTableFilterRuleNameColumn(  name = 'name' ),
    ObjectColumn(                      name = 'operation' ),
    GenericTableFilterRuleValueColumn( name = 'value' )
]

#-------------------------------------------------------------------------------
#  'RuleTableFilter' class:
#-------------------------------------------------------------------------------

class RuleTableFilter ( TableFilter ):

    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    # Override the standard 'name' trait:
    name = 'Default rule-based filter'

    # List of the filter rules to be applied:
    rules = List( GenericTableFilterRule )

    # Event fired when the contents of the filter have changed:
    modified = Event

    # Sample object the filter will apply to:
    _object = Any

    # Trait names and default values map:
    _trait_values = Any

    #---------------------------------------------------------------------------
    #  Traits view definitions:  
    #---------------------------------------------------------------------------

    error_view = View(
                     Item( label = 'A menu or rule based filter can only be '
                                   'created for tables with at least one entry'
                     ),
                     title        = 'Error Creating Filter',
                     kind         = 'livemodal',
                     close_result = False,
                     buttons      = [ 'Cancel' ] )

    #---------------------------------------------------------------------------
    #  Returns whether a specified object meets the filter/search criteria:
    #  (Should normally be overridden)
    #---------------------------------------------------------------------------

    def filter ( self, object ):
        """ Returns whether a specified object meets the filter/search criteria.
        """
        is_first = is_true = True
        for rule in self.rules:
            if rule.and_or == 'or':
                if is_true and (not is_first):
                    return True
                is_true = True
            if is_true:
                is_true = rule.is_true( object )
            is_first = False
        return is_true

    #---------------------------------------------------------------------------
    #  Returns a user readable description of what kind of object will
    #  satisfy the filter:
    #  (Should normally be overridden):
    #---------------------------------------------------------------------------

    def description ( self ):
        """ Returns a user readable description of what kind of object will
            satisfy the filter.
        """
        ors  = []
        ands = []
        if len( self.rules ) > 0:
            for rule in self.rules:
                if rule.and_or == 'or':
                    if len( ands ) > 0:
                        ors.append( ' and '.join( ands ) )
                        ands = []
                ands.append( rule.description() )
        if len( ands ) > 0:
            ors.append( ' and '.join( ands ) )
        if len( ors ) == 1:
            return ors[0]
        if len( ors ) > 1:
            return ' or '.join( [ '(%s)' % t for t in ors ] )
        return super( RuleTableFilter, self ).description()

    #---------------------------------------------------------------------------
    #  Edits the contents of the filter:
    #---------------------------------------------------------------------------

    def edit ( self, object ):
        """ Edits the contents of the filter.

            Note: The 'object' is a sample object for the table the filter will
                  be applied to. It is supplied in case the filter needs to
                  extract data, or meta-data, from the object. If the table is
                  empty, the 'object' will be 'None'.
        """
        self._object = object
        if object is None:
            return self.edit_traits( view = 'error_view' )
            
        names              = object.editable_traits()
        self._trait_values = object.get( names )
        return self.edit_traits( view = View( [
                        [ 'name{Filter name}', '_' ],
                        [ Item( 'rules',
                                editor = self._get_table_editor( names ) ),
                          '|<>' ] ],
                        title     = 'Edit rule-based filter',
                        kind      = 'livemodal',
                        resizable = True,
                        undo      = False,
                        revert    = False,
                        help      = False,
                        width     = 0.2,
                        height    = 0.25 ) )

    #---------------------------------------------------------------------------
    #  Returns a table editor to use for editing the filter:
    #---------------------------------------------------------------------------

    def _get_table_editor ( self, names ):
        """ Returns a table editor to use for editing the filter.
        """
        from neuroimaging.extra.enthought.traits.ui import TableEditor

        return TableEditor( columns        = generic_table_filter_rule_columns,
                            orientation    = 'vertical',
                            deletable      = True,
                            sortable       = False,
                            configurable   = False,
                            row_factory    = GenericTableFilterRule,
                            row_factory_kw = {
                                'filter':      self,
                                'name_editor': EnumEditor( values = names )  } )

    #---------------------------------------------------------------------------
    #  Returns the state to be pickled (override of object):
    #---------------------------------------------------------------------------

    def __getstate__ ( self ):
        """ Returns the state to be pickled (override of object).
        """
        dict = self.__dict__.copy()
        del dict[ '_object' ]
        del dict[ '_trait_values' ]
        return dict

#-------------------------------------------------------------------------------
#  Defines the columns to display in the menu filter rule table:
#-------------------------------------------------------------------------------

menu_table_filter_rule_columns = [
    GenericTableFilterRuleEnabledColumn( name = 'enabled', label = '' ),
    GenericTableFilterRuleNameColumn(    name = 'name' ),
    ObjectColumn(                        name = 'operation' ),
    GenericTableFilterRuleValueColumn(   name = 'value' )
]

#-------------------------------------------------------------------------------
#  'MenuTableFilter' class:
#-------------------------------------------------------------------------------

class MenuTableFilter ( RuleTableFilter ):

    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    # Override the standard 'name' trait:
    name = 'Default menu-based filter'

    #---------------------------------------------------------------------------
    #  Returns whether a specified object meets the filter/search criteria:
    #  (Should normally be overridden)
    #---------------------------------------------------------------------------

    def filter ( self, object ):
        """ Returns whether a specified object meets the filter/search criteria.
        """
        for rule in self.rules:
            if rule.enabled and (not rule.is_true( object )):
                return False
        return True

    #---------------------------------------------------------------------------
    #  Returns a user readable description of what kind of object will
    #  satisfy the filter:
    #  (Should normally be overridden):
    #---------------------------------------------------------------------------

    def description ( self ):
        """ Returns a user readable description of what kind of object will
            satisfy the filter.
        """
        result = ' and '.join( [ rule.description() for rule in self.rules
                                 if rule.enabled ] )
        if result != '':
            return result
        return 'All items'

    #---------------------------------------------------------------------------
    #  Returns a table editor to use for editing the filter:
    #---------------------------------------------------------------------------

    def _get_table_editor ( self, names ):
        """ Returns a table editor to use for editing the filter.
        """
        from neuroimaging.extra.enthought.traits.ui import TableEditor

        names       = self._object.editable_traits()
        name_editor = EnumEditor( values = names )
        if len( self.rules ) == 0:
            self.rules  = [ GenericTableFilterRule(
                                filter      = self,
                                name_editor = name_editor ).set(
                                name        = name )
                            for name in names ]
            for rule in self.rules:
                rule.enabled = False

        return TableEditor( columns        = menu_table_filter_rule_columns,
                            orientation    = 'vertical',
                            deletable      = True,
                            sortable       = False,
                            configurable   = False,
                            row_factory    = GenericTableFilterRule,
                            row_factory_kw = {
                                'filter':      self,
                                'name_editor': name_editor  } )

#-------------------------------------------------------------------------------
#  Define some standard template filters:
#-------------------------------------------------------------------------------

EvalFilterTemplate = EvalTableFilter( name     = 'Evaluation filter template',
                                      template = True )
RuleFilterTemplate = RuleTableFilter( name     = 'Rule-based filter template',
                                      template = True )
MenuFilterTemplate = MenuTableFilter( name     = 'Menu-based filter template',
                                      template = True )

