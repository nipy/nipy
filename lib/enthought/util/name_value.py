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
""" A name/value pair. """


# Enthought library imports.
from enthought.traits import Any, HasTraits, Str, Trait, TraitDict
# from enthought.util.traits.editor.dict.table_editor import TableEditor


class NameValue(HasTraits):
    """ A name/value pair. """

    name = Str

    value = Any
    

def name_value_factory():
    """ A factory function for name/value pairs! """
    
    name_value = NameValue()
    
    result = name_value.ok_cancel_dialog()
# bwd ui hack
##     if result != HasTraitsPlus.OK:
##         name_value = None

    return name_value


# A name/value dicationaryr trait.
name_value_dict_trait = Trait({}, TraitDict()
# bwd ui hack
##                               ,

##     editor=TableEditor(
##         name_value_factory, ['name', 'value']
##     )
)

#### EOF ######################################################################
