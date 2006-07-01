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
# Original Date:                         06/21/2002
# Description: Define the classes needed to implement and support the trait
#              change notification mechanism.
#
#  Symbols defined: TraitChangeNotifyWrapper
#                   StaticAnyTraitChangeNotifyWrapper
#                   StaticTraitChangeNotifyWrapper
#
#  Refactored into a separate module: 07/04/2003
#------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

import weakref

from types \
    import MethodType

import traceback
import sys

#-------------------------------------------------------------------------------
#  Logs any exceptions generated in a trait notification handler:
#-------------------------------------------------------------------------------

traits_logger = None

def log_exception ( ):
    """ Logs any exceptions generated in a trait notification handler.
    """
    # When stack depth is too great, the logger can't always log the message.
    # Make sure that it goes to the console at a minimum.
    t, v, tb = sys.exc_info()
    if t is RuntimeError and v.args[0] == 'maximum recursion depth exceeded':
        s = ''.join(traceback.format_exception(t, v, tb))
        sys.__stderr__.write('Exception occurred in traits ' + \
            'notification handler.\n' + s + '\n')

    global traits_logger

    if traits_logger is None:
        import logging
        traits_logger = logging.getLogger( 'neuroimaging.extra.enthought.traits' )
        handler       = logging.StreamHandler()
        handler.setFormatter( logging.Formatter( '%(message)s' ) )
        traits_logger.addHandler( handler )
        print ('Exception occurred in traits notification handler.\n'
               'Please check the log file for details.')

    try:
        traits_logger.exception( 'Exception occurred in traits notification '
                                 'handler' )
    except Exception:
        # Ignore anything we can't log the above way
        pass

#-------------------------------------------------------------------------------
#  'StaticAnyTraitChangeNotifyWrapper' class:
#-------------------------------------------------------------------------------

class StaticAnyTraitChangeNotifyWrapper:

    def __init__ ( self, handler ):
         self.handler  = handler
         self.__call__ = getattr( self, 'call_%d' %
                                        handler.func_code.co_argcount )

    def equals ( self, handler ):
        return False

    def call_0 ( self, object, trait_name, old, new ):
        try:
            self.handler()
        except:
            log_exception()

    def call_1 ( self, object, trait_name, old, new ):
        try:
            self.handler( object )
        except:
            log_exception()

    def call_2 ( self, object, trait_name, old, new ):
        try:
            self.handler( object, trait_name )
        except:
            log_exception()

    def call_3 ( self, object, trait_name, old, new ):
        try:
            self.handler( object, trait_name, new )
        except:
            log_exception()

    def call_4 ( self, object, trait_name, old, new ):
        try:
            self.handler( object, trait_name, old, new )
        except:
            log_exception()

#-------------------------------------------------------------------------------
#  'StaticTraitChangeNotifyWrapper' class:
#-------------------------------------------------------------------------------

class StaticTraitChangeNotifyWrapper:

    def __init__ ( self, handler ):
        self.handler  = handler
        self.__call__ = getattr( self, 'call_%d' %
                                       handler.func_code.co_argcount )

    def equals ( self, handler ):
        return False

    def call_0 ( self, object, trait_name, old, new ):
        try:
            self.handler()
        except:
            log_exception()

    def call_1 ( self, object, trait_name, old, new ):
        try:
            self.handler( object )
        except:
            log_exception()

    def call_2 ( self, object, trait_name, old, new ):
        try:
            self.handler( object, new )
        except:
            log_exception()

    def call_3 ( self, object, trait_name, old, new ):
        try:
            self.handler( object, old, new )
        except:
            log_exception()

    def call_4 ( self, object, trait_name, old, new ):
        try:
            self.handler( object, trait_name, old, new )
        except:
            log_exception()

#-------------------------------------------------------------------------------
#  'TraitChangeNotifyWrapper' class:
#-------------------------------------------------------------------------------

class TraitChangeNotifyWrapper:

    def __init__ ( self, handler, owner ):
        func = handler
        if type( handler ) is MethodType:
            func   = handler.im_func
            object = handler.im_self
            if object is not None:
                self.object   = weakref.ref( object, self.listener_deleted )
                self.name     = handler.__name__
                self.owner    = owner
                self.__call__ = getattr( self, 'rebind_call_%d' %
                                         (func.func_code.co_argcount - 1) )
                return
        self.name     = None
        self.handler  = handler
        self.__call__ = getattr( self, 'call_%d' %
                                 handler.func_code.co_argcount )

    def equals ( self, handler ):
        if handler is self:
            return True
        if (type( handler ) is MethodType) and (handler.im_self is not None):
            return ((handler.__name__ == self.name) and
                    (handler.im_self is self.object()))
        return ((self.name is None) and (handler == self.handler))

    def listener_deleted ( self, ref ):
        self.owner.remove( self )
        self.object = self.owner = None

    def dispose ( self ):
        self.object = None

    def call_0 ( self, object, trait_name, old, new ):
        try:
            self.handler()
        except:
            log_exception()

    def call_1 ( self, object, trait_name, old, new ):
        try:
            self.handler( new )
        except:
            log_exception()

    def call_2 ( self, object, trait_name, old, new ):
        try:
            self.handler( trait_name, new )
        except:
            log_exception()

    def call_3 ( self, object, trait_name, old, new ):
        try:
            self.handler( object, trait_name, new )
        except:
            log_exception()

    def call_4 ( self, object, trait_name, old, new ):
        try:
            self.handler( object, trait_name, old, new )
        except:
            log_exception()

    def rebind_call_0 ( self, object, trait_name, old, new ):
        try:
            getattr( self.object(), self.name )()
        except:
            log_exception()

    def rebind_call_1 ( self, object, trait_name, old, new ):
        try:
            getattr( self.object(), self.name )( new )
        except:
            log_exception()

    def rebind_call_2 ( self, object, trait_name, old, new ):
        try:
            getattr( self.object(), self.name )( trait_name, new )
        except:
            log_exception()

    def rebind_call_3 ( self, object, trait_name, old, new ):
        try:
            getattr( self.object(), self.name )( object, trait_name, new )
        except:
            log_exception()

    def rebind_call_4 ( self, object, trait_name, old, new ):
        try:
            getattr( self.object(), self.name )( object, trait_name, old, new )
        except:
            log_exception()

