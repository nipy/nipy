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
# Date: 11/26/2004
# Description: Adds all of the core traits to the traits data base.
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    
    from neuroimaging.extra.enthought.traits import Event, List, Dict, Any, Int, Long, Float, Str
    from neuroimaging.extra.enthought.traits import Unicode, Complex, Bool, CInt, CLong, CFloat
    from neuroimaging.extra.enthought.traits import CStr, CUnicode, CComplex, CBool, false, true
    from neuroimaging.extra.enthought.traits import String, Password, File, Directory, Function
    from neuroimaging.extra.enthought.traits import Method, Class, Module, Type, This, self, Python
    from neuroimaging.extra.enthought.traits import ReadOnly, ListInt, ListFloat, ListStr
    from neuroimaging.extra.enthought.traits import ListUnicode, ListComplex, ListBool
    from neuroimaging.extra.enthought.traits import ListFunction, ListMethod, ListClass
    from neuroimaging.extra.enthought.traits import ListInstance, ListThis, DictStrAny, DictStrStr
    from neuroimaging.extra.enthought.traits import DictStrInt, DictStrLong, DictStrFloat
    from neuroimaging.extra.enthought.traits import DictStrBool,DictStrList
    from neuroimaging.extra.enthought.traits import tdb
         
    define = tdb.define
    define( 'Event',        Event )
    define( 'List',         List )
    define( 'Dict',         Dict )
    define( 'Any',          Any )
    define( 'Int',          Int )
    define( 'Long',         Long )
    define( 'Float',        Float )
    define( 'Str',          Str )
    define( 'Unicode',      Unicode )
    define( 'Complex',      Complex )
    define( 'Bool',         Bool )
    define( 'CInt',         CInt )
    define( 'CLong',        CLong )
    define( 'CFloat',       CFloat )
    define( 'CStr',         CStr )
    define( 'CUnicode',     CUnicode )
    define( 'CComplex',     CComplex )
    define( 'CBool',        CBool )
    define( 'false',        false )
    define( 'true',         true )
    define( 'String',       String )
    define( 'Password',     Password )
    define( 'File',         File )
    define( 'Directory',    Directory )
#   define( 'Function',     Function )
#   define( 'Method',       Method )
#   define( 'Class',        Class )
#   define( 'Module',       Module )
    define( 'Type',         Type )
    define( 'This',         This )
#   define( 'self',         self )
    define( 'Python',       Python )
##  define( 'ReadOnly',     ReadOnly ) <-- 'Undefined' doesn't have right
                                         # semantics when persisted
    define( 'ListInt',      ListInt )
    define( 'ListFloat',    ListFloat )
    define( 'ListStr',      ListStr )
    define( 'ListUnicode',  ListUnicode )
    define( 'ListComplex',  ListComplex )
    define( 'ListBool',     ListBool )
#   define( 'ListFunction', ListFunction )
#   define( 'ListMethod',   ListMethod )
#   define( 'ListClass',    ListClass )
#   define( 'ListInstance', ListInstance )
    define( 'ListThis',     ListThis )
    define( 'DictStrAny',   DictStrAny )
    define( 'DictStrStr',   DictStrStr )
    define( 'DictStrInt',   DictStrInt )
    define( 'DictStrLong',  DictStrLong )
    define( 'DictStrFloat', DictStrFloat )
    define( 'DictStrBool',  DictStrBool )
    define( 'DictStrList',  DictStrList )
    
