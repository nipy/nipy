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
# Date: 06/21/2002
# Description: Define a 'traits' package that allows other classes to easily
#              define 'type-checked' and/or 'delegated' traits for their
#              instances.
#
#              Note: A 'trait' is similar to a 'property', but is used instead
#              of the word 'property' to differentiate it from the Python
#              language 'property' feature.
#------------------------------------------------------------------------------

from info_traits \
    import __doc__

from trait_base \
    import Undefined, Missing, Self

from trait_errors \
    import TraitError, DelegationError

from category \
    import Category

from trait_db \
    import tdb
    
from traits \
    import Event, List, Dict, Tuple, Range, Constant, CTrait, Trait, Delegate, \
           Property, Expression, Button, ToolbarButton, PythonValue, Any, Int, \
           Long, Float, Str, Unicode, Complex, Bool, CInt, CLong, CFloat, \
           CStr, CUnicode, WeakRef
           
from traits \
    import CComplex, CBool, false, true, Regex, String, Password, File, \
           Directory, Function, Method, Class, Instance, Module, Type, This, \
           self, Python, Disallow, ReadOnly, undefined, missing, ListInt
           
from traits \
    import ListFloat, ListStr, ListUnicode, ListComplex, ListBool, \
           ListFunction, ListMethod, ListClass, ListInstance, ListThis, \
           DictStrAny, DictStrStr, DictStrInt, DictStrLong, DictStrFloat 
           
from traits \
    import DictStrBool, DictStrList, TraitFactory, Callable, Array, CArray, \
           Enum, Code, HTML, Default, Color, RGBColor, RGBAColor, Font, \
           KivaFont, TraitFactory
    
from has_traits \
    import method, HasTraits, HasStrictTraits, HasPrivateTraits, \
           SingletonHasTraits, SingletonHasStrictTraits, \
           SingletonHasPrivateTraits, MetaHasTraits, Vetoable, VetoableEvent, \
           traits_super
           
from trait_handlers \
    import TraitHandler, TraitRange, TraitString, TraitType, TraitCastType, \
           TraitInstance, ThisClass, TraitClass, TraitFunction, TraitEnum, \
           TraitPrefixList, TraitMap, TraitPrefixMap, TraitCompound, \
           TraitList, TraitListEvent, TraitDict, TraitDictEvent, TraitTuple
     
from traits \
    import UIDebugger
                            
