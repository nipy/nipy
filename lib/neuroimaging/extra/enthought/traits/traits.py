#------------------------------------------------------------------------------
#  Copyright (c) 2005, Enthought, Inc.
#  All rights reserved.
#  
#  This software is provided without warranty under the terms of the BSD
#  license included in enthought/LICENSE.txt and may be redistributed only
#  under the conditions described in the aforementioned license.  The license
#  is also available online at http://www.enthought.com/licenses/BSD.txt
#  Thanks for using Enthought open source!
#  
#  Author: David C. Morrill
#  Original Date: 06/21/2002
#
#  Description:
#
#  Defines the 'core' traits for the traits package, which defines a
#  capability that allows other classes to easily define objects with
#  traits (i.e. attributes) that support:
#    - Initialization (have predefined values that do not need to be explicitly
#                      initialized in the class constructor or elsewhere)
#    - Validation     (have flexible, type checked values)
#    - Delegation     (have values that can be delegated to other objects)
#    - Notification   (can automatically notify interested parties when changes 
#                      are made to their value.
#    - Visualization  (can automatically construct automatic or 
#                      programmer-defined user interfaces that allow their 
#                      values to be edited or displayed)
#
#  Note: 'trait' is a synonym for 'property', but is used instead of the 
#  word 'property' to differentiate it from the Python language 'property'
#  feature.
#
#  Rewritten as a C-based type extension: 06/21/2004
#------------------------------------------------------------------------------
               
#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

import sys
import trait_handlers

from inspect \
    import stack
    
from ctraits \
    import cTrait, CTraitMethod
    
from trait_base \
    import SequenceTypes, Self, Undefined, Missing, TypeTypes, class_of, \
           add_article, enumerate, BooleanType
           
from trait_errors \
    import TraitError
    
from trait_handlers \
    import TraitHandler, TraitInstance, TraitList, TraitDict, TraitFunction, \
           TraitType, TraitCastType, TraitEnum, TraitCompound, TraitMap, \
           TraitString, ThisClass, TraitRange, TraitTuple, TraitCallable, \
           TraitExpression, TraitWeakRef
                           
from types \
    import NoneType, IntType, LongType, FloatType, ComplexType, StringType, \
           UnicodeType, ListType, TupleType, DictType, FunctionType, \
           ClassType, ModuleType, MethodType, InstanceType, TypeType

#-------------------------------------------------------------------------------
#  Editor factory functions: 
#-------------------------------------------------------------------------------

PasswordEditor      = None
MultilineTextEditor = None
SourceCodeEditor    = None
HTMLTextEditor      = None
PythonShellEditor   = None

def password_editor ( ):
    global PasswordEditor
    
    if PasswordEditor is None:
        from neuroimaging.extra.enthought.traits.ui import TextEditor
        PasswordEditor = TextEditor( password = True )
        
    return PasswordEditor
    
def multi_line_text_editor ( ):
    global MultilineTextEditor
    
    if MultilineTextEditor is None:
        from neuroimaging.extra.enthought.traits.ui import TextEditor
        MultilineTextEditor = TextEditor( multi_line = True )
        
    return MultilineTextEditor
    
def code_editor ( ):
    global SourceCodeEditor
    
    if SourceCodeEditor is None:
        from neuroimaging.extra.enthought.traits.ui import CodeEditor
        SourceCodeEditor = CodeEditor()
        
    return SourceCodeEditor
    
def html_editor ( ):
    global HTMLTextEditor
    
    if HTMLTextEditor is None:
        from neuroimaging.extra.enthought.traits.ui import HTMLEditor
        HTMLTextEditor = HTMLEditor()
        
    return HTMLTextEditor
    
def shell_editor ( ):
    global PythonShellEditor
    
    if PythonShellEditor is None:
        from neuroimaging.extra.enthought.traits.ui import ShellEditor
        PythonShellEditor = ShellEditor()
        
    return PythonShellEditor
    
#-------------------------------------------------------------------------------
#  'CTrait' class (extends the underlying cTrait c-based type):
#-------------------------------------------------------------------------------
    
class CTrait ( cTrait ):
    
    #---------------------------------------------------------------------------
    #  Allows a derivative trait to be defined from this one:
    #---------------------------------------------------------------------------
    
    def __call__ ( self, *args, **metadata ):
        if 'parent' not in metadata:
            metadata[ 'parent' ] = self
        return Trait( *(args + ( self, )), **metadata )

    #---------------------------------------------------------------------------
    #  Returns the user interface editor associated with the trait:
    #---------------------------------------------------------------------------

    def get_editor ( self ):
        from neuroimaging.extra.enthought.traits.ui import EditorFactory
        
        # See if we have an editor:
        editor = self.editor
        if editor is None:
            
            # Else see if the trait handler has an editor:
            handler = self.handler
            if handler is not None:
                editor = handler.get_editor( self )
                
            # If not, give up and use a default text editor:
            if editor is None:
                from neuroimaging.extra.enthought.traits.ui import TextEditor
                editor = TextEditor
                    
        # If the result is not an EditoryFactory:
        if not isinstance( editor, EditorFactory ):
            # Then it should be a factory for creating them:
            args   = ()
            traits = {}
            if type( editor ) in SequenceTypes:
                for item in editor[:]: 
                    if type( item ) in SequenceTypes:
                        args = tuple( item )
                    elif isinstance( item, dict ):
                        traits = item
                    else:
                        editor = item
            editor = editor( *args, **traits )
            
        # Cache the result:
        self.editor = editor
        
        # Return the resulting EditorFactory object:
        return editor
    
    #---------------------------------------------------------------------------
    #  Returns the help text for a trait:
    #---------------------------------------------------------------------------
    
    def get_help ( self, full = True ):
        if full:
            help = self.help
            if help is not None:
                return help
        handler = self.handler
        if handler is not None:
            info = 'must be %s.' % handler.info()
        else:
            info = 'may be any value.'
        desc = self.desc            
        if self.desc is None:
            return info.capitalize()
        return 'Specifies %s and %s' % ( desc, info )
    
    #---------------------------------------------------------------------------
    #  Returns the pickleable form of a CTrait object:
    #---------------------------------------------------------------------------
    
    def __reduce_ex__ ( self, protocol ):
        return ( __newobj__, ( self.__class__, 0 ), self.__getstate__() )
        
# Make sure the Python-level version of the trait class is known to all
# interested parties:
import ctraits
ctraits._ctrait( CTrait )
       
#-------------------------------------------------------------------------------
#  Constants:
#-------------------------------------------------------------------------------

ConstantTypes    = ( NoneType, IntType, LongType, FloatType, ComplexType,
                     StringType, UnicodeType )
                
PythonTypes      = ( StringType,   UnicodeType,  IntType,    LongType,
                     FloatType,    ComplexType,  ListType,   TupleType,
                     DictType,     FunctionType, MethodType, ClassType,
                     InstanceType, TypeType,     NoneType )
                     
CallableTypes    = ( FunctionType, MethodType )
                
TraitTypes       = ( TraitHandler, CTrait )

MutableTypes     = ( list, dict )

DefaultValues = {
    StringType:  '',   
    UnicodeType: u'',  
    IntType:     0,
    LongType:    0L,
    FloatType:   0.0,
    ComplexType: 0j,  
    ListType:    [],   
    TupleType:   (),
    DictType:    {},
    BooleanType: False
}

DefaultValueSpecial = [ Missing, Self ]
DefaultValueTypes   = [ ListType, DictType ]

#-------------------------------------------------------------------------------
#  Function used to unpickle new-style objects:
#-------------------------------------------------------------------------------

def __newobj__ ( cls, *args ):
    return cls.__new__( cls, *args )
        
#-------------------------------------------------------------------------------
#  Returns the type of default value specified:
#-------------------------------------------------------------------------------
        
def _default_value_type ( default_value ):
    try:
        return DefaultValueSpecial.index( default_value ) + 1
    except:
        try:
            return DefaultValueTypes.index( type( default_value ) ) + 3
        except:
            return 0
    
#-------------------------------------------------------------------------------
#  Returns the correct argument count for a specified function or method:
#-------------------------------------------------------------------------------
    
def _arg_count ( func ):
    if (type( func ) is MethodType) and (func.im_self is not None):
        return func.func_code.co_argcount - 1
    return func.func_code.co_argcount
    
#-------------------------------------------------------------------------------
#  'TraitFactory' class:
#-------------------------------------------------------------------------------
    
class TraitFactory ( object ):
    
    #---------------------------------------------------------------------------
    #  Initializes the object:
    #---------------------------------------------------------------------------
    
    def __init__ ( self, maker_function = None ):
        if maker_function is not None:
            self.maker_function = maker_function
    
    #---------------------------------------------------------------------------
    #  Creates a CTrait instance:
    #---------------------------------------------------------------------------
        
    def __call__ ( self, *args, **metadata ):
        return self.maker_function( *args, **metadata )

#-------------------------------------------------------------------------------
#  Returns a trait created from a TraitFactory instance:
#-------------------------------------------------------------------------------
       
_trait_factory_instances = {}

def trait_factory ( trait ):
    global _trait_factory_instances

    tid = id( trait )
    if tid not in _trait_factory_instances:
        _trait_factory_instances[ tid ] = trait()
    return _trait_factory_instances[ tid ]
    
#-------------------------------------------------------------------------------
#  Casts a CTrait or TraitFactory to a CTrait but returns None if it is neither:
#-------------------------------------------------------------------------------
    
def trait_cast ( something ):
    """ Casts a CTrait or TraitFactory to a CTrait but returns None if it is 
        neither.
    """
    if isinstance( something, CTrait ):
        return something
    if isinstance( something, TraitFactory ):
        return trait_factory( something )
    return None
    
#-------------------------------------------------------------------------------
#  Returns a trait derived from its input:
#-------------------------------------------------------------------------------
    
def trait_from ( something ):
    """ Returns a trait derived from its input.
    """
    if isinstance( something, CTrait ):
        return something
    if something is None:
        something = Any
    if isinstance( something, TraitFactory ):
        return trait_factory( something )
    return Trait( something )
    
# Patch the reference to 'trait_from' in 'trait_handlers.py':    
trait_handlers.trait_from = trait_from    
    
#-------------------------------------------------------------------------------
#  Define special 'factory' functions for creating common traits:
#-------------------------------------------------------------------------------

def Any ( value = None, **metadata ):
    metadata[ 'type' ] = 'trait'
    trait = CTrait( 0 )
    trait.default_value( _default_value_type( value ), value )
    trait.rich_comparison( metadata.get( 'rich_compare', True ) )
    trait.__dict__ = metadata.copy()
    return trait
    
Any = TraitFactory( Any )    
    
#--- 'Coerced' traits ----------------------------------------------------------    
    
def Int ( value = 0, **metadata ):
    return Trait( value, TraitType( int ), **metadata )
    
Int = TraitFactory( Int )    
    
def Long ( value = 0L, **metadata ):
    return Trait( value, TraitType( long ), **metadata )
    
Long = TraitFactory( Long )
    
def Float ( value = 0.0, **metadata ):
    return Trait( value, TraitType( float ), **metadata )    
    
Float = TraitFactory( Float )
    
def Complex ( value = 0.0+0.0j, **metadata ):
    return Trait( value, TraitType( complex ), **metadata )    
    
Complex = TraitFactory( Complex )
    
def Str ( value = '', **metadata ):
    if 'editor' not in metadata:
        metadata[ 'editor' ] = multi_line_text_editor
    return Trait( value, TraitType( str ), TraitType( unicode ), **metadata )
    
Str = TraitFactory( Str )
    
def Unicode ( value = u'', **metadata ):
    if 'editor' not in metadata:
        metadata[ 'editor' ] = multi_line_text_editor
    return Trait( value, TraitType( unicode ), **metadata )    
    
Unicode = TraitFactory( Unicode )
    
def Bool ( value = False, **metadata ):
    return Trait( value, TraitType( bool ), **metadata ) 
    
Bool = TraitFactory( Bool )
    
#--- 'Cast' traits -------------------------------------------------------------

def CInt ( value = 0, **metadata ):
    return Trait( value, TraitCastType( int ), **metadata )
    
CInt = TraitFactory( CInt )    
    
def CLong ( value = 0L, **metadata ):
    return Trait( value, TraitCastType( long ), **metadata )    
    
CLong = TraitFactory( CLong )    
    
def CFloat ( value = 0.0, **metadata ):
    return Trait( value, TraitCastType( float ), **metadata )    
    
CFloat = TraitFactory( CFloat )    
    
def CComplex ( value = 0.0+0.0j, **metadata ):
    return Trait( value, TraitCastType( complex ), **metadata )    
    
CComplex = TraitFactory( CComplex )    
    
def CStr ( value = '', **metadata ):
    if 'editor' not in metadata:
        metadata[ 'editor' ] = multi_line_text_editor
    return Trait( value, TraitCastType( str ), **metadata )    
    
CStr = TraitFactory( CStr )    
    
def CUnicode ( value = u'', **metadata ):
    if 'editor' not in metadata:
        metadata[ 'editor' ] = multi_line_text_editor
    return Trait( value, TraitCastType( unicode ), **metadata )    
    
CUnicode = TraitFactory( CUnicode )    
    
def CBool ( value = False, **metadata ):
    return Trait( value, TraitCastType( bool ), **metadata ) 
    
CBool = TraitFactory( CBool )    
    
#--- 'sequence' and 'mapping' traits -------------------------------------------

def List ( trait = None, value = None, minlen = 0, maxlen = sys.maxint,
           items = True, **metadata ):
    metadata.setdefault( 'copy', 'deep' )
    if isinstance( trait, SequenceTypes ):
        trait, value = value, list( trait )
    if value is None:
        value = []
    handler = TraitList( trait, minlen, maxlen, items )
    if handler.item_trait.instance_handler == '_instance_changed_handler':
        metadata.setdefault( 'instance_handler', '_list_changed_handler' )
    return Trait( value, handler, **metadata )
    
List = TraitFactory( List )    

def Tuple ( *traits, **metadata ):
    if len( traits ) == 0:
        return Trait( (), TraitType( tuple ), **metadata )
    value = None
    if isinstance( traits[0], tuple ):
        value, traits = traits[0], traits[1:]
        if len( traits ) == 0:
            traits = [ Trait( element ) for element in value ]
    tt = TraitTuple( *traits )
    if value is None:
        value = tuple( [ trait.default_value()[1] for trait in tt.traits ] )
    return Trait( value, tt, **metadata )
    
Tuple = TraitFactory( Tuple )    
    
def Dict ( key_trait = None, value_trait = None, value = None, items = True,
           **metadata ):
    if isinstance( key_trait, dict ):
        key_trait, value_trait, value = value_trait, value, key_trait
    if value is None:
        value = {}
    return Trait( value, TraitDict( key_trait, value_trait, items ), 
                  **metadata )
    
Dict = TraitFactory( Dict )    
    
#--- 'array' traits ------------------------------------------------------------

def Array ( typecode = None, shape = None, value = None, **metadata ):
    return _Array( typecode, shape, value, coerce = False, **metadata )
    
Array = TraitFactory( Array )

def CArray ( typecode = None, shape = None, value = None, **metadata ):
    return _Array( typecode, shape, value, coerce = True, **metadata )
    
CArray = TraitFactory( CArray )    

def _Array ( typecode = None, shape = None, value = None, coerce = False, 
             **metadata ):
    metadata[ 'array' ] = True
    from trait_numeric import zeros, Typecodes, TraitArray
    
    # Normally use object identity to detect array values changing:
    metadata.setdefault( 'rich_compare', False )
    
    if type( typecode ) in SequenceTypes:
        shape, typecode = typecode, shape
        
    if (typecode is not None) and (typecode not in Typecodes):
        raise TraitError, "typecode must be a valid Numeric typecode or None"
        
    if shape is not None:
        if isinstance( shape, SequenceTypes ):
            for item in shape:
                if ((item is None) or (type( item ) is int) or
                    (isinstance( item, SequenceTypes ) and 
                     (len( item ) == 2) and
                     (type( item[0] ) is int) and (item[0] >= 0) and
                     ((item[1] is None) or ((type( item[1] ) is int) and
                       (item[0] <= item[1]))))):
                    continue
                raise TraitError, "shape should be a list or tuple"
        else:
            raise TraitError, "shape should be a list or tuple"
            
        if (len( shape ) == 2) and (metadata.get( 'editor' ) is None):
            from neuroimaging.extra.enthought.traits.ui import ArrayEditor
            metadata[ 'editor' ] = ArrayEditor
            
    if value is None:
        if shape is None:
            value = zeros( ( 0, ), typecode )
        else:
            size = []
            for item in shape:
                if item is None:
                    item = 1
                elif type( item ) in SequenceTypes:
                    item = item[0]
                size.append( item )
            value = zeros( size, typecode )
            
    return Trait( value, TraitArray( typecode, shape, coerce ), **metadata )
                  
#--- 'instance' traits ---------------------------------------------------------
    
def Instance ( klass, args = None, kw = None, allow_none = False, **metadata ):
    """ Values must be instances of the same type (or a subclass) of the
    specified *klass*, which can be a class or an instance.
    
    Parameters
    ----------
    klass : class or instance
        The object that forms the basis for the trait
    args : tuple
        Arguments for generating the default value
    kw : dictionary
        Keyword arguments for generating the default value
    allow_none : boolean
        Flag indicating whether **None** is allowed as a value
        
    Description
    -----------
    If neither *args* nor *kw* is specified, **None** is the default value and 
    an allowed value. Otherwise, the default value is obtained by calling 
    ``type(*args, **kw)``. Note that the constructor call is performed each time
    a default value is assigned, so each default value assigned is a unique 
    instance. If Instance is used by itself (i.e., not called), then only values
    of type **types.InstanceType** (i.e., old style class instances) are allowed.
    In this case the default value is **None**.
    """
    metadata.setdefault( 'copy', 'deep' )
    metadata.setdefault( 'instance_handler', '_instance_changed_handler' )
    ti_klass = TraitInstance( klass, or_none = allow_none,
                   module = stack(1)[1][0].f_globals.get( '__name__' ) )
    if (args is None) and (kw is None):
        return Trait( None, ti_klass, **metadata )
    if kw is None:
        if type( args ) is dict:
            kw   = args
            args = ()
    elif type( kw ) is not dict:
        raise TraitError, "The 'kw' argument must be a dictionary"
    if type( args ) is not tuple:
        return Trait( args, ti_klass, **metadata )
    return Trait( _InstanceArgs( args, kw ), ti_klass, **metadata )
    
class _InstanceArgs ( object ):

    def __init__ ( self, args, kw ):
        self.args = args
        self.kw   = kw
    
def WeakRef ( klass = 'neuroimaging.extra.enthought.traits.HasTraits', allow_none = False, 
              **metadata ):
    """ Values must be instances of the same type (or a subclass) of the
    specified *klass*, which can be a class or an instance. If *klass* is
    omitted, then values must be an instance of HasTraits.
    
    Parameters
    ----------
    klass : class or instance
        The object that forms the basis for the trait
    allow_none : boolean
        Flag indicating whether **None** is allowed as a value
        
    Description
    -----------
    Only a weak reference is maintained to any object assigned to a WeakRef
    trait. If no other references exist to the assigned value, the value may
    be garbage collected, in which case the value of the trait becomes None.
    In all other cases, the value returned by the trait is the original object.
    The default value is always **None**.
    """
    metadata.setdefault( 'copy', 'ref' )
    ti_klass       = TraitWeakRef( klass, or_none = allow_none,
                          module = stack(1)[1][0].f_globals.get( '__name__' ) )
    trait          = CTrait( 4 )
    trait.__dict__ = metadata.copy()
    trait.property( ti_klass._get,     _arg_count( ti_klass._get ),
                    ti_klass._set,     _arg_count( ti_klass._set ),
                    ti_klass.validate, _arg_count( ti_klass.validate ) )
    return trait
    
WeakRef = TraitFactory( WeakRef )    
        
#--- 'creates a run-time default value' ----------------------------------------

class Default ( object ):
    """ Generates a value the first time it is accessed.
    
    A Default object can be used anywhere a default trait value would normally
    be specified, to generate a default value dynamically.
    """
    def __init__ ( self, func = None, args = (), kw = None ):
        self.default_value = ( func, args, kw )
        
#--- 'string' traits -----------------------------------------------------------

def Regex ( value = '', regex = '.*', **metadata ):
    return Trait( value, TraitString( regex = regex ), **metadata )
    
Regex = TraitFactory( Regex )    

def String ( value = '', minlen = 0, maxlen = sys.maxint, regex = '', 
             **metadata ):
    return Trait( value, TraitString( minlen = minlen, 
                                      maxlen = maxlen, 
                                      regex  = regex ), **metadata )    
    
String = TraitFactory( String )    

def Code ( value = '', minlen = 0, maxlen = sys.maxint, regex = '', 
               **metadata ):
    if 'editor' not in metadata:
        metadata[ 'editor' ] = code_editor
    return Trait( value, TraitString( minlen = minlen, maxlen = maxlen, 
                                      regex  = regex ), **metadata )    

Code = TraitFactory( Code )

def HTML ( value = '', **metadata ):
    if 'editor' not in metadata:
        metadata[ 'editor' ] = html_editor
    return Trait( value, TraitString(), **metadata )    

HTML = TraitFactory( HTML )                  

def Password ( value = '', minlen = 0, maxlen = sys.maxint, regex = '', 
               **metadata ):
    if 'editor' not in metadata:
        metadata[ 'editor' ] = password_editor
    return Trait( value, TraitString( minlen = minlen, maxlen = maxlen, 
                                      regex  = regex ), **metadata )    

Password = TraitFactory( Password )   

def Expression ( value = '0', **metadata ):
    return Trait( value, TraitExpression(), **metadata )
    
Expression = TraitFactory( Expression )    

def PythonValue ( value = None, **metadata ):
    if 'editor' not in metadata:
        metadata[ 'editor' ] = shell_editor()
    return Any( value, **metadata )
    
PythonValue = TraitFactory( PythonValue ) 
    
#--- 'file' traits -----------------------------------------------------------

def File ( value = '', filter = None, auto_set = False, **metadata ):
    from neuroimaging.extra.enthought.traits.ui.editors import FileEditor
    
    return Trait( value, editor = FileEditor( filter   = filter or [],
                                              auto_set = auto_set ),
                  **metadata )
    
File = TraitFactory( File )    

def Directory ( value = '', auto_set = False, **metadata ):
    from neuroimaging.extra.enthought.traits.ui.editors import DirectoryEditor
    
    return Trait( value, editor = DirectoryEditor( auto_set = auto_set ),
                  **metadata )
    
Directory = TraitFactory( Directory )    
                  
#-------------------------------------------------------------------------------
#  Factory function for creating range traits:
#-------------------------------------------------------------------------------
    
def Range ( low = None, high = None, value = None, 
            exclude_low = False, exclude_high = False, **metadata ):
    if value is None:
        if low is not None:
            value = low
        else:
            value = high
    return Trait( value, TraitRange( low, high, exclude_low, exclude_high ), 
                  **metadata )
    
Range = TraitFactory( Range )    
                  
#-------------------------------------------------------------------------------
#  Factory function for creating enumerated value traits:
#-------------------------------------------------------------------------------
    
def Enum ( *values, **metadata ):
    dv = values[0]
    if (len( values ) == 2) and (type( values[1] ) in SequenceTypes):
        values = values[1]
    return Trait( dv, TraitEnum( *values ), **metadata )
    
Enum = TraitFactory( Enum )    
    
#-------------------------------------------------------------------------------
#  Factory function for creating constant traits:
#-------------------------------------------------------------------------------
    
def Constant ( value, **metadata ):
    """ Creates a read-only trait whose value is *value*. 
    
    Traits of this type are very space efficient (and fast) because *value* is
    not stored in each instance using the trait, but only in the trait itself.
    The *value* cannot be a list or dictionary, because those types have
    mutable values.
    """
    if type( value ) in MutableTypes:
        raise TraitError, \
              "Cannot define a constant using a mutable list or dictionary"
    metadata[ 'type' ] = 'constant'
    return Trait( value, **metadata )

#-------------------------------------------------------------------------------
#  Factory function for creating C-based events:
#-------------------------------------------------------------------------------

def Event ( *value_type, **metadata ):
    metadata[ 'type' ] = 'event';
    result = Trait( *value_type, **metadata )
    if 'instance_handler' in result.__dict__:
        del result.instance_handler
    return result
    
Event = TraitFactory( Event )   

def Button ( label = '', image = None, style = 'button', 
             orientation = 'vertical', width_padding = 7, height_padding = 5,
             **metadata ):
    from neuroimaging.extra.enthought.traits.ui import ButtonEditor
    
    return Event( editor = ButtonEditor(
                               label          = label,
                               image          = image, 
                               style          = style,
                               orientation    = orientation, 
                               width_padding  = width_padding,
                               height_padding = height_padding,
                               **metadata ) )

Button = TraitFactory( Button )

def ToolbarButton ( label = '', image = None, style = 'toolbar', 
                    orientation = 'vertical', width_padding = 2, 
                    height_padding = 2, **metadata ):
    return Button( label, image, style, orientation, width_padding,
                   height_padding, **metadata )

ToolbarButton = TraitFactory( ToolbarButton )

def UIDebugger ( **metadata ):
    from neuroimaging.extra.enthought.traits.ui.wx.ui_debug_editor import ToolkitEditorFactory
    
    return Event( editor = ToolkitEditorFactory(), **metadata )
    
UIDebugger = TraitFactory( UIDebugger )    
    
#  Handle circular module dependencies:
trait_handlers.Event = Event    

#-------------------------------------------------------------------------------
#  Factory function for creating C-based traits:
#-------------------------------------------------------------------------------

def Trait ( *value_type, **metadata ):
    """ Creates a trait definition.
    """
    return _TraitMaker( *value_type, **metadata ).as_ctrait()
       
#  Handle circular module dependencies:
trait_handlers.Trait = Trait       

#-------------------------------------------------------------------------------
#  '_TraitMaker' class:
#-------------------------------------------------------------------------------

class _TraitMaker ( object ):
    
    # Ctrait type map for special trait types:
    type_map = {
       'event':    2,
       'constant': 7
    }
 
    #---------------------------------------------------------------------------
    #  Initialize the object:
    #---------------------------------------------------------------------------
 
    def __init__ ( self, *value_type, **metadata ):
        metadata.setdefault( 'type', 'trait' )
        self.define( *value_type, **metadata )
 
    #---------------------------------------------------------------------------
    #  Define the trait:
    #---------------------------------------------------------------------------
        
    def define ( self, *value_type, **metadata ):
        default_value_type = -1
        default_value      = handler = clone = None
        if len( value_type ) > 0:
            default_value = value_type[0]
            value_type    = value_type[1:]
            if ((len( value_type ) == 0) and 
                (type( default_value ) in SequenceTypes)):
                default_value, value_type = default_value[0], default_value
            if len( value_type ) == 0:
                if isinstance( default_value, TraitFactory ):
                    default_value = trait_factory( default_value )
                if default_value in PythonTypes:
                    handler       = TraitType( default_value )
                    default_value = DefaultValues.get( default_value )
                elif isinstance( default_value, CTrait ):
                    clone = default_value
                    default_value_type, default_value = clone.default_value()
                    metadata[ 'type' ] = clone.type
                elif isinstance( default_value, TraitHandler ):
                    handler       = default_value
                    default_value = None
                elif default_value is ThisClass:
                    handler       = ThisClass()
                    default_value = None
                else:
                    typeValue = type( default_value )
                    if typeValue is StringType:
                        string_options = self.extract( metadata, 'min_len',
                                                       'max_len', 'regex' )
                        if len( string_options ) == 0:
                            handler = TraitCastType( typeValue )
                        else:
                            handler = TraitString( **string_options )
                    elif typeValue in TypeTypes:
                        handler = TraitCastType( typeValue )
                    else:
                        metadata.setdefault( 'instance_handler', 
                                             '_instance_changed_handler' )
                        handler = TraitInstance( default_value )
                        if default_value is handler.aClass:
                            default_value = DefaultValues.get( default_value )
            else:
                enum  = []
                other = []
                map   = {}
                self.do_list( value_type, enum, map, other )
                if (((len( enum )  == 1) and (enum[0] is None)) and
                    ((len( other ) == 1) and 
                     isinstance( other[0], TraitInstance ))):
                    enum = []
                    other[0].allow_none()
                    metadata.setdefault( 'instance_handler',
                                         '_instance_changed_handler' )
                if len( enum ) > 0:
                    if (((len( map ) + len( other )) == 0) and
                        (default_value not in enum)):
                        enum.insert( 0, default_value )
                    other.append( TraitEnum( enum ) )
                if len( map ) > 0:
                    other.append( TraitMap( map ) )
                if len( other ) == 0:
                    handler = TraitHandler()
                elif len( other ) == 1:
                    handler = other[0]
                    if isinstance( handler, CTrait ):
                        clone, handler = handler, None
                        metadata[ 'type' ] = clone.type
                    elif isinstance( handler, TraitInstance ):
                        metadata.setdefault( 'instance_handler', 
                                             '_instance_changed_handler' )
                        if default_value is None:
                            handler.allow_none()
                        elif isinstance( default_value, _InstanceArgs ):
                            default_value_type = 7
                            default_value = ( handler.create_default_value, 
                                default_value.args, default_value.kw ) 
                        elif (len( enum ) == 0) and (len( map ) == 0):
                            aClass    = handler.aClass
                            typeValue = type( default_value )
                            if typeValue is dict:
                                default_value_type = 7
                                default_value = ( aClass, (), default_value )
                            elif not isinstance( default_value, aClass ):
                                if typeValue is not tuple:
                                    default_value = ( default_value, )
                                default_value_type = 7
                                default_value = ( aClass, default_value, None )
                else:
                    for i, item in enumerate( other ):
                        if isinstance( item, CTrait ):
                            if item.type != 'trait':
                                raise TraitError, ("Cannot create a complex "
                                    "trait containing %s trait." % 
                                    add_article( item.type ) )
                            handler = item.handler
                            if handler is None:
                                break
                            other[i] = handler
                    else:
                        handler = TraitCompound( other )
 
        # Save the results:
        self.handler = handler
        self.clone   = clone
        if default_value_type < 0:
            if isinstance( default_value, Default ):
                default_value_type = 7
                default_value      = default_value.default_value
            else:
                if (handler is None) and (clone is not None):
                    handler = clone.handler
                if handler is not None:
                    default_value_type = handler.default_value_type
                    if default_value_type >= 0:
                        if hasattr( handler, 'default_value' ):
                            default_value = handler.default_value(default_value)
                    else:
                        try:
                            default_value = handler.validate( None, '',
                                                              default_value )
                        except:
                            pass
                if default_value_type < 0:
                    default_value_type = _default_value_type( default_value )
        self.default_value_type = default_value_type
        self.default_value      = default_value
        self.metadata           = metadata.copy()
 
    #---------------------------------------------------------------------------
    #  Determine the correct TraitHandler for each item in a list:
    #---------------------------------------------------------------------------
 
    def do_list ( self, list, enum, map, other ):
        for item in list:
            if item in PythonTypes:
                other.append( TraitType( item ) )
            else:
                if isinstance( item, TraitFactory ):
                    item = trait_factory( item )
                typeItem = type( item )
                if typeItem in ConstantTypes:
                    enum.append( item )
                elif typeItem in SequenceTypes:
                    self.do_list( item, enum, map, other )
                elif typeItem is DictType:
                    map.update( item )
                elif typeItem in CallableTypes:
                    other.append( TraitFunction( item ) )
                elif item is ThisClass:
                    other.append( ThisClass() )
                elif isinstance( item, TraitTypes ):
                    other.append( item )
                else:
                    other.append( TraitInstance( item ) )
                  
    #---------------------------------------------------------------------------
    #  Returns a properly initialized 'CTrait' instance:
    #---------------------------------------------------------------------------
                  
    def as_ctrait ( self ):
        metadata = self.metadata
        trait    = CTrait( self.type_map.get( metadata.get( 'type' ), 0 ) )
        clone    = self.clone
        if clone is not None:
            trait.clone( clone )
            if clone.__dict__ is not None:
                trait.__dict__ = clone.__dict__.copy()
        trait.default_value( self.default_value_type, self.default_value )
        handler = self.handler
        if handler is not None:
            trait.handler = handler
            if hasattr( handler, 'fast_validate' ):
                trait.set_validate( handler.fast_validate )
            else:
                trait.set_validate( handler.validate )
            if hasattr( handler, 'post_setattr' ):
                trait.post_setattr = handler.post_setattr
        trait.rich_comparison( metadata.get( 'rich_compare', True ) )
        if len( metadata ) > 0:
            if trait.__dict__ is None:
                trait.__dict__ = metadata
            else:
                trait.__dict__.update( metadata )
        return trait
       
    #---------------------------------------------------------------------------
    #  Extract a set of keywords from a dictionary:
    #---------------------------------------------------------------------------
           
    def extract ( self, from_dict, *keys ):
        to_dict = {}
        for key in keys:
            if key in from_dict:
                to_dict[ key ] = from_dict[ key ]
                del from_dict[ key ]
        return to_dict
       
#-------------------------------------------------------------------------------
#  Factory function for creating traits with standard Python behavior:
#-------------------------------------------------------------------------------

def TraitPython ( **metadata ):
    metadata.setdefault( 'type', 'python' )
    trait = CTrait( 1 )
    trait.default_value( 0, Undefined )
    trait.__dict__ = metadata.copy()
    return trait

#-------------------------------------------------------------------------------
#  Factory function for creating C-based trait delegates:
#-------------------------------------------------------------------------------

def Delegate ( delegate, prefix = '', modify = False, **metadata ):
    """ Creates a *delegator* trait, whose definition and default value are
    delegated to a *delegate* trait attribute on another object. 
    
    Parameters
    ----------
    delegate : string
        name of the attribute on the current object that references the object 
        that is the trait's delegate
    prefix : string 
        a prefix or substitution applied to the original attribute when looking
        up the delegated attribute
    modify : boolean
        Flag indicating whether changes are made to the delegate attribute, 
        rather than to the delegator attribute
        
    Description
    -----------
    An object containing a delegator trait attribute must contain a second 
    attribute that references the object containing the delegate trait attribute.
    The name of this second attribute is passed as the *delegate* argument to 
    the Delegate() function.
    
    The following rules govern the application of the prefix parameter:
        
        * If *prefix* is empty or omitted, the delegation is to an attribute of
          the delegate object with the same name as the delegator attribute.
        * If *prefix* is a valid Python attribute name, then the delegation is 
          to an attribute whose name is the value of *prefix*.
        * If *prefix* ends with an asterisk ('*') and is longer than one 
          character, then the delegation is to an attribute whose name is the 
          value of *prefix*, minus the trailing asterisk, prepended to the 
          delegator attribute name.
        * If *prefix* is equal to a single asterisk, the delegation is to an 
          attribute whose name is the value of the delegator object's 
          __prefix__ attribute prepended to delegator attribute name.
          
    If the *modify* parameter is ** True, then any changes to the delegator
    attribute are actually applied to the delegate attribute.
    
    """
    metadata.setdefault( 'type', 'delegate' )
    if prefix == '':
        prefix_type = 0
    elif prefix[-1:] != '*':
        prefix_type = 1
    else:
        prefix = prefix[:-1]
        if prefix != '':
            prefix_type = 2
        else:
            prefix_type = 3
    trait = CTrait( 3 )
    trait.delegate( delegate, prefix, prefix_type, modify )
    trait.__dict__ = metadata.copy()
    return trait
       
#-------------------------------------------------------------------------------
#  Factory function for creating C-based trait properties:
#-------------------------------------------------------------------------------
        
def Property ( fget = None, fset = None, fvalidate = None, force = False,
               handler = None, trait = None, **metadata ):
    metadata[ 'type' ] = 'property'
    
    # If no parameters specified, must be a forward reference (if not forced):
    if (not force) and (fset is None) and (fvalidate is None):
        if (fget is None) and (trait is None):
            return ForwardProperty( metadata )
        if (fget is None) ^ (trait is None):
            if fget is not None:
                trait = fget
            trait = trait_cast( trait )
            if trait is not None:
                fvalidate = handler = trait.handler
                if fvalidate is not None:
                    fvalidate = handler.validate
                if 'editor' not in metadata:
                    editor = trait.editor
                    if editor is not None:
                        metadata[ 'editor' ] = editor
                return ForwardProperty( metadata, fvalidate, handler ) 
        
    if fget is None: 
        if fset is None:
            fget = _undefined_get
            fset = _undefined_set
        else:
            fget = _write_only
    elif fset is None: 
        fset = _read_only
        
    if trait is not None:
        trait   = trait_cast( trait )
        handler = trait.handler
        if (fvalidate is None) and (handler is not None):
            fvalidate = handler.validate
        if ('editor' not in metadata) and (trait.editor is not None):
            metadata[ 'editor' ] = trait.editor
        
    n     = 0
    trait = CTrait( 4 )
    trait.__dict__ = metadata.copy()
    if fvalidate is not None:
        n = _arg_count( fvalidate )
    trait.property( fget,      _arg_count( fget ),
                    fset,      _arg_count( fset ),
                    fvalidate, n )
    trait.handler = handler
    return trait
    
Property = TraitFactory( Property )    

class ForwardProperty ( object ):
    
    def __init__ ( self, metadata, validate = None, handler = None ):
        self.metadata = metadata.copy()
        self.validate = validate
        self.handler  = handler
    
#-------------------------------------------------------------------------------
#  Property error handling functions:
#-------------------------------------------------------------------------------
    
def _write_only ( object, name ):
    raise TraitError, "The '%s' trait of %s instance is 'write only'." % ( 
                      name, class_of( object ) )
    
def _read_only ( object, name, value ):
    raise TraitError, "The '%s' trait of %s instance is 'read only'." % ( 
                      name, class_of( object ) )
    
def _undefined_get ( object, name ):
    raise TraitError, ("The '%s' trait of %s instance is a property that has "
                       "no 'get' or 'set' method") % ( 
                       name, class_of( object ) )
    
def _undefined_set ( object, name, value ):
    _undefined_get( object, name )

#-------------------------------------------------------------------------------
#  Dictionary used to handler return type mapping special cases:
#-------------------------------------------------------------------------------

SpecialNames = {
   'int':     trait_factory( Int ),
   'long':    trait_factory( Long ),
   'float':   trait_factory( Float ),
   'complex': trait_factory( Complex ),
   'str':     trait_factory( Str ),
   'unicode': trait_factory( Unicode ),
   'bool':    trait_factory( Bool ),
   'list':    trait_factory( List ),
   'tuple':   trait_factory( Tuple ),
   'dict':    trait_factory( Dict )
}
    
#-------------------------------------------------------------------------------
#  Create predefined, reusable trait instances:
#-------------------------------------------------------------------------------

false           = Bool                  # Synonym for Bool
true            = Bool( True )          # Boolean values only, default = True
Function        = Trait( FunctionType ) # Function values only
Method          = Trait( MethodType )   # Method values only
Class           = Trait( ClassType )    # Class values only
Module          = Trait( ModuleType )   # Module values only
Type            = Trait( TypeType )     # Type values only
This            = Trait( ThisClass )    # Containing class values only
self            = Trait( Self, ThisClass ) # Same as above, default = self
Python          = TraitPython()         # Standard Python value
Disallow        = CTrait( 5 )           # Disallow read/write access  
ReadOnly        = CTrait( 6 )           # Read only access (i.e. 'write once')
ReadOnly.default_value( 0, Undefined )  # This allows it to be written once
undefined       = Any( Undefined )      # No type checking, default = Undefined
missing         = CTrait( 0 )           # Define a missing parameter trait
missing.handler = TraitHandler()
missing.default_value( 1, Missing )
generic_trait   = CTrait( 8 )           # Generic trait with 'object' behavior
Callable        = Trait( TraitCallable(), copy = 'ref' ) # Callable values 

# List traits:
ListInt        = List( int )           # List of int values
ListFloat      = List( float )         # List of float values
ListStr        = List( str )           # List of string values
ListUnicode    = List( unicode )       # List of Unicode string values
ListComplex    = List( complex )       # List of complex values
ListBool       = List( bool )          # List of boolean values
ListFunction   = List( FunctionType )  # List of function values
ListMethod     = List( MethodType )    # List of method values
ListClass      = List( ClassType )     # List of class values
ListInstance   = List( InstanceType )  # List of instance values
ListThis       = List( ThisClass )     # List of container type values

# Dictionary traits:
DictStrAny     = Dict( str, Any )      # Dict of string: any values
DictStrStr     = Dict( str, str )      # Dict of string: string values
DictStrInt     = Dict( str, int )      # Dict of string: integer values
DictStrLong    = Dict( str, long )     # Dict of string: long int values
DictStrFloat   = Dict( str, float )    # Dict of string: float values
DictStrBool    = Dict( str, bool )     # Dict of string: boolean values
DictStrList    = Dict( str, list )     # Dict of string: list values

#-------------------------------------------------------------------------------
#  User interface related color and font traits:
#-------------------------------------------------------------------------------

def Color ( *args, **metadata ):
    from neuroimaging.extra.enthought.traits.ui import ColorTrait
    return ColorTrait( *args, **metadata )
    
Color = TraitFactory( Color )

def RGBColor ( *args, **metadata ):
    from neuroimaging.extra.enthought.traits.ui import RGBColorTrait
    return RGBColorTrait( *args, **metadata )
    
RGBColor = TraitFactory( RGBColor )

def RGBAColor ( *args, **metadata ):
    from neuroimaging.extra.enthought.traits.ui import RGBAColorTrait
    return RGBAColorTrait( *args, **metadata )
    
RGBAColor = TraitFactory( RGBAColor )
    
def Font ( *args, **metadata ):
    from neuroimaging.extra.enthought.traits.ui import FontTrait
    return FontTrait( *args, **metadata )
    
Font = TraitFactory( Font )

def KivaFont ( *args, **metadata ):
    from neuroimaging.extra.enthought.traits.ui import KivaFontTrait
    return KivaFontTrait( *args, **metadata )
    
KivaFont = TraitFactory( KivaFont )
    
