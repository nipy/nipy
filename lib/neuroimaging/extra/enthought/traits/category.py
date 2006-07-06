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
# Date: 11/06/2004
# Description: Adds a Cocoa-like 'category' capability to Traits-base classes.
#------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from has_traits \
    import MetaHasTraits, MetaHasTraitsObject, BaseTraits, ClassTraits, \
           PrefixTraits, ViewTraits  

#-------------------------------------------------------------------------------
#  'MetaCategory' class:
#-------------------------------------------------------------------------------

class MetaCategory ( MetaHasTraits ):
    
    def __new__ ( cls, class_name, bases, class_dict ):

        # Make sure the correct usage is being applied:        
        if len( bases ) > 2:
            raise TypeError, \
                  "Correct usage is: class FooCategory(Category,Foo):"
                  
        # Process any traits-related information in the class dictionary:
        MetaCategoryObject( cls, class_name, bases, class_dict, True )
                  
        # Move all remaining items in our class dictionary to the base class's 
        # dictionary:
        if len( bases ) == 2:
            category_class = bases[1]
            for name, value in class_dict.items():
                if not hasattr( category_class, name ):
                    setattr( category_class, name, value )
                    del class_dict[ name ]
            
        # Finish building the class using the updated class dictionary:
        return type.__new__( cls, class_name, bases, class_dict )
        
#-------------------------------------------------------------------------------
#  'MetaCategoryObject' class:  
#-------------------------------------------------------------------------------
                
class MetaCategoryObject ( MetaHasTraitsObject ):
    
    #---------------------------------------------------------------------------
    #  Adds the traits meta-data to the class:   
    #---------------------------------------------------------------------------
           
    def add_traits_meta_data ( self, bases, class_dict, base_traits, 
                               class_traits, instance_traits, prefix_traits, 
                               view_elements ):
        if len( bases ) == 2:
            # Update the class and each of the existing subclasses:
            bases[1]._add_trait_category( base_traits, class_traits, 
                                 instance_traits, prefix_traits, view_elements )
        else:
            MetaHasTraitsObject.add_traits_meta_data( self, bases,
                   class_dict, base_traits, class_traits, instance_traits, 
                   prefix_traits, view_elements )  
        
#-------------------------------------------------------------------------------
#  'Category' class:
#-------------------------------------------------------------------------------

class Category ( object ):
    
    __metaclass__ = MetaCategory 

