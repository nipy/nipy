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
# Author: Enthought, Inc.
# Description: <Enthought resource package component>
#------------------------------------------------------------------------------
""" Default base-class for resource factories. """


class ResourceFactory:
    """ Default base-class for resource factories. """
    
    ###########################################################################
    # 'ResourceFactory' interface.
    ###########################################################################

    def image_from_file(self, filename):
        """ Creates an image from the data in the specified filename. """
        
        raise NotImplemented

    def image_from_data(self, data):
        """ Creates an image from the specified data. """
        
        raise NotImplemented
    
#### EOF ######################################################################
