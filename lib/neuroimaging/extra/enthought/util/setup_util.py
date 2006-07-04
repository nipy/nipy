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
# Description: <Enthought util package component>
#------------------------------------------------------------------------------

from scipy_distutils.core      import setup, Extension
from scipy_distutils.misc_util import get_subpackages, dict_append, get_path
from scipy_distutils.misc_util import merge_config_dicts, default_config_dict
from scipy_distutils.misc_util import dot_join
import os



def configuration(parent_package='',parent_path=None):
    package_name = 'util'
    local_path = get_path(__name__,parent_path)

    config_dict = default_config_dict(package_name, parent_package)
    config_list = [config_dict]
    config_list += get_subpackages(local_path,
                                   parent=config_dict['name'],
                                   parent_path=parent_path,
                                   recursive=1
                                   )
    config_dict = merge_config_dicts(config_list)

    return config_dict

if __name__ == '__main__':
    setup(**configuration(parent_path=''))
