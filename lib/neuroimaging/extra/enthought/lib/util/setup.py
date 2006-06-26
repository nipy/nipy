#!/usr/bin/env python
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

minimum_numpy_version = '0.9.7.2401'
def configuration(parent_package='',top_path=None):
    import numpy
    if numpy.__version__ < minimum_numpy_version:
        raise RuntimeError, 'numpy version %s or higher required, but got %s'\
              % (minimum_numpy_version, numpy.__version__)

    from numpy.distutils.misc_util import Configuration
    config = Configuration('util',parent_package,top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('distribution')
    config.add_subpackage('distribution.editor')
    config.add_data_dir('distribution/editor/tests')
    config.add_subpackage('traits')
    config.add_subpackage('traits.editor')
    config.add_subpackage('wx')
    config.add_subpackage('wx.spreadsheet')
    config.add_data_dir('test')
    return config

if __name__ == "__main__":
    # Remove current working directory from sys.path
    # to avoid importing math as Python math module:
    import os, sys
    for cwd in ['','.',os.getcwd()]:
        while cwd in sys.path: sys.path.remove(cwd)

    try:
        from numpy.distutils.core import setup
    except ImportError:
        execfile('setup_util.py')
    else:
        setup(configuration=configuration)
