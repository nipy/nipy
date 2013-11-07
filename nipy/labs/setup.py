# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import os
from distutils import log


def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
    from numpy.distutils.system_info import get_info

    config = Configuration('labs', parent_package, top_path)

    # Subpackages
    config.add_subpackage('spatial_models')
    config.add_subpackage('utils')
    config.add_subpackage('viz_tools')
    config.add_subpackage('datasets')
    config.add_subpackage('tests')

    config.make_config_py() # installs __config__.py

    return config

if __name__ == '__main__':
    print('This is the wrong setup.py file to run')
