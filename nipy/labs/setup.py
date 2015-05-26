# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import os
from distutils import log

# Global variables
LIBS = os.path.realpath('lib')


def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
    from numpy.distutils.system_info import get_info, system_info

    config = Configuration('labs', parent_package, top_path)

    # fff library
    config.add_include_dirs(os.path.join(LIBS,'fff'))
    config.add_include_dirs(os.path.join(LIBS,'fff_python_wrapper'))
    config.add_include_dirs(get_numpy_include_dirs())

    sources = [os.path.join(LIBS,'fff','*.c')]
    sources.append(os.path.join(LIBS,'fff_python_wrapper','*.c'))

    config.add_library('cstat', sources=sources)

    # Subpackages
    config.add_subpackage('bindings')
    config.add_subpackage('glm')
    config.add_subpackage('group')
    config.add_subpackage('spatial_models')
    config.add_subpackage('utils')
    config.add_subpackage('viz_tools')
    config.add_subpackage('datasets')
    config.add_subpackage('tests')

    config.make_config_py() # installs __config__.py

    return config

if __name__ == '__main__':
    print('This is the wrong setup.py file to run')
