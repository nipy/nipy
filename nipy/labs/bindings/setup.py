# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import os, sys

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('bindings', parent_package, top_path)
    config.add_subpackage('tests')
    config.add_subpackage('benchmarks')
    # We need this because libcstat.a is linked to lapack, which can
    # be a fortran library, and the linker needs this information.
    from numpy.distutils.system_info import get_info
    lapack_info = get_info('lapack_opt',0)
    if 'libraries' not in lapack_info:
        # But on OSX that may not give us what we need, so try with 'lapack'
        # instead.  NOTE: scipy.linalg uses lapack_opt, not 'lapack'...
        lapack_info = get_info('lapack',0)
    config.add_extension('linalg', sources=['linalg.pyx'],
                            libraries=['cstat'],
                            extra_info=lapack_info)
    config.add_extension('array', sources=['array.pyx'],
                            libraries=['cstat'],
                            extra_info=lapack_info)
    config.add_extension('wrapper', sources=['wrapper.pyx'],
                            libraries=['cstat'],
                            extra_info=lapack_info)
    return config


if __name__ == '__main__':
    print('This is the wrong setup.py file to run')
