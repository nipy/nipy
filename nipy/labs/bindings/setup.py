from __future__ import absolute_import
from __future__ import print_function
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('bindings', parent_package, top_path)
    config.add_subpackage('tests')
    config.add_subpackage('benchmarks')
    config.add_extension('linalg', sources=['linalg.pyx'],
                            libraries=['cstat'])
    config.add_extension('array', sources=['array.pyx'],
                            libraries=['cstat'])
    config.add_extension('wrapper', sources=['wrapper.pyx'],
                            libraries=['cstat'])
    return config


if __name__ == '__main__':
    print('This is the wrong setup.py file to run')
