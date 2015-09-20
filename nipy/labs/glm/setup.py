#!/usr/bin/env python
from __future__ import absolute_import, print_function


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('glm', parent_package, top_path)
    config.add_subpackage('tests')
    config.add_subpackage('benchmarks')
    config.add_extension(
                'kalman',
                sources=['kalman.pyx'],
                libraries=['cstat']
                )
    return config


if __name__ == '__main__':
    print('This is the wrong setup.py file to run')
