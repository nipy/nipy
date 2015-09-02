#!/usr/bin/env python


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('utils', parent_package, top_path)
    config.add_subpackage('tests')
    config.add_extension(
                'routines',
                sources=['routines.pyx'],
                libraries=['cstat']
                )

    return config


if __name__ == '__main__':
    print('This is the wrong setup.py file to run')
