#!/usr/bin/env python
import numpy


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('graph', parent_package, top_path)
    config.add_data_dir('tests')
    config.add_extension('_graph',
                         sources=['_graph.c'],
                         include_dirs=[numpy.get_include()])
    return config


if __name__ == '__main__':
    print 'This is the wrong setup.py file to run'
