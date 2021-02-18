from __future__ import absolute_import
from __future__ import print_function
#!/usr/bin/env python3
import numpy


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('graph', parent_package, top_path)
    config.add_subpackage('tests')
    config.add_extension('_graph',
                         sources=['_graph.c'],
                         include_dirs=[numpy.get_include()])
    return config


if __name__ == '__main__':
    print('This is the wrong setup.py file to run')
