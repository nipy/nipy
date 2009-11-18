#!/usr/bin/env python

def configuration(parent_package='',top_path=None):
    
    from numpy.distutils.misc_util import Configuration

    config = Configuration('segment', parent_package, top_path)
    config.add_data_dir('tests')
    config.add_data_dir('benchmarks')
    config.add_extension('mrf_module', sources=['mrf_module.c'])

    return config


if __name__ == '__main__':
    print 'This is the wrong setup.py file to run'


