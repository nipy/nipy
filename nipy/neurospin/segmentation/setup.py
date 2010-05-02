import os

def configuration(parent_package='',top_path=None):
    
    from numpy.distutils.misc_util import Configuration

    config = Configuration('segmentation', parent_package, top_path)
    config.add_data_dir('tests')
    config.add_data_dir('benchmarks')
    config.add_include_dirs(config.name.replace('.', os.sep))
    config.add_extension('_mrf', sources=['_mrf.pyx', 'mrf.c'])

    return config


if __name__ == '__main__':
    print 'This is the wrong setup.py file to run'


