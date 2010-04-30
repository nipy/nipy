import os 

def configuration(parent_package='',top_path=None):
    
    from numpy.distutils.misc_util import Configuration

    config = Configuration('image', parent_package, top_path)
    config.add_data_dir('tests')
    config.add_data_dir('benchmarks')
    config.add_include_dirs(config.name.replace('.', os.sep))
    config.add_extension('image_module', sources=['image_module.pyx', 'cubic_spline.c'])

    print config 

    return config


if __name__ == '__main__':
    print 'This is the wrong setup.py file to run'


