import os

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('neuroimaging', parent_package, top_path)

    config.add_subpackage('algorithms')
    config.add_subpackage('core')
    config.add_subpackage('externals')
    config.add_subpackage('fixes')
    config.add_subpackage('io')
    config.add_subpackage('modalities')
    config.add_subpackage('ui')
    config.add_subpackage('utils')
    config.add_subpackage('testing')
    config.add_data_dir('testing')

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
