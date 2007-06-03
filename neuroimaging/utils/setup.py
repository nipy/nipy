from os.path import join
from neuroimaging import ENTHOUGHT_TRAITS_DEF

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('utils', parent_package, top_path)

    # We want to unzip our data here where we have permissions
    # before moving it into the install location.
    from data_io.datasource import unzip
    import os
    path = "neuroimaging/utils/tests/data"
    for filename in os.listdir(path):
        if filename.endswith('bz2'):
            filename = os.path.join(path, filename)
            unzip(filename)
            os.remove(filename)


    config.add_data_dir('tests')


    config.add_subpackage('config')

    config.add_subpackage('enthought')
    config.add_extension('enthought.traits.ctraits',
          [join(*('enthought/traits/ctraits.c'.split('/')))])


    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
