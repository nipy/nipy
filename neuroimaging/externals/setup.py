from os.path import join

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('externals', parent_package, top_path)

    config.add_subpackage('enthought')
    config.add_extension('enthought.traits.ctraits',
          [join(*('enthought/traits/ctraits.c'.split('/')))])


    config.add_subpackage('config')

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
