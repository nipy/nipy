from neuroimaging import ENTHOUGHT_TRAITS_DEF


def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('utils', parent_package, top_path)

    config.add_data_dir('tests')


    config.add_subpackage('config')

    if not ENTHOUGHT_TRAITS_DEF:
        print "yo"
        config.add_subpackage('enthough')
        config.add_extension('enthought.traits.ctraits',
          [join(*('enthought/traits/ctraits.c'.split('/')))])


    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
