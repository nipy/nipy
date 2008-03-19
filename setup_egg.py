from setuptools import setup
execfile('setup.py') #,
#         {'additional_params' :
#         {'namespace_packages' : ['mpl_toolkits']}})

#from os.path import join
#import sys
#from setuptools import setup, Extension
#
#from neuroimaging import packages, __version__, __doc__, ENTHOUGHT_TRAITS_DEF
#
#def main(packages):
#        
#    packages = ['']+list(packages)
#    ext_modules = [Extension('data_io.formats.minc._mincutils',
#      [join(*('neuroimaging/data_io/formats/minc/_mincutils.c'.split('/')))],
#      extra_link_args=["-lminc"],
#      include_dirs=get_numpy_include_dirs())]
#
#    package_dir = {'': '.'}
#
#    if not ENTHOUGHT_TRAITS_DEF:
#        ext_modules += [Extension('utils.enthought.traits.ctraits',
#          [join(*('neuroimaging/utils/enthought/traits/ctraits.c'.split('/')))])]
#        package_dir['neuroimaging.utils.enthought'] = \
#          join(*('neuroimaging/utils/enthought/'.split('/')))
#
#    setup( name = 'neuroimaging',
#           version = __version__,
#           description = 'This is a neuroimaging python package',
#           author = 'Various',
#           author_email = 'nipy-devel@neuroimaging.scipy.org',
#           ext_package = 'neuroimaging',
#           packages=packages,
#           ext_modules=ext_modules,
#           package_dir = package_dir,
#           url = 'http://neuroimaging.scipy.org',
#           long_description = __doc__)
#

if __name__ == "__main__":
    main(packages)
