#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from os.path import join as pjoin
import sys
from glob import glob
from distutils import log
from distutils.cmd import Command

# monkey-patch numpy distutils to use Cython instead of Pyrex
from build_helpers import generate_a_pyrex_source, package_check, cmdclass
from numpy.distutils.command.build_src import build_src
build_src.generate_a_pyrex_source = generate_a_pyrex_source


def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)
    # The quiet=True option will silence all of the name setting warnings:
    # Ignoring attempt to set 'name' (from 'nipy.core' to 
    #    'nipy.core.image')
    # Robert Kern recommends setting quiet=True on the numpy list, stating
    # these messages are probably only used in debugging numpy distutils.
    config.get_version('nipy/version.py') # sets config.version
    config.add_subpackage('nipy', 'nipy')
    return config

################################################################################
# For some commands, use setuptools

if len(set(('develop', 'bdist_egg', 'bdist_rpm', 'bdist', 'bdist_dumb', 
            'bdist_wininst', 'install_egg_info', 'egg_info', 'easy_install',
            )).intersection(sys.argv)) > 0:
    from setup_egg import extra_setuptools_args

# extra_setuptools_args can be defined from the line above, but it can
# also be defined here because setup.py has been exec'ed from
# setup_egg.py.
if not 'extra_setuptools_args' in globals():
    extra_setuptools_args = dict()


# Hard and soft dependency checking
package_check('scipy', '0.5')
package_check('sympy', '0.6.6')
def _mayavi_version(pkg_name):
    from enthought.mayavi import version
    return version.version
package_check('mayavi', '3.0', optional=True,
              version_getter=_mayavi_version)
def _cython_version(pkg_name):
    from Cython.Compiler.Version import version
    return version
# Cython is a build dependency
package_check('cython', '0.11.1', optional=False,
              version_getter=_cython_version)
    

################################################################################
# commands for installing the data
from numpy.distutils.command.install_data import install_data
from numpy.distutils.command.build_ext import build_ext

def data_install_msgs():
    from nipy.utils import make_datasource, DataError
    for name in ('templates', 'data'):
        try:
            make_datasource('nipy', name)
        except DataError, exception:
            log.warn('%s\n%s' % ('_'*80, exception))
        

class MyInstallData(install_data):
    """ Subclass the install_data to generate data install warnings if necessary
    """
    def run(self):
        install_data.run(self)
        data_install_msgs()


class MyBuildExt(build_ext):
    """ Subclass the build_ext to generate data install warnings if
        necessary: warn at build == warn early
        This is also important to get a warning when run a 'develop'.
    """
    def run(self):
        build_ext.run(self)
        data_install_msgs()


cmdclass['install_data'] = MyInstallData
cmdclass['build_ext'] = MyBuildExt

################################################################################

# Get project related strings.  Please do not change this line to use
# execfile because execfile is not available in Python 3
info_fname = pjoin('nipy', 'info.py')
release_vars = {}
exec(open(info_fname, 'rt').read(), {}, release_vars)

def main(**extra_args):
    from numpy.distutils.core import setup
    
    setup( name = 'nipy',
           description = 'This is a neuroimaging python package',
           author = 'Various',
           author_email = 'nipy-devel@neuroimaging.scipy.org',
           url = 'http://neuroimaging.scipy.org',
           long_description = release_vars['long_description'],
           configuration = configuration,
           cmdclass = cmdclass,
           scripts = glob('scripts/*.py'),
           **extra_args)


if __name__ == "__main__":
    main(**extra_setuptools_args)
