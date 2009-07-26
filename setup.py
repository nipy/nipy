#!/usr/bin/env python
import sys
import os
import tarfile
import tempfile
from distutils import log
from distutils.cmd import Command

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


################################################################################
# Import the documentation building classes. 

try:
    from build_docs import cmdclass
except ImportError:
    """ Pass by the doc build gracefully if sphinx is not installed """
    print "Sphinx is not installed, docs cannot be built"
    cmdclass = {}


################################################################################
# commands for installing the data
from distutils.command.install import install
from numpy.distutils.command.build_ext import build_ext

def data_install_msgs():
    from nipy.utils import make_datasource, DataError
    try:
        make_datasource('nipy', 'templates')
    except DataError, exception:
        print '#' * 80
        print exception
    try:
        make_datasource('nipy', 'data')
    except DataError, exception:
        print '#' * 80
        print exception
        


class MyInstall(install):
    """ Subclass the install to generate data install warnings if necessary
    """
    def run(self):
        install.run(self)
        data_install_msgs()


cmdclass['install'] = MyInstall

################################################################################

# We need to import nipy as late as possible, 
from nipy import  __doc__

def main(**extra_args):
    from numpy.distutils.core import setup
    
    setup( name = 'nipy',
           description = 'This is a neuroimaging python package',
           author = 'Various',
           author_email = 'nipy-devel@neuroimaging.scipy.org',
           url = 'http://neuroimaging.scipy.org',
           long_description = __doc__,
           configuration = configuration,
           cmdclass = cmdclass,
           **extra_args)


if __name__ == "__main__":
    main(**extra_setuptools_args)
