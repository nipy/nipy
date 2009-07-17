#!/usr/bin/env python
import sys
import os
import tarfile
from distutils import log

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

def install_data_tgz():
    """ Check if the data tarball is there and install it.
    """
    # Grab from nipy.setup without importing (as the package is not yet
    # installed
    ns = dict(__name__='')
    execfile(os.path.join('nipy', 'setup.py'), ns)
    get_nipy_info = ns['get_nipy_info']
    data_dir = get_nipy_info()['data_dir']
    if not os.path.exists(os.path.expanduser(data_dir)):
        filename = 'nipy_data.tar.gz'
        if os.path.exists(filename):
            log.info('extracting data tarball to %s' % data_dir)
            tar = tarfile.open(filename)
            tar.extractall(data_dir)
            tar.close()
        else:
            log.warn(80*"_" + "\n"
                    "The nipy data file was not found. This install of "
                    "nipy does not contain import data files such as "
                    "templates. You can download "
                    "https://cirl.berkeley.edu/nipy/nipy_data.tar.gz "
                    "in this directory, and rerun the install to "
                    "set up the data\n" 
                    + 80*"_"
                    )


class MyInstall(install):
    """ Subclass the install to install also the data, if present.
    """
    def run(self):
        install.run(self)
        install_data_tgz()


class MyBuildExt(build_ext):
    """ Subclass the 'build_ext --inplace' to install the data, if present.
    """
    def run(self):
        build_ext.run(self)
        if self.inplace:
            install_data_tgz()


cmdclass['install'] = MyInstall
cmdclass['build_ext'] = MyBuildExt


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
