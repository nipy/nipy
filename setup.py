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


def get_nipy_info():
    ''' Fetch NIPY info from NIPY setup file
    
    Get from nipy.setup without importing (as the package is not yet
    installed
    '''
    ns = dict(__name__='')
    execfile(os.path.join('nipy', 'setup.py'), ns)
    return ns['get_nipy_info']()


def install_data_tgz():
    """ Check if the data is there, install from tarball if not
    """
    for dkey, tarname, descrip in (('template_dir',
                                    'nipy_templates.tar.gz',
                                    'templates'),
                                   ('example_data_dir',
                                    'nipy_example_data.tar.gz',
                                    'example data')):
        data_dir = get_nipy_info()[dkey]
        if not os.path.exists(os.path.expanduser(data_dir)):
            if os.path.exists(tarname):
                log.info('extracting data tarball to %s' % data_dir)
                tar = tarfile.open(tarname)
                tar.extractall(data_dir)
                tar.close()
                continue
            msg = """
We did not find the nipy data directory '%(data_dir)s'.
Neither could we find the archive '%(tarname)s' in the current directory.
If you want the NIPY %(descrip)s please download

https://cirl.berkeley.edu/nipy/%(tarname)s

and run

python setup.py data_install

in this directory
""" % locals()
            log.warn(80*"_" + msg + 80*"_")


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


class DataInstall(Command):
    description = 'unpack templates and example data'
    user_options = [('None', None, 'this command has no options')]
    
    def run(self):
        install_data_tgz()
        
    def initialize_options(self):
        pass
    
    def finalize_options(self):
        pass
    


cmdclass['data_install'] = DataInstall
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
