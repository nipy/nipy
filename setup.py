#!/usr/bin/env python3
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import os
from os.path import join as pjoin, exists
from glob import glob
from distutils import log

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if exists('MANIFEST'): os.remove('MANIFEST')

# Always use setuptools
import setuptools

from setup_helpers import (generate_a_pyrex_source, get_comrec_build,
                           cmdclass, INFO_VARS)

# monkey-patch numpy distutils to use Cython instead of Pyrex
from numpy.distutils.command.build_src import build_src
build_src.generate_a_pyrex_source = generate_a_pyrex_source

# Add custom commit-recording build command
from numpy.distutils.command.build_py import build_py as _build_py
cmdclass['build_py'] = get_comrec_build('nipy', _build_py)

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
    config.get_version(pjoin('nipy', 'info.py')) # sets config.version
    config.add_subpackage('nipy', 'nipy')
    return config

################################################################################

# Hard and soft dependency checking
DEPS = (
    ('numpy', INFO_VARS['NUMPY_MIN_VERSION'], 'setup_requires'),
    ('scipy', INFO_VARS['SCIPY_MIN_VERSION'], 'install_requires'),
    ('nibabel', INFO_VARS['NIBABEL_MIN_VERSION'], 'install_requires'),
    ('sympy', INFO_VARS['SYMPY_MIN_VERSION'], 'install_requires'),
)

requirement_kwargs = {'setup_requires': [], 'install_requires': []}
for name, min_ver, req_type in DEPS:
    new_req = '{0}>={1}'.format(name, min_ver)
    requirement_kwargs[req_type].append(new_req)


################################################################################
# commands for installing the data
from numpy.distutils.command.install_data import install_data
from numpy.distutils.command.build_ext import build_ext

def data_install_msgs():
    # Check whether we have data packages
    try:  # Allow setup.py to run without nibabel
        from nibabel.data import datasource_or_bomber
    except ImportError:
        log.warn('Cannot check for optional data packages: see: '
                 'http://nipy.org/nipy/users/install_data.html')
        return
    DATA_PKGS = INFO_VARS['DATA_PKGS']
    templates = datasource_or_bomber(DATA_PKGS['nipy-templates'])
    example_data = datasource_or_bomber(DATA_PKGS['nipy-data'])
    for dpkg in (templates, example_data):
        if hasattr(dpkg, 'msg'): # a bomber object, warn
            log.warn('%s\n%s' % ('_'*80, dpkg.msg))


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


def main():
    from numpy.distutils.core import setup

    setup(name=INFO_VARS['NAME'],
          maintainer=INFO_VARS['MAINTAINER'],
          maintainer_email=INFO_VARS['MAINTAINER_EMAIL'],
          description=INFO_VARS['DESCRIPTION'],
          long_description=INFO_VARS['LONG_DESCRIPTION'],
          url=INFO_VARS['URL'],
          download_url=INFO_VARS['DOWNLOAD_URL'],
          license=INFO_VARS['LICENSE'],
          classifiers=INFO_VARS['CLASSIFIERS'],
          author=INFO_VARS['AUTHOR'],
          author_email=INFO_VARS['AUTHOR_EMAIL'],
          platforms=INFO_VARS['PLATFORMS'],
          # version set by config.get_version() above
          configuration = configuration,
          cmdclass = cmdclass,
          tests_require=['nose>=1.0'],
          test_suite='nose.collector',
          zip_safe=False,
          entry_points={
              'console_scripts': [
                  'nipy_3dto4d = nipy.cli.img3dto4d:main',
                  'nipy_4dto3d = nipy.cli.img4dto3d:main',
                  'nipy_4d_realign = nipy.cli.realign4d:main',
                  'nipy_tsdiffana = nipy.cli.tsdiffana:main',
                  'nipy_diagnose = nipy.cli.diagnose:main',
              ],
          },
          **requirement_kwargs)


if __name__ == "__main__":
    main()
