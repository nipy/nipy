#!/usr/bin/env python3
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import os
import sys
import textwrap
import warnings
import sysconfig
import importlib
import subprocess
from os.path import join as pjoin, exists
from distutils import log

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if exists('MANIFEST'): os.remove('MANIFEST')

# Always use setuptools
import setuptools

from setup_helpers import (get_comrec_build, cmdclass, INFO_VARS)

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

def generate_cython():
    cwd = os.path.abspath(os.path.dirname(__file__))
    print("Cythonizing sources")
    p = subprocess.call([sys.executable,
                         os.path.join(cwd, 'tools', 'cythonize.py'),
                         'scipy'],
                        cwd=cwd)
    if p != 0:
        # Could be due to a too old pip version and build isolation, check that
        try:
            # Note, pip may not be installed or not have been used
            import pip
        except (ImportError, ModuleNotFoundError):
            raise RuntimeError("Running cythonize failed!")
        else:
            _pep440 = importlib.import_module('scipy._lib._pep440')
            if _pep440.parse(pip.__version__) < _pep440.Version('18.0.0'):
                raise RuntimeError("Cython not found or too old. Possibly due "
                                   "to `pip` being too old, found version {}, "
                                   "needed is >= 18.0.0.".format(
                                   pip.__version__))
            else:
                raise RuntimeError("Running cythonize failed!")



def parse_setuppy_commands():
    """Check the commands and respond appropriately.  Disable broken commands.

    Return a boolean value for whether or not to run the build or not (avoid
    parsing Cython and template files if False).
    """
    args = sys.argv[1:]

    if not args:
        # User forgot to give an argument probably, let setuptools handle that.
        return True

    info_commands = ['--help-commands', '--name', '--version', '-V',
                     '--fullname', '--author', '--author-email',
                     '--maintainer', '--maintainer-email', '--contact',
                     '--contact-email', '--url', '--license', '--description',
                     '--long-description', '--platforms', '--classifiers',
                     '--keywords', '--provides', '--requires', '--obsoletes']

    for command in info_commands:
        if command in args:
            return False

    # Note that 'alias', 'saveopts' and 'setopt' commands also seem to work
    # fine as they are, but are usually used together with one of the commands
    # below and not standalone.  Hence they're not added to good_commands.
    good_commands = ('develop', 'sdist', 'build', 'build_ext', 'build_py',
                     'build_clib', 'build_scripts', 'bdist_wheel', 'bdist_rpm',
                     'bdist_wininst', 'bdist_msi', 'bdist_mpkg')

    for command in good_commands:
        if command in args:
            return True

    # The following commands are supported, but we need to show more
    # useful messages to the user
    if 'install' in args:
        print(textwrap.dedent("""
            Note: for reliable uninstall behaviour and dependency installation
            and uninstallation, please use pip instead of using
            `setup.py install`:

              - `pip install .`       (from a git repo or downloaded source
                                       release)
              - `pip install scipy`   (last SciPy release on PyPI)

            """))
        return True

    if '--help' in args or '-h' in sys.argv[1]:
        print(textwrap.dedent("""
            SciPy-specific help
            -------------------

            To install SciPy from here with reliable uninstall, we recommend
            that you use `pip install .`. To install the latest SciPy release
            from PyPI, use `pip install scipy`.

            For help with build/installation issues, please ask on the
            scipy-user mailing list.  If you are sure that you have run
            into a bug, please report it at https://github.com/scipy/scipy/issues.

            Setuptools commands help
            ------------------------
            """))
        return False


    # The following commands aren't supported.  They can only be executed when
    # the user explicitly adds a --force command-line argument.
    bad_commands = dict(
        test="""
            `setup.py test` is not supported.  Use one of the following
            instead:

              - `python runtests.py`              (to build and test)
              - `python runtests.py --no-build`   (to test installed scipy)
              - `>>> scipy.test()`           (run tests for installed scipy
                                              from within an interpreter)
            """,
        upload="""
            `setup.py upload` is not supported, because it's insecure.
            Instead, build what you want to upload and upload those files
            with `twine upload -s <filenames>` instead.
            """,
        upload_docs="`setup.py upload_docs` is not supported",
        easy_install="`setup.py easy_install` is not supported",
        clean="""
            `setup.py clean` is not supported, use one of the following instead:

              - `git clean -xdf` (cleans all files)
              - `git clean -Xdf` (cleans all versioned files, doesn't touch
                                  files that aren't checked into the git repo)
            """,
        check="`setup.py check` is not supported",
        register="`setup.py register` is not supported",
        bdist_dumb="`setup.py bdist_dumb` is not supported",
        bdist="`setup.py bdist` is not supported",
        flake8="`setup.py flake8` is not supported, use flake8 standalone",
        build_sphinx="`setup.py build_sphinx` is not supported, see doc/README.md",
        )
    bad_commands['nosetests'] = bad_commands['test']
    for command in ('upload_docs', 'easy_install', 'bdist', 'bdist_dumb',
                     'register', 'check', 'install_data', 'install_headers',
                     'install_lib', 'install_scripts', ):
        bad_commands[command] = "`setup.py %s` is not supported" % command

    for command in bad_commands.keys():
        if command in args:
            print(textwrap.dedent(bad_commands[command]) +
                  "\nAdd `--force` to your command to use it anyway if you "
                  "must (unsupported).\n")
            sys.exit(1)

    # Commands that do more than print info, but also don't need Cython and
    # template parsing.
    other_commands = ['egg_info', 'install_egg_info', 'rotate']
    for command in other_commands:
        if command in args:
            return False

    # If we got here, we didn't detect what setup.py command was given
    warnings.warn("Unrecognized setuptools command ('{}'), proceeding with "
                  "generating Cython sources and expanding templates".format(
                  ' '.join(sys.argv[1:])))
    return True

def check_setuppy_command():
    run_build = parse_setuppy_commands()
    if run_build:
        try:
            pkgname = 'numpy'
            import numpy
        except ImportError as exc:  # We do not have our build deps installed
            print(textwrap.dedent(
                    """Error: '%s' must be installed before running the build.
                    """
                    % (pkgname,)))
            sys.exit(1)

    return run_build


def get_build_ext_override():
    """
    Custom build_ext command to tweak extension building.
    """
    from numpy.distutils.command.build_ext import build_ext as npy_build_ext
    if int(os.environ.get('SCIPY_USE_PYTHRAN', 1)):
        try:
            import pythran
            from pythran.dist import PythranBuildExt
        except ImportError:
            BaseBuildExt = npy_build_ext
        else:
            BaseBuildExt = PythranBuildExt[npy_build_ext]
            _pep440 = importlib.import_module('scipy._lib._pep440')
            if _pep440.parse(pythran.__version__) < _pep440.Version('0.11.0'):
                raise RuntimeError("The installed `pythran` is too old, >= "
                                   "0.11.0 is needed, {} detected. Please "
                                   "upgrade Pythran, or use `export "
                                   "SCIPY_USE_PYTHRAN=0`.".format(
                                   pythran.__version__))
    else:
        BaseBuildExt = npy_build_ext

    class build_ext(BaseBuildExt):
        def finalize_options(self):
            super().finalize_options()

            # Disable distutils parallel build, due to race conditions
            # in numpy.distutils (Numpy issue gh-15957)
            if self.parallel:
                print("NOTE: -j build option not supported. Set NPY_NUM_BUILD_JOBS=4 "
                      "for parallel build.")
            self.parallel = None

        def build_extension(self, ext):
            # When compiling with GNU compilers, use a version script to
            # hide symbols during linking.
            if self.__is_using_gnu_linker(ext):
                export_symbols = self.get_export_symbols(ext)
                text = '{global: %s; local: *; };' % (';'.join(export_symbols),)

                script_fn = os.path.join(self.build_temp, 'link-version-{}.map'.format(ext.name))
                with open(script_fn, 'w') as f:
                    f.write(text)
                    # line below fixes gh-8680
                    ext.extra_link_args = [arg for arg in ext.extra_link_args if not "version-script" in arg]
                    ext.extra_link_args.append('-Wl,--version-script=' + script_fn)

            # Allow late configuration
            hooks = getattr(ext, '_pre_build_hook', ())
            _run_pre_build_hooks(hooks, (self, ext))

            super().build_extension(ext)

        def __is_using_gnu_linker(self, ext):
            if not sys.platform.startswith('linux'):
                return False

            # Fortran compilation with gfortran uses it also for
            # linking. For the C compiler, we detect gcc in a similar
            # way as distutils does it in
            # UnixCCompiler.runtime_library_dir_option
            if ext.language == 'f90':
                is_gcc = (self._f90_compiler.compiler_type in ('gnu', 'gnu95'))
            elif ext.language == 'f77':
                is_gcc = (self._f77_compiler.compiler_type in ('gnu', 'gnu95'))
            else:
                is_gcc = False
                if self.compiler.compiler_type == 'unix':
                    cc = sysconfig.get_config_var("CC")
                    if not cc:
                        cc = ""
                    compiler_name = os.path.basename(cc.split(" ")[0])
                    is_gcc = "gcc" in compiler_name or "g++" in compiler_name
            return is_gcc and sysconfig.get_config_var('GNULD') == 'yes'

    return build_ext


def get_build_clib_override():
    """
    Custom build_clib command to tweak library building.
    """
    from numpy.distutils.command.build_clib import build_clib as old_build_clib

    class build_clib(old_build_clib):
        def finalize_options(self):
            super().finalize_options()

            # Disable parallelization (see build_ext above)
            self.parallel = None

        def build_a_library(self, build_info, lib_name, libraries):
            # Allow late configuration
            hooks = build_info.get('_pre_build_hook', ())
            _run_pre_build_hooks(hooks, (self, build_info))
            old_build_clib.build_a_library(self, build_info, lib_name, libraries)

    return build_clib




def main():

    metadata = dict(
        name=INFO_VARS['NAME'],
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
        tests_require=['nose3'],
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

    if "--force" in sys.argv:
        run_build = True
        sys.argv.remove('--force')
    else:
        # Raise errors for unsupported commands, improve help output, etc.
        run_build = check_setuppy_command()

    if run_build:
        from numpy.distutils.core import setup

        # Customize extension building
        cmdclass['build_ext'] = get_build_ext_override()
        cmdclass['build_clib'] = get_build_clib_override()

        if not 'sdist' in sys.argv:
            # Generate Cython sources, unless we're creating an sdist
            # Cython is a build dependency, and shipping generated .c files
            # can cause problems (see gh-14199)
            generate_cython()

        metadata['configuration'] = configuration

    setup(**metadata)


if __name__ == "__main__":
    main()
