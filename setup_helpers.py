# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Build helpers for setup.py

Includes package dependency checks, and code to build the documentation

To build the docs, run::

    python setup.py build_sphinx

This module has to work for python 2 and python 3.
"""
from __future__ import with_statement

# Standard library imports
import sys
import os
from os.path import join as pjoin, dirname, splitext, split as psplit
import zipfile
import warnings
import shutil
from distutils.cmd import Command
from distutils.command.clean import clean
from distutils.command.install_scripts import install_scripts
from distutils.version import LooseVersion
from distutils.dep_util import newer_group
from distutils.errors import DistutilsError

from numpy.distutils.misc_util import appendpath
from numpy.distutils import log
from numpy.distutils.misc_util import is_string

# Python 3 builder
from nisext.py3builder import build_py as old_build_py

# Sphinx import
try:
    from sphinx.setup_command import BuildDoc
except ImportError:
    have_sphinx = False
else:
    have_sphinx = True

# Get project related strings.  Please do not change this line to use
# execfile because execfile is not available in Python 3
_info_fname = pjoin('nipy', 'info.py')
INFO_VARS = {}
exec(open(_info_fname, 'rt').read(), {}, INFO_VARS)

DOC_BUILD_DIR = os.path.join('build', 'html')

CYTHON_MIN_VERSION = INFO_VARS['CYTHON_MIN_VERSION']

################################################################################
# Distutils Command class for installing nipy to a temporary location. 
class TempInstall(Command):
    temp_install_dir = os.path.join('build', 'install')

    def run(self):
        """ build and install nipy in a temporary location. """
        install = self.distribution.get_command_obj('install')
        install.install_scripts = self.temp_install_dir
        install.install_base    = self.temp_install_dir
        install.install_platlib = self.temp_install_dir 
        install.install_purelib = self.temp_install_dir 
        install.install_data    = self.temp_install_dir 
        install.install_lib     = self.temp_install_dir 
        install.install_headers = self.temp_install_dir 
        install.run()

        # Horrible trick to reload nipy with our temporary instal
        for key in sys.modules.keys():
            if key.startswith('nipy'):
                sys.modules.pop(key, None)
        sys.path.append(os.path.abspath(self.temp_install_dir))
        # Pop the cwd
        sys.path.pop(0)
        import nipy

    def initialize_options(self):
        pass
    
    def finalize_options(self):
        pass


################################################################################
# Distutils Command class for API generation 
class APIDocs(TempInstall):
    description = \
    """generate API docs """

    user_options = [
        ('None', None, 'this command has no options'),
        ]


    def run(self):
        # First build the project and install it to a temporary location.
        TempInstall.run(self)
        os.chdir('doc')
        try:
            # We are running the API-building script via an
            # system call, but overriding the import path.
            toolsdir = os.path.abspath(pjoin('..', 'tools'))
            build_templates = pjoin(toolsdir, 'build_modref_templates.py')
            cmd = """%s -c 'import sys; sys.path.append("%s"); sys.path.append("%s"); execfile("%s", dict(__name__="__main__"))'""" \
                % (sys.executable, 
                   toolsdir,
                   self.temp_install_dir,
                   build_templates)
            os.system(cmd)
        finally:
            os.chdir('..')


################################################################################
# Code to copy the sphinx-generated html docs in the distribution.
def relative_path(filename):
    """ Return the relative path to the file, assuming the file is
        in the DOC_BUILD_DIR directory.
    """
    length = len(os.path.abspath(DOC_BUILD_DIR)) + 1
    return os.path.abspath(filename)[length:]



################################################################################
# Distutils Command class to clean
class Clean(clean):

    def run(self):
        clean.run(self)
        api_path = os.path.join('doc', 'api', 'generated')
        if os.path.exists(api_path):
            print("Removing %s" % api_path)
            shutil.rmtree(api_path)
        if os.path.exists(DOC_BUILD_DIR):
            print("Removing %s" % DOC_BUILD_DIR)
            shutil.rmtree(DOC_BUILD_DIR)


################################################################################
# Distutils Command class build the docs
if have_sphinx:
    class MyBuildDoc(BuildDoc):
        """ Sub-class the standard sphinx documentation building system, to
            add logics for API generation and matplotlib's plot directive.
        """

        def run(self):
            self.run_command('api_docs')
            # We need to be in the doc directory for to plot_directive
            # and API generation to work
            os.chdir('doc')
            try:
                BuildDoc.run(self)
            finally:
                os.chdir('..')
            self.zip_docs()

        def zip_docs(self):
            if not os.path.exists(DOC_BUILD_DIR):
                raise OSError('Doc directory does not exist.')
            target_file = os.path.join('doc', 'documentation.zip')
            # ZIP_DEFLATED actually compresses the archive. However, there
            # will be a RuntimeError if zlib is not installed, so we check
            # for it. ZIP_STORED produces an uncompressed zip, but does not
            # require zlib.
            try:
                zf = zipfile.ZipFile(target_file, 'w', 
                                            compression=zipfile.ZIP_DEFLATED)
            except RuntimeError:
                warnings.warn('zlib not installed, storing the docs '
                                'without compression')
                zf = zipfile.ZipFile(target_file, 'w', 
                                            compression=zipfile.ZIP_STORED)    

            for root, dirs, files in os.walk(DOC_BUILD_DIR):
                relative = relative_path(root)
                if not relative.startswith('.doctrees'):
                    for f in files:
                        zf.write(os.path.join(root, f), 
                                os.path.join(relative, 'html_docs', f))
            zf.close()

        def finalize_options(self):
            """ Override the default for the documentation build
                directory.
            """
            self.build_dir = os.path.join(*DOC_BUILD_DIR.split(os.sep)[:-1])
            BuildDoc.finalize_options(self)

else: # failed Sphinx import
    # Raise an error when trying to build docs
    class MyBuildDoc(Command):
        user_options = []
        def run(self):
            raise ImportError(
                "Sphinx is not installed, docs cannot be built")
        def initialize_options(self):
            pass
        def finalize_options(self):
            pass

# Start of numpy.distutils.command.build_py.py copy

class build_py(old_build_py):
    """ Copied verbatim from numpy.distutils.command.build_py.py

    If we are on python 2, this code merely replicates numpy distutils, and we
    could get the same effect by importing build_py from numpy.distutils.  If we
    are on python 3, then `old_build_py` will have python 2to3 routines
    contained, and we also need the methods below to work with numpy distutils.

    If you know a better way to do this, please, send a patch and make me glad.
    """

    def run(self):
        build_src = self.get_finalized_command('build_src')
        if build_src.py_modules_dict and self.packages is None:
            self.packages = build_src.py_modules_dict.keys ()
        old_build_py.run(self)

    def find_package_modules(self, package, package_dir):
        modules = old_build_py.find_package_modules(self, package, package_dir)

        # Find build_src generated *.py files.
        build_src = self.get_finalized_command('build_src')
        modules += build_src.py_modules_dict.get(package,[])

        return modules

    def find_modules(self):
        old_py_modules = self.py_modules[:]
        new_py_modules = filter(is_string, self.py_modules)
        self.py_modules[:] = new_py_modules
        modules = old_build_py.find_modules(self)
        self.py_modules[:] = old_py_modules

        return modules

    # XXX: Fix find_source_files for item in py_modules such that item is 3-tuple
    # and item[2] is source file.

# End of numpy.distutils.command.build_py.py copy

def have_good_cython():
    try:
        from Cython.Compiler.Version import version
    except ImportError:
        return False
    return LooseVersion(version) >= LooseVersion(CYTHON_MIN_VERSION)


def generate_a_pyrex_source(self, base, ext_name, source, extension):
    ''' Monkey patch for numpy build_src.build_src method

    Uses Cython instead of Pyrex.
    '''
    good_cython = have_good_cython()
    if self.inplace or not good_cython:
        target_dir = dirname(base)
    else:
        target_dir = appendpath(self.build_src, dirname(base))
    target_file = pjoin(target_dir, ext_name + '.c')
    depends = [source] + extension.depends
    sources_changed = newer_group(depends, target_file, 'newer') 
    if self.force or sources_changed:
        if good_cython:
            # add distribution (package-wide) include directories, in order
            # to pick up needed .pxd files for cython compilation
            incl_dirs = extension.include_dirs[:]
            dist_incl_dirs = self.distribution.include_dirs
            if not dist_incl_dirs is None:
                incl_dirs += dist_incl_dirs
            import Cython.Compiler.Main
            log.info("cythonc:> %s" % (target_file))
            self.mkpath(target_dir)
            options = Cython.Compiler.Main.CompilationOptions(
                defaults=Cython.Compiler.Main.default_options,
                include_path=incl_dirs,
                output_file=target_file)
            cython_result = Cython.Compiler.Main.compile(source,
                                                       options=options)
            if cython_result.num_errors != 0:
                raise DistutilsError("%d errors while compiling "
                                     "%r with Cython"
                                     % (cython_result.num_errors, source))
        elif sources_changed and os.path.isfile(target_file):
            raise DistutilsError("Cython >=%s required for compiling %r"
                                 " because sources (%s) have changed" %
                                 (CYTHON_MIN_VERSION, source, ','.join(depends)))
        else:
            raise DistutilsError("Cython >=%s required for compiling %r"
                                 " but not available" %
                                 (CYTHON_MIN_VERSION, source))
    return target_file

BAT_TEMPLATE = \
r"""@echo off
REM wrapper to use shebang first line of {FNAME}
set mypath=%~dp0
set pyscript="%mypath%{FNAME}"
set /p line1=<%pyscript%
if "%line1:~0,2%" == "#!" (goto :goodstart)
echo First line of %pyscript% does not start with "#!"
exit /b 1
:goodstart
set py_exe=%line1:~2%
call %py_exe% %pyscript% %*
"""

class install_scripts_nipy(install_scripts):
    """ Make scripts executable on Windows

    Scripts are bare file names without extension on Unix, fitting (for example)
    Debian rules. They identify as python scripts with the usual ``#!`` first
    line. Unix recognizes and uses this first "shebang" line, but Windows does
    not. So, on Windows only we add a ``.bat`` wrapper of name
    ``bare_script_name.bat`` to call ``bare_script_name`` using the python
    interpreter from the #! first line of the script.

    Notes
    -----
    See discussion at
    http://matthew-brett.github.com/pydagogue/installing_scripts.html and
    example at git://github.com/matthew-brett/myscripter.git for more
    background.
    """
    def run(self):
        install_scripts.run(self)
        if not os.name == "nt":
            return
        for filepath in self.get_outputs():
            # If we can find an executable name in the #! top line of the script
            # file, make .bat wrapper for script.
            with open(filepath, 'rt') as fobj:
                first_line = fobj.readline()
            if not (first_line.startswith('#!') and
                    'python' in first_line.lower()):
                log.info("No #!python executable found, skipping .bat "
                            "wrapper")
                continue
            pth, fname = psplit(filepath)
            froot, ext = splitext(fname)
            bat_file = pjoin(pth, froot + '.bat')
            bat_contents = BAT_TEMPLATE.replace('{FNAME}', fname)
            log.info("Making %s wrapper for %s" % (bat_file, filepath))
            if self.dry_run:
                continue
            with open(bat_file, 'wt') as fobj:
                fobj.write(bat_contents)


# The command classes for distutils, used by setup.py
cmdclass = {'api_docs': APIDocs,
            'clean': Clean,
            'build_sphinx': MyBuildDoc,
            'install_scripts': install_scripts_nipy}
