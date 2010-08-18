# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import os
from distutils import log 
from distutils.msvccompiler import get_build_version as get_msvc_build_version

# Global variables
LIBS = os.path.realpath('libcstat')

# Configuration copied from numpy/random/setup.py for fixing mingw ftime
# DLL import problem
def needs_mingw_ftime_workaround():
    # We need the mingw workaround for _ftime if the msvc runtime version is
    # 7.1 or above and we build with mingw ...
    # ... but we can't easily detect compiler version outside distutils command
    # context, so we will need to detect in randomkit whether we build with gcc
    msver = get_msvc_build_version()
    if msver and msver >= 8:
        return True
    return False


def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
    from numpy.distutils.system_info import get_info

    config = Configuration('neurospin', parent_package, top_path)

    # This is also from numpy/random/setup.py - ftime fix
    defs = []
    if needs_mingw_ftime_workaround():
        defs.append(("NPY_NEEDS_MINGW_TIME_WORKAROUND", None))

    # cstat library
    config.add_include_dirs(os.path.join(LIBS,'fff'))
    config.add_include_dirs(os.path.join(LIBS,'randomkit'))
    config.add_include_dirs(os.path.join(LIBS,'wrapper'))
    config.add_include_dirs(get_numpy_include_dirs())

    sources = [os.path.join(LIBS,'fff','*.c')]
    sources.append(os.path.join(LIBS,'wrapper','*.c'))

    # FIXME: the following external library 'mtrand' (C) is copied from 
    # numpy, and included in the fff library for installation simplicity. 
    # If numpy happens to expose its API one day, it would be neat to link 
    # with them rather than copying the source code.
    #
    # numpy-trunk/numpy/random/mtrand/
    sources.append(os.path.join(LIBS,'randomkit','*.c'))

    # Link with lapack if found on the system

    # XXX: We need to better sort out the use of get_info() for Lapack, because
    # using 'lapack' and 'lapack_opt' returns different things even comparing
    # Ubuntu 8.10 machines on 32 vs 64 bit setups.  On OSX
    # get_info('lapack_opt') does not return the keys: 'libraries' and
    # 'library_dirs', but get_info('lapack') does.
    #
    # For now this code should do the right thing on OSX and linux, but we
    # should ask on the numpy list for clarification on the proper approach.

    # XXX: If you modify these lines, remember to pass the information
    # along to the different .so in the neurospin build system.
    # First, try 'lapack_info', as that seems to provide more details on Linux
    # (both 32 and 64 bits):
    lapack_info = get_info('lapack_opt', 0)
    if 'libraries' not in lapack_info:
        # But on OSX that may not give us what we need, so try with 'lapack'
        # instead.  NOTE: scipy.linalg uses lapack_opt, not 'lapack'...
        lapack_info = get_info('lapack', 0)

    # If no lapack install is found, we use the rescue lapack lite
    # distribution included in the package (sources have been
    # translated to C using f2c)
    if not lapack_info:
        log.warn('No lapack installation found, using lapack lite distribution')
        sources.append(os.path.join(LIBS,'lapack_lite','*.c'))
        library_dirs = []
        libraries = []

    # Best-case scenario: lapack found 
    else: 
        library_dirs = lapack_info['library_dirs']
        libraries = lapack_info['libraries']
        if 'include_dirs' in lapack_info:
            config.add_include_dirs(lapack_info['include_dirs'])    

    # Information message
    print('LAPACK build options:')
    print('library_dirs: %s ' % library_dirs)
    print('libraries: %s ' % libraries)
    print('lapack_info: %s ' % lapack_info)

    config.add_library('cstat',
                       sources=sources,
                       macros=defs,
                       library_dirs=library_dirs,
                       libraries=libraries,
                       extra_info=lapack_info)

    # Subpackages
    config.add_subpackage('bindings')
    config.add_subpackage('clustering')
    config.add_subpackage('eda')
    config.add_subpackage('glm')
    config.add_subpackage('glm_files_layout')
    config.add_subpackage('graph')
    config.add_subpackage('group')
    config.add_subpackage('scripts')
    config.add_subpackage('spatial_models')
    config.add_subpackage('utils')
    config.add_subpackage('viz_tools')
    config.add_subpackage('datasets')
    config.add_subpackage('image')
    config.add_subpackage('segmentation')
    config.add_subpackage('registration')
    config.add_subpackage('tests')
    
    config.make_config_py() # installs __config__.py

    return config

if __name__ == '__main__':
    print 'This is the wrong setup.py file to run'
