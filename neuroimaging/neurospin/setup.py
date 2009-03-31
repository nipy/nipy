import os

# Global variables
LIBS = os.path.realpath('libfffpy')

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
    from numpy.distutils.system_info import get_info, NotFoundError
    from glob import glob

    config = Configuration('neurospin', parent_package, top_path)

    # fffpy library
    config.add_include_dirs(os.path.join(LIBS,'fff'))
    config.add_include_dirs(os.path.join(LIBS,'randomkit'))
    config.add_include_dirs(os.path.join(LIBS,'wrapper'))
    config.add_include_dirs(get_numpy_include_dirs())

    sources = [os.path.join(LIBS,'fff','*.c')]
    sources.append(os.path.join(LIBS,'wrapper','*.c'))

    """
    FIXME: the following external library 'mtrand' (C) is copied from 
     numpy, and included in the fff library for installation simplicity. 
     If numpy happens to expose its API one day, it would be neat to link 
     with them rather than copying the source code.

     numpy-trunk/numpy/random/mtrand/
    """
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
        lapack_info = get_info('lapack',0)

    if not lapack_info:
        raise  NotFoundError('no lapack installation found on this system')

    # OK, we found lapack, continue
    library_dirs = lapack_info['library_dirs']
    libraries = lapack_info['libraries']
    if 'include_dirs' in lapack_info:
        config.add_include_dirs(lapack_info['include_dirs'])    

    config.add_library('fffpy',
                       sources=sources,
                       library_dirs=library_dirs,
                       libraries=libraries,
                       extra_info=lapack_info)

    # Subpackages
    config.add_subpackage('bindings')
    config.add_subpackage('clustering')
    config.add_subpackage('eda')
    config.add_subpackage('glm')
    config.add_subpackage('graph')
    config.add_subpackage('group')
    config.add_subpackage('neuro')
    config.add_subpackage('registration')
    config.add_subpackage('scripts')
    config.add_subpackage('spatial_models')
    config.add_subpackage('utils')
    config.add_subpackage('viz')

    # ----------------------------------------------------------------------
    # Packages likely not to be moved over into nipy
    ## Data will be handled separately (nipy already has tools for this)
    ## config.add_subpackage('data')
    # ----------------------------------------------------------------------

    ## # Unitary tests 
    ## config.add_data_dir('tests')
    ## config.add_data_dir(os.path.join('tests', 'data'))
    
    ## config.add_data_dir('data')

    ## """
    ## Add an extension for each C file found in the source directory. 
    ## """
    ## root = os.path.split(__file__)[0]
    ## Cfiles = glob(os.path.join(root, '*.c'))
    ## for Cfile in Cfiles:
    ##     name, ext = os.path.splitext(os.path.basename(Cfile))
    ##     print('Adding extension: %s' % name)
    ##     config.add_extension(
    ##         '_'+name,
    ##         sources=[Cfile],
    ##         libraries=['fffpy']
    ##         )
    
    config.make_config_py() # installs __config__.py

    return config

if __name__ == '__main__':
    print 'This is the wrong setup.py file to run'
