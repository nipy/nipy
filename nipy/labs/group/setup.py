
def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    # We need this because libcstat.a is linked to lapack, which can
    # be a fortran library, and the linker needs this information.
    from numpy.distutils.system_info import get_info
    # First, try 'lapack_info', as that seems to provide more details on Linux
    # (both 32 and 64 bits):
    lapack_info = get_info('lapack_opt', 0)
    if 'libraries' not in lapack_info:
        # But on OSX that may not give us what we need, so try with 'lapack'
        # instead.  NOTE: scipy.linalg uses lapack_opt, not 'lapack'...
        lapack_info = get_info('lapack',0)
    config = Configuration('group', parent_package, top_path)
    config.add_subpackage('tests')
    config.add_extension(
        'onesample',
        sources=['onesample.pyx'],
        libraries=['cstat'],
        extra_info=lapack_info,
        )
    config.add_extension(
        'twosample',
        sources=['twosample.pyx'],
        libraries=['cstat'],
        extra_info=lapack_info,
        )
    config.add_extension(
        'glm_twolevel',
        sources=['glm_twolevel.pyx'],
        libraries=['cstat'],
        extra_info=lapack_info,
        )
    return config


if __name__ == '__main__':
    print('This is the wrong setup.py file to run')
