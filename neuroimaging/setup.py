import os

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('neuroimaging', parent_package, top_path)

    config.add_subpackage('algorithms')
    config.add_subpackage('core')
    config.add_subpackage('data_io')
    config.add_subpackage('modalities')
    config.add_subpackage('ui')
    config.add_subpackage('utils')

    # FIXME: remove once testing is setup
    config.add_subpackage('data')
    config.add_data_dir('data')

    config.add_subpackage('testing')
    config.add_data_dir('testing')

    try: os.remove("lib/neuroimaging/__svn_version__.py")
    except OSError: pass
    config.make_svn_version_py(delete=False)

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
