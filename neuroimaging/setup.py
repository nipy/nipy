
def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('neuroimaging', parent_package, top_path)

    # List all packages to be loaded here
    config.add_subpackage('algorithms')
    config.add_subpackage('core')
    config.add_subpackage('fixes')
    config.add_subpackage('io')
    config.add_subpackage('modalities')
    config.add_subpackage('utils')
    config.add_subpackage('testing')

    # Note: this is a special subpackage, where all the code from Neurospin
    # that up until now had been living in the 'fff2' branch will go.
    # Eventually the code contained therein will be migrated to whichever parts
    # of the main package it logically belongs in.  But initially we are
    # putting everythin under this subpackage to make the management and
    # migration easier.
    config.add_subpackage('neurospin')

    # List all data directories to be loaded here
    config.add_data_dir('testing')

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
