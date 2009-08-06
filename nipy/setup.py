import ConfigParser
import os
from os.path import join as pjoin

NIPY_DEFAULTS = dict()

################################################################################
def get_nipy_info():
    """ Reuse numpy's distutils to get and store information about nipy
        in the site.cfg.
    """
    from numpy.distutils.system_info import get_standard_file
    files = get_standard_file('site.cfg')
    cp = ConfigParser.ConfigParser(NIPY_DEFAULTS)
    cp.read(files)
    if not cp.has_section('nipy'):
        cp.add_section('nipy')
    info = dict(cp.items('nipy'))
    for key, value in info.iteritems():
        if value.startswith('~'):
            info[key] = os.path.expanduser(value)
    # Ugly fix for bug 409269
    if info.has_key('libraries'): 
        info['libraries'] = list(info['libraries'])
    # End of ugly fix
    return info


################################################################################
def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import system_info
    config = Configuration('nipy', parent_package, top_path)

    # List all packages to be loaded here
    config.add_subpackage('algorithms')
    config.add_subpackage('interfaces')
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

    #####################################################################
    # Store the setup information, including the nipy-specific
    # information in a __config__ file.
    class nipy_info(system_info):
        """ We are subclassing numpy.distutils's system_info to
            insert information in the __config__ file.
            The class name determines the name of the variable
            in the __config__ file.
        """
    nipy_info().set_info(**get_nipy_info())
    config.make_config_py()

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
