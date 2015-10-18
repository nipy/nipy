# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from __future__ import absolute_import

import os
import sys

# Cannot use internal copy of six because can't import from nipy tree
# This is to allow setup.py to run without a full nipy
PY3 = sys.version_info[0] == 3
if PY3:
    string_types = str,
    from configparser import ConfigParser
else:
    string_types = basestring,
    from ConfigParser import ConfigParser

NIPY_DEFAULTS = dict()

################################################################################
def get_nipy_info():
    """ Reuse numpy's distutils to get and store information about nipy
        in the site.cfg.
    """
    from numpy.distutils.system_info import get_standard_file
    files = get_standard_file('site.cfg')
    cp = ConfigParser(NIPY_DEFAULTS)
    cp.read(files)
    if not cp.has_section('nipy'):
        cp.add_section('nipy')
    info = dict(cp.items('nipy'))
    for key, value in info.items():
        if value.startswith('~'):
            info[key] = os.path.expanduser(value)
    # Ugly fix for bug 409269
    if 'libraries' in info and isinstance(info['libraries'], string_types):
        info['libraries'] = [info['libraries']]
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
    config.add_subpackage('tests')
    config.add_subpackage('externals')
    config.add_subpackage('testing')

    # Note: this is a special subpackage containing that will later be
    # migrated to whichever parts of the main package they logically
    # belong in. But initially we are putting everythin under this
    # subpackage to make the management and migration easier.
    config.add_subpackage('labs')

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
