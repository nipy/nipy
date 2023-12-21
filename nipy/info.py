""" Data package information
"""

# For compatibility
from . import __version__

# Versions and locations of optional data packages
NIPY_DATA_URL= 'http://nipy.org/data-packages/'
DATA_PKGS = {'nipy-data': {'min version':'0.3',
                           'relpath':'nipy/data'},
             'nipy-templates': {'min version':'0.3',
                                'relpath':'nipy/templates'}
            }
NIPY_INSTALL_HINT = \
"""You can download and install the package from:

%s

Check the instructions in the ``doc/users/install_data.rst`` file in the nipy
source tree, or online at http://nipy.org/nipy/users/install_data.html

If you have the package, have you set the path to the package correctly?"""

for key, value in DATA_PKGS.items():
    url = f"{NIPY_DATA_URL}{key}-{value['min version']}.tar.gz"
    value['name'] = key
    value['install hint'] = NIPY_INSTALL_HINT % url

del key, value, url
