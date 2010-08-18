''' Release data for NIPY

This script should do no imports.  It only defines variables.
'''

LONG_DESCRIPTION = \
"""
Neuroimaging tools for Python (NIPY).

The aim of NIPY is to produce a platform-independent Python environment for
the analysis of brain imaging data using an open development model.

While
the project is still in its initial stages, packages for file I/O, script
support as well as single subject fMRI and random effects group comparisons
model are currently available.

Specifically, we aim to:

   1. Provide an open source, mixed language scientific programming
      environment suitable for rapid development.

   2. Create sofware components in this environment to make it easy
      to develop tools for MRI, EEG, PET and other modalities.

   3. Create and maintain a wide base of developers to contribute to
      this platform.

   4. To maintain and develop this framework as a single, easily
      installable bundle.

Package Organization 
==================== 
The nipy package contains the following subpackages and modules: 

.. packagetree:: 
   :style: UML  
"""

URL='http://nipy.org/nipy'
STATUS='alpha'

# Dependencies
SCIPY_MIN_VERSION = '0.5'
SYMPY_MIN_VERSION = '0.6.6'
MAYAVI_MIN_VERSION = '3.0'
CYTHON_MIN_VERSION = '0.12.1'

# Versions and locations of optional data packages
NIPY_DATA_URL= 'http://nipy.sourceforge.net/data-packages/'
DATA_PKGS = {'nipy-data': {'version':'0.2'},
             'nipy-templates': {'version':'0.2'}}
for key, value in DATA_PKGS.items():
    value['url'] = '%s%s-%s.tar.gz' % (NIPY_DATA_URL,
                                       key,
                                       value['version'])
del key, value
