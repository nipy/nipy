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
The neuroimaging package contains the following subpackages and modules: 

.. packagetree:: 
   :style: UML  
"""
__docformat__ = 'restructuredtext en'



from version import version as __version__
__revision__ = int("$Rev$".split()[-2])
__status__   = 'alpha'
__date__    = "$LastChangedDate$"
__url__     = 'http://neuroimaging.scipy.org'


import re
from copy import copy

from numpy import product

from neuroimaging import defines
from neuroimaging.utils.path import path

packages = (
  'neuroimaging',
  'neuroimaging.algorithms',
  'neuroimaging.algorithms.tests',
  'neuroimaging.algorithms.statistics',
  'neuroimaging.algorithms.statistics.tests',
  'neuroimaging.core',
  'neuroimaging.core.image',
  'neuroimaging.core.image.tests',
  'neuroimaging.core.reference',
  'neuroimaging.core.reference.tests',
  'neuroimaging.data_io',
  'neuroimaging.data_io.tests',
  'neuroimaging.data_io.formats',
  'neuroimaging.data_io.formats.tests',
  'neuroimaging.modalities',
  'neuroimaging.modalities.fmri',
  'neuroimaging.modalities.fmri.tests',
  'neuroimaging.modalities.fmri.fmristat',
  'neuroimaging.modalities.fmri.fmristat.tests',
  'neuroimaging.utils',
  'neuroimaging.utils.tests',
  'neuroimaging.utils.tests.data')

PYLAB_DEF, pylab = defines.pylab_def()
if PYLAB_DEF:
    packages += ('neuroimaging.ui',
                 'neuroimaging.ui.visualization',
                 'neuroimaging.ui.visualization.cmap',
                 'neuroimaging.ui.visualization.tests')

ENTHOUGHT_TRAITS_DEF, traits = defines.enthought_traits_def()
if not ENTHOUGHT_TRAITS_DEF:
    packages += ('neuroimaging.utils',
                 'neuroimaging.utils.enthought',
                 'neuroimaging.utils.enthought.traits',
                 'neuroimaging.utils.enthought.traits.ui',
                 'neuroimaging.utils.enthought.traits.ui.null',
                 'neuroimaging.utils.enthought.util',
                 'neuroimaging.utils.enthought.resource')

testmatch = re.compile(".*tests").search
nontest_packages = [p for p in packages if not testmatch(p)]


def import_from(modulename, objectname):
    """Import and return objectname from modulename."""
    module = __import__(modulename, {}, {}, (objectname,))
    try:
        return getattr(module, objectname)
    except AttributeError:
        return None

from numpy.testing import NumpyTest, importall

def test(level=1, verbosity=1):
    importall('neuroimaging')
    if level <= 10:
        return NumpyTest().test(level, verbosity)
    else:
        return NumpyTest().testall(level, verbosity)
test.__doc__ = NumpyTest.test.__doc__

