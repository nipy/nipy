"""
Neuroimaging tools for Python (NiPy).

The aim of NiPy is to produce a platform-independent Python environment for
the analysis of brain imaging data using an open development model.  While
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
"""

__version__  = "0.01a"
__revision__ = int("$Rev$".split()[-2])
__status__   = "alpha"
__date__     = "$LastChangedDate$"
__url__      = "http://neuroimaging.scipy.org"

import re
from copy import copy

from numpy import product

import defines
from neuroimaging.utils.path import path

packages = (
  'neuroimaging',
  'neuroimaging.utils.tests',
  'neuroimaging.data_io',
  'neuroimaging.data_io.tests',
  'neuroimaging.fmri',
  'neuroimaging.fmri.tests',
  'neuroimaging.fmri.fmristat',
  'neuroimaging.fmri.fmristat.tests',
  'neuroimaging.core.image',
  'neuroimaging.core.image.tests',
  'neuroimaging.core.image.formats',
  'neuroimaging.core.image.formats.tests',
  'neuroimaging.reference',
  'neuroimaging.reference.tests',
  'neuroimaging.algorithms.statistics',
  'neuroimaging.algorithms.statistics.tests')

PYLAB_DEF, pylab = defines.pylab_def()
if PYLAB_DEF:
    packages += ('neuroimaging.visualization',
                'neuroimaging.visualization.cmap',
                'neuroimaging.visualization.tests')

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

# modules to be pre-imported for convenience
_preload_modules = (
  'neuroimaging.core.image.formats.analyze',
  'neuroimaging.core.image.formats.nifti1',
  'neuroimaging.core.image.interpolation',
  'neuroimaging.core.image.onesample',
  'neuroimaging.core.image.regression',
  'neuroimaging.reference.axis',
  'neuroimaging.reference.coordinate_system',
  'neuroimaging.reference.grid',
  'neuroimaging.reference.iterators',
  'neuroimaging.reference.mapping',
  'neuroimaging.reference.slices',
  'neuroimaging.algorithms.statistics.regression',
  'neuroimaging.algorithms.statistics.classification',
  'neuroimaging.visualization.viewer',)

#-----------------------------------------------------------------------------
def hasattrs(obj, *attrs):
    for attr in attrs:
        if not hasattr(obj, attr): return False
    return True

#-----------------------------------------------------------------------------
def haslength(obj): return hasattr(obj,"__len__")

#-----------------------------------------------------------------------------
def flatten(arr, dim=0):
    if len(arr.shape) < 2: return
    oldshape = arr.shape
    arr.shape = oldshape[0:dim] + (product(oldshape[dim:]),)

#-----------------------------------------------------------------------------
def reorder(seq, order): return [seq[i] for i in order]

#-----------------------------------------------------------------------------
def reverse(seq): return reorder(seq, range(len(seq)-1, -1, -1))

#-----------------------------------------------------------------------------
def ensuredirs(directory):
    if not isinstance(directory, path): directory= path(directory)
    if not directory.exists(): directory.makedirs()

#-----------------------------------------------------------------------------
#def keywords(func):
#    if not hasattrs(func, "func_code", "func_defaults"):
#        raise ValueError(
#          "please pass a function or method object (got %s)"%func)
#    argnames = func.func_code.co_ 

#-----------------------------------------------------------------------------
def preload(packages=nontest_packages):
    """
    Import the specified modules/packages (enabling fewer imports in client
    scripts).  By default, import all non-test packages:\n%s
    and the following modules:\n%s
    """%("\n".join(nontest_packages),"\n".join(_preload_modules))
    for package in packages: __import__(package, {}, {})
    for module in _preload_modules: __import__(module, {}, {})

#-----------------------------------------------------------------------------
def import_from(modulename, objectname):
    "Import and return objectname from modulename."
    module = __import__(modulename, {}, {}, (objectname,))
    try: return getattr(module, objectname)
    except AttributeError: return None


# Always preload all packages.  This should be removed as soon as the client
# scripts can be modified to call it themselves.
#preload()

