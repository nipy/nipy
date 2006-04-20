"""
Insert long description here.
"""
import re
from path import path

__version__ = "0.01a"

packages = (
  'neuroimaging',
  'neuroimaging.tests',
  'neuroimaging.data',
  'neuroimaging.data.tests',
  'neuroimaging.fmri',
  'neuroimaging.fmri.tests',
  'neuroimaging.fmri.fmristat',
  'neuroimaging.fmri.fmristat.tests',
  'neuroimaging.image',
  'neuroimaging.image.tests',
  'neuroimaging.image.formats',
  'neuroimaging.image.formats.tests',
  'neuroimaging.reference',
  'neuroimaging.reference.tests',
  'neuroimaging.statistics',
  'neuroimaging.statistics.tests',
  'neuroimaging.visualization',
  'neuroimaging.visualization.cmap',
  'neuroimaging.visualization.tests')

testmatch = re.compile(".*tests").search
nontest_packages = [p for p in packages if not testmatch(p)]

# modules to be pre-imported for convenience
_preload_modules = (
  'neuroimaging.image.formats.analyze',
  'neuroimaging.image.interpolation',
  'neuroimaging.image.onesample',
  'neuroimaging.image.regression',
  'neuroimaging.reference.axis',
  'neuroimaging.reference.coordinate_system',
  'neuroimaging.reference.grid',
  'neuroimaging.reference.grid_iterators',
  'neuroimaging.reference.mapping',
  'neuroimaging.reference.slices',
  'neuroimaging.statistics.model',
  'neuroimaging.statistics.regression',
  'neuroimaging.statistics.classification',
  'neuroimaging.statistics.iterators',
  'neuroimaging.statistics.contrast',
  'neuroimaging.statistics.utils',
  'neuroimaging.visualization.viewer',)

#-----------------------------------------------------------------------------
def ensuredirs(dir):
    if not isinstance(dir, path): dir= path(dir)
    if not dir.exists(): dir.makedirs()

#-----------------------------------------------------------------------------
def preload(packages=nontest_packages):
    """
    Import the specified modules/packages (enabling fewer imports in client
    scripts).  By default, import all non-test packages:\n%s
    and the following modules:\n%s
    """%("\n".join(nontest_packages),"\n".join(_preload_modules))
    for package in packages: __import__(package, globals(), locals())
    for module in _preload_modules: __import__(module, globals(), locals())

#-----------------------------------------------------------------------------
def import_from(modulename, objectname):
    "Import and return objectname from modulename."
    module = __import__(modulename, globals(), locals(), (objectname,))
    try: return getattr(module, objectname)
    except AttributeError: return None


# Always preload all packages.  This should be removed as soon as the client
# scripts can be modified to call it themselves.
preload()
