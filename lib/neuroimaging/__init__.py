import re

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

def preload(packages=nontest_packages):
    """
    Import the specified packages (enabling fewer imports in client scripts).
    By default, import all non-test packages:\n%s
    """%"\n".join(nontest_packages)
    for package in packages: __import__(package, globals(), locals())

# Always preload all packages.  This should be removed as soon as the client
# scripts can be modified to call it themselves.
preload()
