import image
import reference
import fmri
import statistics
import visualization
import re

__version__ = "0.01a"

packages = (
  'neuroimaging',
  'neuroimaging.tests',
  'neuroimaging.statistics',
  'neuroimaging.statistics.tests',
  'neuroimaging.image',
  'neuroimaging.image.tests',
  'neuroimaging.reference',
  'neuroimaging.reference.tests',
  'neuroimaging.data',
  'neuroimaging.data.tests',
  'neuroimaging.image.formats',
  'neuroimaging.image.formats.tests',
  'neuroimaging.image.formats.analyze',
  'neuroimaging.fmri',
  'neuroimaging.fmri.tests',
  'neuroimaging.fmri.fmristat',
  'neuroimaging.fmri.fmristat.tests',
  'neuroimaging.visualization',
  'neuroimaging.visualization.cmap',
  'neuroimaging.visualization.tests')

testmatch = re.compile(".*tests").search
nontest_packages = [p for p in packages if not testmatch(p)]
