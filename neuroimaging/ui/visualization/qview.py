__docformat__ = 'restructuredtext'

import sys
import pylab
from neuroimaging.core.api import Image
import neuroimaging.ui.visualization.viewer as viewer

x = Image(sys.argv[1])
v = viewer.BoxViewer(x)

if len(sys.argv) == 3:
    m, M = map(float, sys.argv[1:])
    v.m = m
    v.M = M

v.draw()
pylab.show()
