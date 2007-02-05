import pylab
import numpy as N

from neuroimaging.core.image.image import Image
from neuroimaging.ui.visualization.viewer import BoxViewer

import keith

path = "../../../nipy-data/fmri/FIAC"

def nipy_run(subject=3, run=3, which='contrasts', contrast='overall',
             stat='t', **extra):

    runfile = '%s/fiac%d/fonc%d/fsl/fmristat_run/%s/%s/%s.img' % (path, subject, run, which, contrast, stat)

    return Image(runfile)


def nipy_rho(subject=3, run=3, **kw):

    runfile = '%s/fiac%d/fonc%d/fsl/fmristat_run/rho.img' % (path, subject, run)

    return Image(runfile)

def mask(subject=3, run=3, **kw):
    runfile = '%s/fiac%d/fonc%d/fsl/mask.img' % (path, subject, run)

    return Image(runfile)

import optparse

parser = optparse.OptionParser()

parser.add_option('', '--which', help='contrasts or delays', dest='which',
                      default='contrasts')
parser.add_option('', '--run', help='which run?', dest='run', default=1,
                  type='int')
parser.add_option('', '--subject', help='which subject?', dest='subject',
                  default=0, type='int')
parser.add_option('', '--stat', help='t, sd or effect?', dest='stat', default='t')
parser.add_option('', '--contrast', help='overall, sentence, speaker, or interaction', dest='contrast', default='overall')
parser.add_option('', '--vmin', help='min for colorbar', dest='m', type='float')
parser.add_option('', '--vmax', help='max for colorbar', dest='M', type='float')
parser.add_option('', '--rho', help='compare AR(1) coefs', dest='rho',
                  action='store_true', default=False)

options, args = parser.parse_args()
options = parser.values.__dict__

print options

if parser.values.rho:
    x = nipy_rho(**options)
    y = keith.rho(**options)
else:
    x = nipy_run(**options)
    y = keith.result(**options)
    
m = mask(**options)

vx = BoxViewer(x, mask=m, colormap='spectral')
vy = BoxViewer(y, mask=m, colormap='spectral')

if options['m'] is not None:
    vx.m = options['m']
    vy.m = options['m']

if options['M'] is not None:
    vx.M = options['M']
    vy.M = options['M']

vx.draw()
vy.draw()

X = x.readall() * m.readall()
X.shape = N.product(X.shape)

Y = y.readall() * m.readall()
Y.shape = N.product(Y.shape)

print N.corrcoef(X, Y)

pylab.show()
