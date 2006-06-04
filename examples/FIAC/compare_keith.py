import pylab
import numpy as N

from neuroimaging.image import Image
from neuroimaging.visualization.viewer import BoxViewer

contrast_map = {'sentence': 'sen',
                'speaker': 'spk',
                'overall': 'all',
                'interaction': 'snp'}

which_map = {'contrasts': 'mag',
             'delays': 'del'}

stat_map = {'t':'t',
            'effect': 'ef',
            'sd': 'sd'}

def fmristat_run(subject=3, run=3, which='contrasts', contrast='overall', stat='t', **extra):
    contrast = contrast_map[contrast]
    which = which_map[which]
    stat = stat_map[stat]

    runfile = '/home/analysis/FIAC/fmristat/fiac%d/fiac%d_fonc%d_%s_%s_%s.img' % (subject, subject, run, contrast, which, stat)
    
    return Image(runfile)

def fmristat_rho(subject=3, run=3, **extra):

    runfile = '/home/analysis/FIAC/fmristat/fiac%d/fiac%d_fonc%d_all_cor.img' % (subject, subject, run)
    
    return Image(runfile)



def nipy_run(subject=3, run=3, which='contrasts', contrast='overall', stat='t', **extra):

    runfile = '/home/analysis/FIAC/fiac%d/fonc%d/fsl/fmristat_run/%s/%s/%s.img' % (subject, run, which, contrast, stat)

    return Image(runfile)


def nipy_rho(subject=3, run=3, **extra):

    runfile = '/home/analysis/FIAC/fiac%d/fonc%d/fsl/fmristat_run/rho.img' % (subject, run)

    return Image(runfile)

def mask(subject=3, run=3, **extra):
    runfile = '/home/analysis/FIAC/fiac%d/fonc%d/fsl/mask.img' % (subject, run)

    return Image(runfile)

import optparse

parser = optparse.OptionParser()

parser.add_option('', '--which', help='contrasts or delays', dest='which',
                      default='contrasts')
parser.add_option('', '--run', help='which run?', dest='run', default=1, type='int')
parser.add_option('', '--subject', help='which subject?', dest='subject', default=0, type='int')
parser.add_option('', '--stat', help='t, sd or effect?', dest='stat', default='t')
parser.add_option('', '--contrast', help='overall, sentence, speaker, or interaction', dest='contrast', default='overall')
parser.add_option('', '--vmin', help='min for colorbar', dest='m', type='float')
parser.add_option('', '--vmax', help='max for colorbar', dest='M', type='float')
parser.add_option('', '--rho', help='compare AR(1) coefs', dest='rho', action='store_true', default=False)

options, args = parser.parse_args()
options = parser.values.__dict__

if not parser.values.rho:
    x = nipy_run(**options)
    y = fmristat_run(**options)
else:
    x = nipy_rho(**options)
    y = fmristat_rho(**options)
    
m = mask(**options)

vx = BoxViewer(x, mask=m, colormap='spectral')
vy = BoxViewer(y, mask=m, colormap='spectral')

print options
if options['m'] is not None:
    vx.m = options['m']; vy.m = options['m']

if options['M'] is not None:
    vx.M = options['M']; vy.M = options['M']

vx.draw(); vy.draw()

X = x.readall() * m.readall()
X.shape = N.product(X.shape)

Y = y.readall() * m.readall()
Y.shape = N.product(Y.shape)

print N.corrcoef(X,Y)

pylab.show()
