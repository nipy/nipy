import pylab
import numpy as N

from neuroimaging.core.image.image import Image
from neuroimaging.algorithms.interpolation import ImageInterpolator
from neuroimaging.ui.visualization import slices, montage


standard = Image('http://kff.stanford.edu/FIAC/avg152T1_brain.img')

options = {'design':'block',
           'contrast':'overall',
           'which':'contrasts',
           'stat':'effect',
           'subj':3}

def FIACfixedpath(**opts):
    return 'http://kff.stanford.edu/FIAC/fiac%(subj)d/fixed/%(design)s/%(which)s/%(contrast)s/%(stat)s.img' % opts

def FIACfixedslice(**opts):

    i = Image(FIACfixedpath(**opts))
    i.grid = standard.grid

    idata = i.readall()
    idata.shape = N.product(idata.shape)
    vmin = float(N.nanmin(idata))
    vmax = float(N.nanmax(idata))

    zslice = slices.transversal(i, z=-2, xlim=[-70,70], ylim=[-46,6], shape=(27,71))
    inter = ImageInterpolator(i)

    dslice = slices.DataSlicePlot(inter, zslice, vmin=vmin, vmax=vmax)
    return dslice

def FIACmontage(**opts):

    subjects = [0,1,3,4,6,7,8,9,10,11,12,13,14,15]
    dataslices = {}

    vmin = []
    vmax = []
    for i in range(len(subjects)):
        opts['subj'] = subjects[i]
        dataslices[(0,i)] = FIACfixedslice(**opts)
        vmin.append(dataslices[(0,i)].vmin)
        vmax.append(dataslices[(0,i)].vmax)

    # determine colormap

    vmin = float(N.array(vmin).mean())
    vmax = float(N.array(vmax).mean())

    for i in range(len(subjects)):
        opts['subj'] = subjects[i]
        dataslices[(0,i)] = FIACfixedslice(**opts)
        dataslices[(0,i)].vmax = vmax
        dataslices[(0,i)].vmin = vmin
        
    datamontage = montage.Montage(slices=dataslices, AR=71./26, width=10.,
                                  vmax=vmax, vmin=vmin)
    datamontage.draw(redraw=False)
    return datamontage

if __name__ == '__main__':

    import optparse

    parser = optparse.OptionParser()

    parser.add_option('', '--design', help='block or event?', dest='design', default='block')
    parser.add_option('', '--which', help='contrasts or delays', dest='which',
                      default='contrasts')
    parser.add_option('', '--contrast', help='overall, sentence, speaker or interaction?', dest='contrast', default='overall')
    parser.add_option('', '--stat', help='t, sd or effect?', dest='stat',
                      default='effect')

    options, args = parser.parse_args()
        
    options = parser.values.__dict__

    plot = FIACmontage(**options)
    outfile = '/home/analysis/FIAC/multi/%(design)s/%(which)s/%(contrast)s/%(stat)s.png' % options
    outurl = 'http://kff.stanford.edu/FIAC/multi/%(design)s/%(which)s/%(contrast)s/%(stat)s.png' % options

    #pylab.savefig(outfile)
    pylab.close(plot.figure)
