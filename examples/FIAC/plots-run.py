import os

import numpy as N
import pylab

from neuroimaging.utils.tests.data import repository
from neuroimaging.core.image import Image
from neuroimaging.algorithms.interpolation import ImageInterpolator
from neuroimaging.ui.visualization import slices, montage
from fixed import FIACresample

#standard = Image('avg152T1_brain.hdr', datasource=repository)
standard = Image('avg152T1.hdr', datasource=repository)

vmax = {'rho':0.7,
        'fwhmOLS':15.}
vmin = {'rho':-0.2,
        'fwhmOLS':5.}

options = {'run':3,
           'subj':3,
           'what':'rho'}

def FIACrunpath(resampled=False, what='rho', **opts):
    opts['what'] = what
    if not resampled:
        return '/home/analysis/FIAC/fiac%(subj)d/fonc%(run)d/fsl/fmristat_run/%(what)s.img' % opts
    else:
        return '/home/analysis/FIAC/fiac%(subj)d/fonc%(run)d/fsl/fmristat_run/%(what)s_rsmpl.img' % opts

def FIACfixedslice(**opts):

    print FIACrunpath(resampled=True, **opts)
    if not os.path.exists(FIACrunpath(resampled=True, **opts)) or options['force']:
        try:
            FIACresample(FIACrunpath(**opts), FIACrunpath(resampled=True, **opts), **opts)

            i = Image(FIACrunpath(resampled=True, **opts))
            haveit = True
        except:
            haveit = False
            pass
    else:
        i = Image(FIACrunpath(resampled=True, **opts))
        haveit = True

    if haveit:    
        i.grid = standard.grid

        slab = standard.grid.slab([35,0,0], [1,1,1], [1,109,91])
        i.grid = slab
        idata = i.readall()
        idata.shape = N.product(idata.shape)
        vmin = float(N.nanmin(idata))
        vmax = float(N.nanmax(idata))

        zslice = slices.transversal(i, z=-2, xlim=[-70,70], ylim=[-46,6], shape=(27,71))
        inter = ImageInterpolator(i)

        dslice = slices.PylabDataSlice(inter, zslice, vmin=vmin, vmax=vmax)
        return dslice

def FIACmontage(**opts):

    subjects = [0,1,3,4,6,7,8,9,10,11,12,13,14,15]
    dataslices = {}

    for i in range(len(subjects)):
        opts['subj'] = subjects[i]
        for j in range(1,5):
            opts['run'] = j 
            curslice = FIACfixedslice(**opts)
            if curslice:
                dataslices[(j-1,i)] = curslice
            else:
                dataslices[(j-1,i)] = None


    # determine colormap

    _vmin = vmin[opts['what']]
    _vmax = vmax[opts['what']]

    for i in range(len(subjects)):
        opts['subj'] = subjects[i]
        for j in range(1,5):
            opts['run'] = j
            if dataslices[(j-1,i)] is not None:
                dataslices[(j-1,i)].vmin = _vmin
                dataslices[(j-1,i)].vmax = _vmax
        
    datamontage = montage.Montage(slices=dataslices, AR=71./26, width=10.,
                                  vmax=_vmax, vmin=_vmin)
    datamontage.draw(redraw=False)
    return datamontage

if __name__ == '__main__':

    import optparse

    parser = optparse.OptionParser()

    parser.add_option('', '--what', help='rho or fwhmOLS?', dest='what', default='rho')
    parser.add_option('', '--force', help='force resampling?', dest='force', default=False,
                      action='store_true')

    options, args = parser.parse_args()
    options = parser.values.__dict__

    FIACmontage(**options)
    plot = FIACmontage(**options)

    outfile = '/home/analysis/FIAC/runs/%s.png' % options['what']
    outurl = 'http://kff.stanford.edu/FIAC/runs/%s.png' % options['what']

    pylab.savefig(outfile)
    pylab.close(plot.figure)
