from os.path import join

import numpy as N

import pylab

from neuroimaging.core.api import Image, SamplingGrid
from neuroimaging.algorithms.interpolation import ImageInterpolator
from neuroimaging.ui.visualization import slices, montage

standard = Image('http://kff.stanford.edu/FIAC/avg152T1_brain.img')

import fixed, io, fiac
io.data_path='/home/analysis/FIAC'

class Slice:

    vmax = 5.
    vmin = -5.
    z = -2.
    zbuff = 2
    xlim = [-70,70]
    ylim = [-46,6]
    shape = (27,71)

    def __init__(self, image):

        self.image = image
        origin = N.array([self.z, N.mean(self.xlim), N.mean(self.ylim)])
        i = standard.grid.mapping.inverse()
        zvox = N.around(i(origin) - [self.zbuff,0,0])[0]

        grid = standard.grid.slab([zvox-self.zbuff,0,0],
                                  [1,1,1],
                                  [2*self.zbuff+1, standard.grid.shape[1], standard.grid.shape[2]])

        self.slab = Image(image[zvox:(zvox+2*self.zbuff+1)], grid=grid)
        self.interp = ImageInterpolator(self.slab)
        self.zslice = slices.transversal(self.slab.grid, z=self.z, xlim=self.xlim, ylim=self.ylim, shape=self.shape)
        self.dslice = slices.DataSlicePlot(self.interp, self.zslice)
        self.dslice.vmax = float(N.nanmax(self.slab[:]))
        self.dslice.vmin = float(N.nanmin(self.slab[:]))

class Fixed(fixed.Fixed):

    def __init__(self, stat='t', **args):
        fixed.Fixed.__init__(self, **args)
        self.stat = stat
        self._get_images()
        
    def _get_images(self):
        vmin = []; vmax = []
        self.images = {}
        for s in fiac.subjects:
            im = Image(self.resultpath(join('fiac%d' % s, '%s.nii' % self.stat)))
            self.images[s] = Slice(im)
            vmin.append(N.nanmin(self.images[s].slab[:])); vmax.append(N.nanmax(self.images[s].slab[:]))

        self.vmin = N.array(vmin).mean(); self.vmax = N.array(vmax).mean()
        
    def draw(self):
        """
        Draw montage of slices
        """

        dslices = {}
        for i in range(len(fiac.subjects)):
            dslices[(0,i)] = self.images[fiac.subjects[i]].dslice
            dslices[(0,i)].vmax = self.vmax
            dslices[(0,i)].vmin = self.vmin

        dmontage = montage.Montage(slices=dslices, AR=71./26, width=10.,
                                   vmax=self.vmax, vmin=self.vmin)
        dmontage.draw(redraw=False)
        return dmontage

    def output(self):
        pylab.savefig(self.resultpath('%s.png' % self.stat))


def run(contrast='average', which='contrasts', design='event'):
    for stat in ['effect', 'sd', 't']:
        v = Fixed(root=io.data_path,
                  stat=stat,
                  which=which,
                  contrast=contrast,
                  design=design)
        if stat == 't':
            v.vmax = 4.5; v.vmin = -4.5
        v.draw()
        v.output()
        
        htmlfile = file(v.resultpath("index.html"), 'w')
        htmlfile.write("""
        <!DOCTYPE HTML PUBLIC "-//IETF//DTD HTML//EN">
        <html> <head>
        <title></title>
        </head>
        
        <body>
        <h2>Contrast %s, %s design, %s</h2>
        <h3>Effect</h3>
        <img src="effect.png">
        <h3>SD</h3>
        <img src="sd.png">
        <h3>T</h3>
        <img src="t.png">
        </body>
        </html>
        """ % (contrast, design, {'contrasts': 'magnitude', 'delays':'delay'}[which]))
        htmlfile.close()
