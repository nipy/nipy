from os.path import join
from gc import collect

import numpy as N

import pylab

from neuroimaging.core.api import Image, SamplingGrid
from neuroimaging.ui.visualization import slices, montage
from neuroimaging.core.reference.mapping import permutation_matrix, Affine

standard = Image('http://kff.stanford.edu/FIAC/avg152T1_brain.img')

import fixed, io, fiac
reload(fixed)
io.data_path='/home/analysis/FIAC'

xlim = [-90,90]
ylim = [-126,90]
zlim = [-74, 106]
shape = (91,109,91)

defaults = {'z':(slices.TransversalPlot, {'z':-2,
                                      'xlim':xlim,
                                      'ylim':ylim,
                                      'shape':shape}),
            'x':(slices.SagittalPlot, {'x':0,
                                       'zlim':zlim,
                                       'ylim':ylim,
                                       'shape':shape}),
            
            'y':(slices.CoronalPlot, {'y':20,
                                      'xlim':xlim,
                                      'zlim':zlim,
                                      'shape':shape}),
            'fixed':(slices.TransversalPlot, {'z':-2,
                                              'xlim':[-70,70],
                                              'ylim':[-46,6],
                                              'shape':(91,27,71)})
            }
            
class Slice:

    axis = 'z'

    colormap = 'spectral'
    vmax = 5.
    vmin = -5.

    def __init__(self, image, axis='z'):
        self.axis = axis
        self.image = image
        self.setup()
        
    def setup(self, args=defaults):

        self.plotmaker, self.defaults = args[self.axis]
        self.dslice = self.plotmaker(self.image, **self.defaults)
        self.dslice.colormap = self.colormap
        d = self.dslice.scalardata()
        self.dslice.vmax = float(N.nanmax(d))
        self.dslice.vmin = float(N.nanmin(d))

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
            fixed.set_transform(im)
            self.images[s] = Slice(im, axis='fixed')
            self.images[s].setup()
            self.images[s].dslice.transpose = True
            d = self.images[s].dslice.scalardata()
            vmin.append(N.nanmin(d)); vmax.append(N.nanmax(d))
            del(im)
            
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

    def __del__(self):
        del(self.images); collect()

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
    del(v)
