import fpformat

import numpy as N
import pylab
from enthought import traits

from neuroimaging.visualization.cmap import cmap, interpolation, getcmap

class Montage(traits.HasTraits):
    
    slices = traits.Dict()

    ncol = traits.Int()
    nrow = traits.Int()

    interpolation = interpolation
    origin = traits.Trait('lower', 'upper')

    ybuf = traits.Float(0.05)
    xlbuf = traits.Float(0.01)
    xrbuf = traits.Float(0.09)
    AR = traits.Float(1.0)
    width = traits.Float(8)
    ndecimal = traits.Int(1)

    # for colorbar

    vmin = traits.Float()
    vmax = traits.Float()
    colormap = cmap

    def __init__(self, **keywords):

        traits.HasTraits.__init__(self, **keywords)
        if self.slices is None:
            raise ValueError, 'need slices for a Montage'

        if self.ncol == 0:
            self.ncol = N.array([key[1] for key in self.slices.keys()]).max() + 1

        if self.nrow == 0:
            self.nrow = N.array([key[0] for key in self.slices.keys()]).max() + 1

        self.shape = self.slices.values()[0].grid.shape
        test = []
        for _slice in self.slices.values():
            if _slice:
                test.append((_slice.grid.shape != self.shape))
        if N.sum(test) > 0:
            raise ValueError, 'all slices have to have the same shape in Montage'

        AR = self.AR * self.nrow / self.ncol
        self.figure = pylab.figure(figsize=(self.width,
                                            AR*self.width))

    def draw(self, redraw=False):
        pylab.figure(num=self.figure.number)

        data = {}
        for ij, _slice in self.slices.items():
            i, j = ij
            if _slice is not None:
                data[ij] = _slice.RGBA()

        if not redraw:
            self.axes = {}
            self.imshow = {}
            for ij, _slice in self.slices.items():
                i, j = ij
                dx = (1 - self.xlbuf - self.xrbuf) / self.ncol
                dy = (1 - 2. * self.ybuf) / self.nrow

                self.axes[ij] = pylab.axes([self.xlbuf + j * dx, 
                                            self.ybuf + i * dy,
                                            dx, dy])

                self.axes[ij].set_xticklabels([])
                self.axes[ij].set_yticks([])
                
                if _slice is not None:
                    pylab.imshow(_slice.RGBA())
                    pylab.show()
                    self.imshow[ij] = pylab.imshow(_slice.RGBA(),
                                                   interpolation=self.interpolation,
                                                   aspect='auto',
                                                   origin=self.origin)
            
            self.draw_colorbar()


        else:
            for ij, _slice in self.slices.keys():
                pylab.axes(self.axes[ij])
                self.imshow[ij].set_data(self.data)
            
        pylab.draw()

        
    def draw_colorbar(self):
 
        dy = (1 - 2. * self.ybuf) / self.nrow

        self.color_axes = pylab.axes([1 - 0.8 * self.xrbuf,
                                      self.ybuf,
                                      0.6 * self.xrbuf,
                                      dy])
        self.color_axes.set_xticks([])
        self.color_axes.set_yticks(range(0,120,20))
        self.color_axes.set_yticklabels([fpformat.fix(x, self.ndecimal) for x in N.linspace(self.vmin, self.vmax, 6)])
        

        v = N.linspace(self.vmin, self.vmax, 101)
        v = N.multiply.outer(v, N.ones((2,)))

        _cmap = getcmap(self.colormap)
        self.colorbar = pylab.imshow(v, vmin=self.vmin, vmax=self.vmax,
                                     cmap=_cmap,
                                     origin='lower')


