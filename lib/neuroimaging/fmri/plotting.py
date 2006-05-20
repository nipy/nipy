"""
Defines a class MultiPlot to plot multiple functions of time simultaneously.
"""

import pylab
import numpy as N
from enthought import traits

class MultiPlot(traits.HasTraits):
    """
    Class to plot multi-valued function of time simultaneously.
    """
    
    tmin = traits.Float(0.)
    tmax = traits.Float(300.)
    dt = traits.Float(0.2)
    figure = traits.Any()
    title = traits.Str()

    def __init__(self, fn, **keywords):
        traits.HasTraits.__init__(self, **keywords)
        self.t = N.arange(self.tmin, self.tmax, self.dt)
        self.fn = fn
        self.figure = pylab.gcf()

    def draw(self, t=None, **keywords):
        pylab.figure(num=self.figure.number)
        if t is None:
            t = self.t
        self.lines = []
        if callable(self.fn):
            v = self.fn(time=t, **keywords)
        else:
            v = self.fn
        if v.ndim == 1:
            v.shape = (1, v.shape[0])
        if not callable(self.fn):
            t = N.arange(v.shape[1])
        v = v[::-1]
            
        n = v.shape[0]
        dy = 0.9 / n
        for i in range(n):
            a = pylab.axes([0.05,0.05+i*dy,0.9,dy])
            a.set_xticklabels([])
            a.set_yticks([])
            a.set_yticklabels([])
            m = N.nanmin(v[i])
            M = N.nanmax(v[i])
            pylab.plot(t, v[i])
            r = M - m
            l = m - 0.2 * r
            u = M + 0.2 * r
            if l == u:
                u += 1.
                l -= 1.
            a.set_ylim([l, u])

        pylab.title(self.title)

