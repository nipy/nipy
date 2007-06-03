"""
Defines a class MultiPlot to plot multiple functions of time simultaneously.
"""

__docformat__ = 'restructuredtext'

from neuroimaging.defines import pylab_def
PYLAB_DEF, pylab = pylab_def()

if PYLAB_DEF:
    import numpy as N
    from neuroimaging import traits

    class MultiPlot(traits.HasTraits):
        """
        Class to plot multi-valued function of time or
        all columns of an array simultaneously.

        If 'data' is callable, then the time axis is
        linspace(tmin, tmax, (tmax-tmin)/dt).

        If args and kw are supplied they are passed as additional
        arguments to 'data' (if callable), with the first argument
        being time.

        """
    
        tmin = traits.Float(0., desc='Origin of time axis for MultiPlot')
        tmax = traits.Float(300., desc='End of time axis for MultiPlot')
        dt = traits.Float(0.2, desc='Time step for time axis of MultiPlot')
        figure = traits.Any()
        title = traits.Str()
        args = traits.Tuple(desc='Extra arguments if "data" is callable.')
        kw = traits.Dict(desc='Extra keyword arguments if "data" is callable.')

        def __init__(self, data, tmin=0., tmax=300., title='',
                     dt=0.2, args=(), kw={}, **keywords):
            self.tmin, self.tmax, self.dt = tmin, tmax, dt
            self.data = data
            self.figure = pylab.gcf()
            self.args = args
            self.kw = kw
            self.title = title

        def draw(self, t=None, args=(), kw={}):
            """
            Draw the multiplot.

            If self.data is callable, evaluate the functions at 't'. The
            args and kw arguments override self.args and self.kw if supplied.
            """
            
            args = args or self.args
            kw = kw or self.kw
            
            pylab.figure(num=self.figure.number)
            if t is None:
                t = N.linspace(self.tmin, self.tmax, (self.tmax - self.tmin) / self.dt)
            self.lines = []
            if callable(self.data):
                v = self.data(t, *args, **kw)
            else:
                v = self.data

            if v.ndim == 1:
                v.shape = (1, v.shape[0])
            if not callable(self.data):
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

            if self.title:
                pylab.title(self.title)

