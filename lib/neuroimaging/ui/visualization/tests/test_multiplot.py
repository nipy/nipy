import unittest

import numpy as N

from neuroimaging.defines import pylab_def
PYLAB_DEF, pylab = pylab_def()
if PYLAB_DEF:
    from neuroimaging.ui.visualization.multiplot import MultiPlot


class MultiPlotTest(unittest.TestCase):
    if PYLAB_DEF:

        def setUp(self):
            def fn(x, *powers, **kw):
                x = N.asarray(x)
                if not powers:
                    powers = (1,2,3)
                v = N.array([x**i for i in powers])
                if kw.has_key('coefs'):
                    for i in range(v.shape[0]):
                        v[i] *= kw['coefs'][i]
                return v
                        
            self.fnplot = MultiPlot(fn, tmin=0, tmax=10., dt=0.1, title='Testing')
            self.dataplot = MultiPlot(fn(N.linspace(0,10,101)), tmin=0, tmax=10., dt=0.1, title='Testing')

        def test1(self):
            self.fnplot.draw()
            pylab.show()

        def test2(self):
            self.dataplot.draw()
            pylab.show()

        def test3(self):
            self.fnplot.draw(args=(4,7))
            pylab.show()

        def test4(self):
            self.fnplot.draw(args=(4,7), kw={'coefs':[2,6]})
            pylab.show()


def suite():
    return unittest.makeSuite(MultiPlotTest)


if __name__ == '__main__':
    unittest.main()
