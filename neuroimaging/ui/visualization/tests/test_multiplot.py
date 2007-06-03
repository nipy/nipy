import numpy as N
from numpy.testing import NumpyTest, NumpyTestCase

from neuroimaging.utils.test_decorators import gui

from neuroimaging.defines import pylab_def
PYLAB_DEF, pylab = pylab_def()
if PYLAB_DEF:
    from neuroimaging.ui.visualization.multiplot import MultiPlot


class test_MultiPlot(NumpyTestCase):
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

        @gui
        def test1(self):
            self.fnplot.draw()
            pylab.show()
            
        @gui
        def test2(self):
            self.dataplot.draw()
            pylab.show()

        @gui
        def test3(self):
            self.fnplot.draw(args=(4,7))
            pylab.show()

        @gui
        def test4(self):
            self.fnplot.draw(args=(4,7), kw={'coefs':[2,6]})
            pylab.show()



from neuroimaging.utils.testutils import make_doctest_suite
test_suite = make_doctest_suite('neuroimaging.ui.visualization.multiplot')


if __name__ == '__main__':
    NumpyTest.run()
