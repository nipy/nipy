import unittest
import numpy as N
import scipy

from neuroimaging.fmri.utils import CutPoly, WaveFunction

class utilTest(unittest.TestCase):
    
    def test_CutPoly(self):
        f = CutPoly(2.0)
        t = N.arange(0, 10.0, 0.1)
        y = f(t)
        scipy.testing.assert_almost_equal(y, [x*x for x in t])

        f = CutPoly(2.0, (5, 7))
        y = f(t)
        scipy.testing.assert_almost_equal(y, [x*x*(x >= 5 and x < 7) for x in t])

        f = CutPoly(2.0, (None, 7))
        y = f(t)
        scipy.testing.assert_almost_equal(y, [x*x*(x < 7) for x in t])

        f = CutPoly(2.0, (5, None))
        y = f(t)
        scipy.testing.assert_almost_equal(y, [x*x*(x >= 5) for x in t])


    def test_WaveFunction(self):
        start = 5.0
        duration = 2.0
        height = 3.0
        f = WaveFunction(5, 2, 3)
        t = N.arange(0, 10.0, 0.1)
        y = f(t)
        scipy.testing.assert_almost_equal(y, [height*(x >= start and x < start + duration) for x in t])

if __name__ == '__main__':
    unittest.main()
