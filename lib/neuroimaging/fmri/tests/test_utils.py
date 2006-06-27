import unittest
import numpy as N
import scipy

from neuroimaging.fmri.utils import CutPoly

class utilTest(unittest.TestCase):
    
    def test_CutPoly(self):
        p = CutPoly(2.0)
        t = N.arange(0, 10.0, 0.1)
        y = p(t)
        scipy.testing.assert_almost_equal(y, [x*x for x in t])

        p = CutPoly(2.0, (5, 7))
        y = p(t)
        scipy.testing.assert_almost_equal(y, [x*x*(x >= 5 and x < 7) for x in t])

        p = CutPoly(2.0, (None, 7))
        y = p(t)
        scipy.testing.assert_almost_equal(y, [x*x*(x < 7) for x in t])

        p = CutPoly(2.0, (5, None))
        y = p(t)
        scipy.testing.assert_almost_equal(y, [x*x*(x >= 5) for x in t])


if __name__ == '__main__':
    unittest.main()
