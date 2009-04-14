import csv, os
import numpy as np
from nipy.testing import *
from nipy.fixes.scipy.stats.models.utils import recipr0
from nipy.fixes.scipy.stats.models import contrast

from nipy.modalities.fmri import hrf, protocol, functions


class test_ProtocolSetup(TestCase):

    def setUp(self):
        """
        Setup an iterator corresponding to the following .csv file:

        hot,9.0,18.0
        warm,27.0,36.0
        hot,45.0,54.0
        warm,63.0,72.0
        hot,81.0,90.0
        warm,99.0,108.0
        hot,117.0,126.0
        warm,135.0,144.0
        hot,153.0,162.0
        warm,171.0,180.0
        hot,189.0,198.0
        warm,207.0,216.0
        hot,225.0,234.0
        warm,243.0,252.0
        hot,261.0,270.0
        warm,279.0,288.0
        hot,297.0,306.0
        warm,315.0,324.0
        hot,333.0,342.0
        warm,351.0,360.0

        """

        on = [9.0, 27.0, 45.0, 63.0, 81.0, 99.0, 117.0,
              135.0, 153.0, 171.0, 189.0, 207.0, 225.0,
              243.0, 261.0, 279.0, 297.0, 315.0, 333.0, 351.0]
        off = [18.0, 36.0, 54.0, 72.0, 90.0, 108.0, 126.0,
               144.0, 162.0, 180.0, 198.0, 216.0, 234.0,
               252.0, 270.0, 288.0, 306.0, 324.0, 342.0, 360.0]
        p = ['hot', 'warm'] * 10
        all = []
        for i in range(20):
            all.append([p[i], on[i], off[i]])
        self.all = all

    def setup_terms(self):
        self.p = protocol.ExperimentalFactor('pain', self.all, delta=False)
        self.p.convolved = False

        self.IRF1 = hrf.glover
        self.IRF2 = hrf.glover_deriv

        self.t = np.arange(0,300,1)

class test_Protocol(test_ProtocolSetup):

    def testFromFile(self):
        out = file('tmp.csv', 'w')
        writer = csv.writer(out)
        for row in self.all:
            writer.writerow(row)
        out.close()
        
        p = protocol.ExperimentalFactor('pain', file('tmp.csv'), delta=False)
        os.remove('tmp.csv')


    def testFromFileName(self):
        out = file('tmp.csv', 'w')
        writer = csv.writer(out)
        for row in self.all:
            writer.writerow(row)
        out.close()
        
        p = protocol.ExperimentalFactor('pain', 'tmp.csv', delta=False)
        os.remove('tmp.csv')


    # FIXME: Fix recursion error: c =
    #     contrast.Contrast(self.p.main_effect(), formula) File
    #     "/Users/cburns/src/nipy-trunk/nipy/modalities/fmri/protocol.py",
    #     line 307, in main_effect return
    #     ExperimentalQuantitative('%s:maineffect' % self.termname, f)
    #     File
    #     "/Users/cburns/src/nipy-trunk/nipy/modalities/fmri/protocol.py",
    #     line 139, in __init__ test =
    #     np.array(self.func(np.array([4.0,5.0,6]))) File
    #     "/Users/cburns/src/nipy-trunk/nipy/modalities/fmri/protocol.py",
    #     line 305, in <lambda> f = lambda t: f(t)
    @dec.knownfailure
    def testContrast2(self):
        self.setup_terms()
        drift_fn = functions.SplineConfound(4, window=(0,300))
        drift = protocol.ExperimentalQuantitative('drift', drift_fn)
        formula = self.p + drift
        c = contrast.Contrast(self.p.main_effect(), formula)
        c.compute_matrix(self.t)
        assert_almost_equal(c.matrix,
                                          [[0.,0.,0.,0.,1.,0.],
                                           [0.,0.,0.,0.,0.,1.]])

    @dec.slow
    def testDesign2(self):
        self.setup_terms()
        
        for df in range(4, 10):
            drift_fn = functions.SplineConfound(df, window=(0,300))
            drift = protocol.ExperimentalQuantitative('drift', drift_fn)

            formula = self.p + drift
            y = formula.names()
            self.assertEquals(len(y), 2 + df)

            formula = self.p + drift
            y = formula.names()
            self.assertEquals(len(y), 2 + df)

            self.p.convolve(self.IRF2)
            formula = self.p + drift
            y = formula.names()
            self.assertEquals(len(y), 4 + df)

            self.p.convolve(self.IRF1)
            formula = self.p + drift * self.p
            y = formula.names()
            self.assertEquals(len(y), 2 + df * 2)

    def testTimeFn1(self):
        self.setup_terms()
        
        drift_fn = functions.SplineConfound(7, window=(0,300))
        drift = protocol.ExperimentalQuantitative('drift', drift_fn)
        t = np.arange(0,30,1.)

        Z = float(np.random.standard_normal(()))

        D = lambda t: drift(t) * Z
        assert_almost_equal(D(t), np.array(drift_fn(t)) * Z)

        D = lambda t: drift(t) / Z
        assert_almost_equal(D(t), np.array(drift_fn(t)) / Z)

        D = lambda t: drift(t) - Z
        assert_almost_equal(D(t), np.array(drift_fn(t)) - Z)

        D = lambda t: drift(t) + Z
        assert_almost_equal(D(t), np.array(drift_fn(t)) + Z)


    def testTimeFn2(self):
        self.setup_terms()
        
        drift_fn = functions.SplineConfound(7, window=(0,300))
        drift = protocol.ExperimentalQuantitative('drift', drift_fn)
        d = lambda t: drift(t)
        t = np.arange(0,30,1.)

        D = lambda t: d(t) * d(t)
        assert_almost_equal(D(t), np.array(drift_fn(t))**2)



    def test_DeltaFunction(self):
        a = np.arange(0,5,0.1)
        d = functions.DeltaFunction()
        d.start = 3.0
        d.dt = 0.5
        x = d(a)
        y = np.array(30*[0.] + 5*[2.] + 15*[0.])
        assert_array_equal(x, y)



    def testContrast1(self):
        self.setup_terms()
        drift_fn = functions.SplineConfound(4, window=(0,300))
        drift = protocol.ExperimentalQuantitative('drift', drift_fn)
        formula = self.p + drift
        c = contrast.Contrast(self.p, formula)
        c.compute_matrix(self.t)
        assert_almost_equal(c.matrix,
                                          [[0.,0.,0.,0.,1.,0.],
                                           [0.,0.,0.,0.,0.,1.]])


    @dec.slow
    def testShape(self):
        self.setup_terms()
        
        for df in range(4, 10):
            drift_fn = functions.SplineConfound(df, window=(0,300))
            drift = protocol.ExperimentalQuantitative('drift', drift_fn)

            formula = self.p + drift
            y = formula(self.t)
            self.assertEquals(y.shape, (2 + df, self.t.shape[0]))

            self.p.convolve(self.IRF1)
            formula = self.p + drift
            y = formula(self.t)
            self.assertEquals(y.shape, (2 + df, self.t.shape[0]))

            self.p.convolve(self.IRF2)
            formula = self.p + drift
            y = formula(self.t)
            self.assertEquals(y.shape, (4 + df, self.t.shape[0]))

            self.p.convolve(self.IRF1)
            formula = self.p + drift * self.p
            y = formula(self.t)
            self.assertEquals(y.shape, (2 + df * 2, self.t.shape[0]))


    @dec.slow
    def testDesign1(self):
        self.setup_terms()
        
        for df in range(4, 10):
            drift_fn = functions.SplineConfound(df, window=(0,300))
            drift = protocol.ExperimentalQuantitative('drift', drift_fn)

            formula = self.p + drift
            y = formula.design(self.t)
            self.assertEquals(y.shape[::-1], (2 + df, self.t.shape[0]))

            self.p.convolve(self.IRF1)
            self.p.convolved = True
            formula = self.p + drift
            y = formula.design(self.t)
            self.assertEquals(y.shape[::-1], (2 + df, self.t.shape[0]))

            self.p.convolve(self.IRF2)
            self.p.convolved = True
            formula = self.p + drift
            y = formula.design(self.t)
            self.assertEquals(y.shape[::-1], (4 + df, self.t.shape[0]))

            self.p.convolve(self.IRF1)
            formula = self.p + drift * self.p
            y = formula.design(self.t)
            self.assertEquals(y.shape[::-1], (2 + df * 2, self.t.shape[0]))

    @dec.slow
    def test_toggle(self):
	"""
	Test to make sure that .convolved flag works on regression terms.
	"""

        self.setup_terms()

        for df in range(4, 10):

            drift_fn = functions.SplineConfound(df, window=(0,300))
            drift = protocol.ExperimentalQuantitative('drift', drift_fn)

	    self.p.convolved = False
            formula = self.p + drift * self.p
            y = formula.design(self.t)
            self.assertEquals(y.shape[::-1], (2 + df * 2, self.t.shape[0]))

            self.p.convolve(self.IRF2)
            formula = self.p + drift * self.p
            y = formula.design(self.t)
            self.assertEquals(y.shape[::-1], (4 + df * 4, self.t.shape[0]))

	    self.p.convolved = False
            formula = self.p + drift * self.p
            y = formula.design(self.t)
            self.assertEquals(y.shape[::-1], (2 + df * 2, self.t.shape[0]))






