import unittest, csv, os
import numpy as N
from scipy.sandbox.models.utils import recipr0
from scipy.sandbox.models import contrast

from neuroimaging.modalities.fmri import hrf, protocol, functions


class ProtocolTest(unittest.TestCase):

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
        for i in range(10):
            all.append([p[i], on[i], off[i]])
        self.all = all

    def setup_terms(self):
        self.p = protocol.ExperimentalFactor('pain', self.all, delta=False)
        self.p.convolved = False

        self.IRF1 = hrf.glover
        self.IRF2 = hrf.glover_deriv

        self.t = N.arange(0,300,1)

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

    def testContrast1(self):
        self.setup_terms()
        drift_fn = functions.SplineConfound(4, window=(0,300))
        drift = protocol.ExperimentalQuantitative('drift', drift_fn)
        formula = self.p + drift
        c = contrast.Contrast(self.p, formula)
        c.getmatrix(time=self.t)
        N.testing.assert_almost_equal(c.matrix,
                                          [[0.,0.,0.,0.,1.,0.],
                                           [0.,0.,0.,0.,0.,1.]])

    def testContrast2(self):
        self.setup_terms()
        drift_fn = functions.SplineConfound(4, window=(0,300))
        drift = protocol.ExperimentalQuantitative('drift', drift_fn)
        formula = self.p + drift
        c = contrast.Contrast(self.p.main_effect(), formula)
        c.getmatrix(time=self.t)
        N.testing.assert_almost_equal(c.matrix,
                                          [[0.,0.,0.,0.,1.,0.],
                                           [0.,0.,0.,0.,0.,1.]])

    def testDesign1(self):
        self.setup_terms()
        
        for df in range(4, 10):
            drift_fn = functions.SplineConfound(df, window=(0,300))
            drift = protocol.ExperimentalQuantitative('drift', drift_fn)

            formula = self.p + drift
            y = formula.design(time=self.t)
            self.assertEquals(y.shape[::-1], (2 + df, self.t.shape[0]))

            self.p.convolve(self.IRF1)
            self.p.convolved = True
            formula = self.p + drift
            y = formula.design(time=self.t)
            self.assertEquals(y.shape[::-1], (2 + df, self.t.shape[0]))

            self.p.convolve(self.IRF2)
            self.p.convolved = True
            formula = self.p + drift
            y = formula.design(time=self.t)
            self.assertEquals(y.shape[::-1], (4 + df, self.t.shape[0]))

            self.p.convolve(self.IRF1)
            formula = self.p + drift * self.p
            y = formula.design(time=self.t)
            self.assertEquals(y.shape[::-1], (2 + df * 2, self.t.shape[0]))

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
            self.convolved = True
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
        d = drift.astimefn()
        t = N.arange(0,30,1.)

        Z = float(N.random.standard_normal(()))

        D = d * Z
        N.testing.assert_almost_equal(D(t), N.array(drift_fn(t)) * Z)

        D = d / Z
        N.testing.assert_almost_equal(D(t), N.array(drift_fn(t)) / Z)

        D = d - Z
        N.testing.assert_almost_equal(D(t), N.array(drift_fn(t)) - Z)

        D = d + Z
        N.testing.assert_almost_equal(D(t), N.array(drift_fn(t)) + Z)


    def testTimeFn2(self):
        self.setup_terms()
        
        drift_fn = functions.SplineConfound(7, window=(0,300))
        drift = protocol.ExperimentalQuantitative('drift', drift_fn)
        d = drift.astimefn()
        t = N.arange(0,30,1.)

        D = d * d
        N.testing.assert_almost_equal(D(t), N.array(drift_fn(t))**2)

        D = d / d
        N.testing.assert_almost_equal(D(t), N.array(drift_fn(t)) * recipr0(drift_fn(t)))

        D = d - d
        N.testing.assert_almost_equal(D(t), N.zeros(D(t).shape, N.float64))

        D = d + d
        N.testing.assert_almost_equal(D(t), 2 * N.array(drift_fn(t)))

    def testTimeFn3(self):
        self.setup_terms()
        
        drift_fn = functions.SplineConfound(7, window=(0,300))
        drift = protocol.ExperimentalQuantitative('drift', drift_fn)
        d = drift.astimefn()
        t = N.arange(0,30,1.)

        n = d(t).shape[0]
        c = N.random.standard_normal((n,))

        i = N.random.random_integers(0, n-1)
        D = d * c
        N.testing.assert_almost_equal(D(t)[i], c[i] * N.array(drift_fn(t)[i]) )

        i = N.random.random_integers(0, n-1)
        D = d / c
        N.testing.assert_almost_equal(D(t)[i], N.array(drift_fn(t))[i] * recipr0(c)[i])

        i = N.random.random_integers(0, n-1)
        D = d - c
        N.testing.assert_almost_equal(D(t)[i], N.array(drift_fn(t))[i] - c[i])

        i = N.random.random_integers(0, n-1)
        D = d + c
        N.testing.assert_almost_equal(D(t)[i], N.array(drift_fn(t))[i] + c[i])


    def testTimeFn4(self):
        self.setup_terms()
        
        drift_fn = functions.SplineConfound(7, window=(0,300))
        drift = protocol.ExperimentalQuantitative('drift', drift_fn)
        d = drift.astimefn()

        t = N.arange(0,30,1.)
        n = d(t).shape[0]

        i = N.random.random_integers(0, n-1)
        x = d[i]

        N.testing.assert_almost_equal(N.squeeze(x(t)), d(t)[i])

    def testTimeFn5(self):
        t = N.arange(0,60,1.)
        self.setup_terms()
        q = self.p['hot']
        r = self.p['warm']
        b = q.astimefn()
        Z = float(N.random.standard_normal(()))

        a = q.astimefn() - r.astimefn() * Z
        N.testing.assert_almost_equal(a(t), (N.array(q(time=t)) - Z * N.array(r(time=t))).flatten())

class DeltaTest(unittest.TestCase):

    def test_DeltaFunction(self):
        a = N.arange(0,5,0.1)
        d = functions.DeltaFunction()
        d.start = 3.0
        d.dt = 0.5
        x = d(a)
        y = N.array(30*[0.] + 5*[2.] + 15*[0.])
        N.testing.assert_array_equal(x, y)


def suite():
    suite = unittest.makeSuite(ProtocolTest)
    return suite

if __name__ == '__main__':
    unittest.main()
