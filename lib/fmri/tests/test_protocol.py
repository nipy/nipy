import unittest, csv, os
import numpy as N
import scipy

from neuroimaging.fmri import hrf, protocol

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
        self.p = protocol.ExperimentalFactor('pain', self.all)
        IRF1 = hrf.HRF()
        self.pc1 = self.p.convolve(IRF1)

        IRF2 = hrf.HRF(deriv=True)
        self.pc2 = self.p.convolve(IRF2)

        self.t = N.arange(0,300,1)

    def testFromFile(self):
        out = file('tmp.csv', 'w')
        writer = csv.writer(out)
        for row in self.all:
            writer.writerow(row)
        out.close()
        
        p = protocol.ExperimentalFactor('pain', file('tmp.csv'))
        os.remove('tmp.csv')

    def testFromFileName(self):
        out = file('tmp.csv', 'w')
        writer = csv.writer(out)
        for row in self.all:
            writer.writerow(row)
        out.close()
        
        p = protocol.ExperimentalFactor('pain', 'tmp.csv')
        os.remove('tmp.csv')

    def testShape(self):
        self.setup_terms()
        
        for df in range(4, 10):
            drift_fn = protocol.SplineConfound(window=[0,300], df=df)
            drift = protocol.ExperimentalQuantitative('drift', drift_fn)

            formula = self.p + drift
            y = formula(self.t)
            self.assertEquals(y.shape, (2 + df, self.t.shape[0]))

            formula = self.pc1 + drift
            y = formula(self.t)
            self.assertEquals(y.shape, (2 + df, self.t.shape[0]))

            formula = self.pc2 + drift
            y = formula(self.t)
            self.assertEquals(y.shape, (4 + df, self.t.shape[0]))

            formula = self.p + drift * self.pc1
            y = formula(self.t)
            self.assertEquals(y.shape, (2 + df * 2, self.t.shape[0]))

    def testContrast1(self):
        self.setup_terms()
        drift_fn = protocol.SplineConfound(window=[0,300], df=4)
        drift = protocol.ExperimentalQuantitative('drift', drift_fn)
        formula = self.p + drift
        scipy.testing.assert_almost_equal(self.p.contrast(formula),
                                          [0.,0.,0.,0.,-1.,1.])

    def testContrast2(self):
        self.setup_terms()
        drift_fn = protocol.SplineConfound(window=[0,300], df=4)
        drift = protocol.ExperimentalQuantitative('drift', drift_fn)
        formula = self.p + drift
        C = self.p.contrast(formula, term=True)
        self.assertEquals(C(time=self.t).shape, (1,300))

    def testDesign(self):
        self.setup_terms()
        
        for df in range(4, 10):
            drift_fn = protocol.SplineConfound(window=[0,300], df=df)
            drift = protocol.ExperimentalQuantitative('drift', drift_fn)

            formula = self.p + drift
            y = formula.design(self.t)
            self.assertEquals(y.shape[::-1], (2 + df, self.t.shape[0]))

            formula = self.pc1 + drift
            y = formula.design(self.t)
            self.assertEquals(y.shape[::-1], (2 + df, self.t.shape[0]))

            formula = self.pc2 + drift
            y = formula.design(self.t)
            self.assertEquals(y.shape[::-1], (4 + df, self.t.shape[0]))

            formula = self.p + drift * self.pc1
            y = formula.design(self.t)
            self.assertEquals(y.shape[::-1], (2 + df * 2, self.t.shape[0]))

    def testDesign(self):
        self.setup_terms()
        
        for df in range(4, 10):
            drift_fn = protocol.SplineConfound(window=[0,300], df=df)
            drift = protocol.ExperimentalQuantitative('drift', drift_fn)

            formula = self.p + drift
            y = formula.names()
            self.assertEquals(len(y), 2 + df)

            formula = self.pc1 + drift
            y = formula.names()
            self.assertEquals(len(y), 2 + df)

            formula = self.pc2 + drift
            y = formula.names()
            self.assertEquals(len(y), 4 + df)

            formula = self.p + drift * self.pc1
            y = formula.names()
            self.assertEquals(len(y), 2 + df * 2)

if __name__ == '__main__':
    unittest.main()
