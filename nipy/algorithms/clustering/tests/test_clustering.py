from __future__ import absolute_import
#!/usr/bin/env python

# to run only the simple tests:
# python testClustering.py Test_Clustering

from ..utils import kmeans
import nose
import numpy as np
import numpy.random as nr
from unittest import TestCase


class TestClustering(TestCase):
    def testkmeans1(self):
        X = nr.randn(10, 2)
        A = np.concatenate([np.ones((7, 2)),np.zeros((3, 2))])
        X = X + 3 * A;
        L = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        C, L, J = kmeans(X, 2, L)
        self.assert_(np.mean(L[:7]) < 0.5)

    def testkmeans2(self):
        X = nr.randn(10000, 2)
        A = np.concatenate([np.ones((7000, 2)), np.zeros((3000, 2))])
        X = X + 3 * A
        L = np.concatenate([np.ones(5000), np.zeros(5000)]).astype(np.int)
        C, L, J = kmeans(X, 2, L)
        l = L[:7000].astype(np.float)
        self.assert_(np.mean(l) > 0.9)


if __name__ == '__main__':
    nose.run(argv=['', __file__])
