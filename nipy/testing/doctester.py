""" Custom doctester based on Numpy doctester
"""
import numpy as np

from ..fixes.numpy.testing.noseclasses import NumpyDoctest

class NipyDoctest(NumpyDoctest):
    name = 'nipydoctest'   # call nosetests with --with-nipydoctest

    def set_test_context(self, test):
        # set namespace for tests
        test.globs['np'] = np
