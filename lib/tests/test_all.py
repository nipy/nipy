import unittest

import neuroimaging.data.tests
suite = neuroimaging.data.tests.suite()

import neuroimaging.reference.tests
suite.addTests(neuroimaging.reference.tests.suite())

import neuroimaging.statistics.tests
suite.addTests(neuroimaging.statistics.tests.suite())

import neuroimaging.image.tests
suite = unittest.TestSuite((suite, neuroimaging.image.tests.suite()))

import neuroimaging.fmri.tests 
suite = unittest.TestSuite((suite, neuroimaging.fmri.tests.suite()))

if __name__ == '__main__':
    import shutil, tempfile, os
    tmpdir = tempfile.mkdtemp(suffix='nipy_unittest')

    curdir = os.path.abspath(os.curdir)
    os.chdir(tmpdir)

    unittest.TextTestRunner(verbosity=2).run(suite)

    os.chdir(curdir)
    shutil.rmtree(tmpdir, ignore_errors=False)
