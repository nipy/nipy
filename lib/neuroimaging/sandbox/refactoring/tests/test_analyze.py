import unittest


from neuroimaging.defines import qt_def, pylab_def
QT_DEF, qt = qt_def()
PYLAB_DEF, pylab = pylab_def()

from neuroimaging.sandbox.refactoring.analyze import AnalyzeImage
from neuroimaging.utils.tests.data import repository
if PYLAB_DEF and QT_DEF:
    from neuroimaging.visualization.arrayview import arrayview

class AnalyzeImageTest(unittest.TestCase):

    def setUp(self):
        self.image = AnalyzeImage("rho", datasource=repository)

    def test_header(self):
        self.image.array

    if PYLAB_DEF and QT_DEF:
        def test_arrayview(self):
            arrayview(self.image.array)

if __name__ == '__main__': unittest.main()
