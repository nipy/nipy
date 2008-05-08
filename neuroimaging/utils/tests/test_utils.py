from numpy.testing import NumpyTest, NumpyTestCase

class test_Template(NumpyTestCase):

    def setUp(self):
        pass
        #print "TestCase initialization..."

    def test_foo(self):
        self.fail('neuroimaging.utils, odict, path, etc... have _NO_ tests!')

if __name__ == '__main__':
    NumpyTest().run()
