"""Test file for the ordered dictionary module, odict.py."""

from neuroimaging.externals.scipy.testing import *

from neuroimaging.utils.odict import odict

class TestOdict(TestCase):
    def setUp(self):
        print 'setUp'
        self.thedict = odict((('one', 1.0), ('two', 2.0), ('three', 3.0)))

    def test_copy(self):
        """Test odict.copy method."""
        print self.thedict
        cpydict = self.thedict.copy()
        assert cpydict == self.thedict
        # test that it's a copy and not a reference
        assert cpydict is not self.thedict
        
if __name__ == "__main__":
    nose.runmodule()
