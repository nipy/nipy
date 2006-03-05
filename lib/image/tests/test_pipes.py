import unittest, os, shutil
import neuroimaging.image.pipes as pipes
import neuroimaging.image as image

class PipeTest(unittest.TestCase):

    def setUp(self):
        self.url = 'http://kff.stanford.edu/~jtaylo/BrainSTAT/avg152T1.img'

    def tearDown(self):
        shutil.rmtree('/tmp/blah', ignore_errors=True)

    def test_download(self):
        self.img = image.Image(self.url, repository='/tmp/blah')
        test = os.path.exists('/tmp/blah/kff.stanford.edu/~jtaylo/BrainSTAT/avg152T1.img')
        self.assertEqual(test, True)
        test = os.path.exists('/tmp/blah/kff.stanford.edu/~jtaylo/BrainSTAT/avg152T1.hdr')
        self.assertEqual(test, True)
        return 
        

if __name__ == '__main__':
    unittest.main()
