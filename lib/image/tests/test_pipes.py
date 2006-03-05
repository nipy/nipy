import unittest, os, shutil
import neuroimaging.image.pipes as pipes

class PipeTest(unittest.TestCase):

    def test_analyze(self):
        p = pipes.URLPipe('http://kff.stanford.edu/~jtaylo/BrainSTAT/avg152T1.img')

    def test_repository(self):
        shutil.rmtree('/tmp/blah', ignore_errors=True)
        os.makedirs('/tmp/blah')
        self.img = pipes.URLPipe('http://kff.stanford.edu/~jtaylo/BrainSTAT/avg152T1.img', repository='/tmp/blah')
        shutil.rmtree('/tmp/blah')
        

if __name__ == '__main__':
    unittest.main()
