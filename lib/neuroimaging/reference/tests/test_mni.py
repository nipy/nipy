import unittest
import numpy as N

import neuroimaging.reference.mni as mni

class MNITest(unittest.TestCase):

    def testMNI(self):
        """ ensure all elementes of the interface exist """
        m = mni.MNI
        g = mni.generic
        m_v = mni.MNI_voxel
        m_w = mni.MNI_world        
        m_m = mni.MNI_mapping

if __name__ == '__main__':
    unittest.main()
