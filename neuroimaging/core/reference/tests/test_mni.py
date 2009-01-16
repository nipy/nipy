from neuroimaging.testing import *

import neuroimaging.core.reference.mni as mni

class test_MNI(TestCase):

    def testMNI(self):
        """ ensure all elementes of the interface exist """
        m_v = mni.MNI_voxel
        m_w = mni.MNI_world        
        m_m = mni.MNI_mapping






