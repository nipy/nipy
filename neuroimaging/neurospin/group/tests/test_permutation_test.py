import unittest

import numpy as np

from neuroimaging.neurospin.group import permutation_test as PT
import neuroimaging.neurospin.graph as fg

nperms = 2
ndraws = 10

def make_data(n=10,mask_shape=(10,10,10),axis=0):
    mask = np.zeros(mask_shape,int)
    XYZ = np.array(np.where(mask==0))
    p = XYZ.shape[1]
    data = np.random.randn(n,p)
    vardata = np.random.randn(n,p)**2
    if axis==1:
        data = data.transpose()
        vardata = vardata.transpose()
    return data, vardata, XYZ

class test_permutation_test(unittest.TestCase):
    
    def test_onesample(self):
        data, vardata, XYZ = make_data()
        # rfx calibration
        P = PT.permutation_test_onesample(data, XYZ, ndraws=ndraws)
        c = [(P.random_Tvalues[P.ndraws*(0.95)],None),(P.random_Tvalues[P.ndraws*(0.5)],10)]
        r = np.ones(data.shape[1],int)
        r[data.shape[1]/2:] *= 10
        #p_values, cluster_results, region_results = P.calibrate(nperms=100, clusters=c, regions=[r])
        # mfx calibration
        P = PT.permutation_test_onesample(data, XYZ, vardata=vardata, stat_id="student_mfx", ndraws=ndraws)
        p_values, cluster_results, region_results = P.calibrate(nperms=nperms, clusters=c, regions=[r])
        
    def test_onesample_graph(self):
        data, vardata, XYZ = make_data()
        A, B, D = fg.graph_3d_grid(XYZ.transpose())
        V = max(A) + 1
        E = len(A)
        edges = np.concatenate((A.reshape(E, 1), B.reshape(E, 1)), axis=1)
        weights = D
        G = fg.WeightedGraph(V, edges, weights)
        # rfx calibration
        P = PT.permutation_test_onesample_graph(data, G, ndraws=ndraws)
        c = [(P.random_Tvalues[P.ndraws*(0.95)],None)]
        r = np.ones(data.shape[1],int)
        r[data.shape[1]/2:] *= 10
        #p_values, cluster_results, region_results = P.calibrate(nperms=100, clusters=c, regions=[r])
        # mfx calibration
        P = PT.permutation_test_onesample_graph(data, G, vardata=vardata, stat_id="student_mfx", ndraws=ndraws)
        p_values, cluster_results, region_results = P.calibrate(nperms=nperms, clusters=c, regions=[r])
        
    def test_twosample(self):
        data, vardata, XYZ = make_data(n=20)
        data1,vardata1,data2,vardata2 = data[:10],vardata[:10],data[10:],vardata[10:]
        # rfx calibration
        P = PT.permutation_test_twosample(data1, data2, XYZ, ndraws=ndraws)
        c = [(P.random_Tvalues[P.ndraws*(0.95)],None),(P.random_Tvalues[P.ndraws*(0.5)],10)]
        r = [np.zeros(data.shape[1])]
        r[data.shape[1]/2:] *= 10
        #p_values, cluster_results, region_results=P.calibrate(nperms=100, clusters=c, regions=r)
        # mfx calibration
        P = PT.permutation_test_twosample(data1, data2, XYZ, vardata1=vardata1, vardata2=vardata2, stat_id="student_mfx", ndraws=ndraws)
        p_values, cluster_results, region_results = P.calibrate(nperms=nperms, clusters=c, regions=r)

if __name__ == "__main__":
    unittest.main()
