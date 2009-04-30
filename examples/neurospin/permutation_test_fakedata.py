import numpy as np
from fff2.group import permutation_test as PT

def make_data(n=10,mask_shape=(10,10,10),axis=0):
    """
    Generate Gaussian noise in a cubic volume
    """
    mask = np.zeros(mask_shape,int)
    ZYX = np.array(np.where(mask==0))
    p = ZYX.shape[1]
    data = np.random.randn(n,p)
    vardata = np.random.randn(n,p)**2
    if axis==1:
        data = data.transpose()
        vardata = vardata.transpose()
    return data, vardata, ZYX

################################################################################
# Example for using permutation_test_onesample class
data, vardata, ZYX = make_data()
# rfx calibration
P = PT.permutation_test_onesample(data,ZYX)
# clusters definition (height threshold, max diameter)
c = [(P.random_Tvalues[P.ndraws * (0.95)], None), 
     (P.random_Tvalues[P.ndraws * (0.5)],    10)]
# regions definition (label vector)
r = np.ones(data.shape[1], int)
r[data.shape[1]/2.:] *= 10
voxel_results, cluster_results, region_results = \
                P.calibrate(nperms=100, clusters=c, regions=[r])
# mfx calibration
P = PT.permutation_test_onesample(data, ZYX, vardata=vardata,
                                  stat_id="student_mfx")
voxel_results, cluster_results, region_results = \
                P.calibrate(nperms=100, clusters=c, regions=[r])

################################################################################
# Example for using permutation_test_twosample class
data, vardata, ZYX = make_data(n=20)
data1, vardata1, data2, vardata2 = data[:10], vardata[:10], data[10:], vardata[10:]
# rfx calibration
P = PT.permutation_test_twosample(data1,data2,ZYX)
# clusters definition (height threshold / max diameter)
c = [(P.random_Tvalues[P.ndraws * (0.95)], None), 
     (P.random_Tvalues[P.ndraws * (0.5)], 10)]
# regions definition (label vector)
r = [np.zeros(data.shape[1])]
r[data.shape[1]/2:] *= 10
voxel_results, cluster_results, region_results = \
            P.calibrate(nperms=100, clusters=c, regions = r)
# mfx calibration
P = PT.permutation_test_twosample(data1, data2, ZYX, vardata1=vardata1, 
            vardata2=vardata2, stat_id="student_mfx")
voxel_results, cluster_results, region_results = \
            P.calibrate(nperms=100, clusters=c, regions=r)
