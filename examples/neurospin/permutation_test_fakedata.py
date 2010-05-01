import numpy as np
from nipy.neurospin.group import permutation_test as PT

def make_data(n=10,mask_shape=(10,10,10), axis=0, r=3, signal=5):
    """
    Generate Gaussian noise in a cubic volume
    + cubic actviations
    """
    mask = np.zeros(mask_shape,int)
    XYZ = np.array(np.where(mask==0))
    p = XYZ.shape[1]
    data = np.random.randn(n,p)
    I = np.where(np.square(XYZ - XYZ.max(axis=1).reshape(-1,1)/2).sum(axis=0) <= r**2 )[0]
    data[:, I] += signal
    vardata = np.random.randn(n,p)**2
    if axis==1:
        data = data.transpose()
        vardata = vardata.transpose()
    return data, vardata, XYZ

################################################################################
# Example for using permutation_test_onesample class
data, vardata, XYZ = make_data()
# rfx calibration
P = PT.permutation_test_onesample(data,XYZ)
# clusters definition (height threshold, max diameter)
c = [(P.random_Tvalues[P.ndraws * (0.95)], None), 
     (P.random_Tvalues[P.ndraws * (0.5)],    10)]
# regions definition (label vector)
r = np.ones(data.shape[1], int)
r[data.shape[1]/2.:] *= 10
voxel_results, cluster_results, region_results = \
                P.calibrate(nperms=100, clusters=c, regions=[r])
# mfx calibration
P = PT.permutation_test_onesample(data, XYZ, vardata=vardata,
                                  stat_id="student_mfx")
voxel_results, cluster_results, region_results = \
                P.calibrate(nperms=100, clusters=c, regions=[r])

################################################################################
# Example for using permutation_test_twosample class
data, vardata, XYZ = make_data(n=20)
data1, vardata1, data2, vardata2 = data[:10], vardata[:10], data[10:], vardata[10:]
# rfx calibration
P = PT.permutation_test_twosample(data1,data2,XYZ)
# clusters definition (height threshold / max diameter)
c = [(P.random_Tvalues[P.ndraws * (0.95)], None), 
     (P.random_Tvalues[P.ndraws * (0.5)], 10)]
# regions definition (label vector)
r = [np.zeros(data.shape[1])]
r[data.shape[1]/2:] *= 10
voxel_results, cluster_results, region_results = \
            P.calibrate(nperms=100, clusters=c, regions = r)
# mfx calibration
P = PT.permutation_test_twosample(data1, data2, XYZ, vardata1=vardata1, 
            vardata2=vardata2, stat_id="student_mfx")
voxel_results, cluster_results, region_results = \
            P.calibrate(nperms=100, clusters=c, regions=r)

################################################################################
# Print cluster statistics

level = 0.05

for results in cluster_results:
    nclust = results["labels"].max() + 1
    Tmax = np.zeros(nclust, float)
    Tmax_P = np.zeros(nclust, float)
    Diam = np.zeros(nclust, int)
    for j in xrange(nclust):
        I = np.where(results["labels"]==j)[0]
        Tmax[j] = P.Tvalues[I].max()
        Tmax_P[j] = voxel_results["Corr_p_values"][I].min()
        Diam[j]= PT.max_dist(XYZ,I,I)
    J = np.where(1 - (results["size_Corr_p_values"] > level)*(results["Fisher_Corr_p_values"] > level)*(Tmax_P > level))[0]
    print "\nDETECTED CLUSTERS STATISTICS:\n"
    print "Cluster detection threshold:", round(results["thresh"], 2)
    if results["diam"] != None:
        print "minimum cluster diameter", results["diam"]
    print "Cluster level FWER controled at", level
    #print "Peak".ljust(m+2), "Diameter".ljust(m), "Size".ljust(m), "Voxel".ljust(m), "Size".ljust(m), "Fisher".ljust(m)
    #print "XYZ".ljust(m+2), "".ljust(m), "".ljust(m), "P-Corr".ljust(m), "P-Corr".ljust(m), "P-Corr".ljust(m),"\n"
    for j in J:
            X, Y, Z = results["peak_XYZ"][:, j]
            strXYZ = str(X).zfill(2) + " " + str(Y).zfill(2) + " " + str(Z).zfill(2)
            #print strXYZ.ljust(m+2),
            #print str(round(Diam[j],2)).ljust(m),
            #print str(int(results["size_values"][j])).ljust(m),
            #print str(Tmax_P[j]).ljust(m),
            #print str(results["size_Corr_p_values"][j]).ljust(m),
            #print str(results["Fisher_Corr_p_values"][j]).ljust(m)












