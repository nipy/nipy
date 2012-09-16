# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""One and two sample permutation tests.
"""
# Third-party imports
import numpy as np
import scipy.misc as sm
import warnings

# Our own imports
from nipy.algorithms.graph import wgraph_from_3d_grid
from nipy.algorithms.graph.field import Field, field_from_graph_and_data

from ..utils import zscore 
from .onesample import stat as os_stat, stat_mfx as os_stat_mfx
from .twosample import stat as ts_stat, stat_mfx as ts_stat_mfx


# Default parameters
DEF_NDRAWS = int(1e5)
DEF_NPERMS = int(1e4)
DEF_NITER = 5
DEF_STAT_ONESAMPLE = 'student'
DEF_STAT_TWOSAMPLE = 'student'


#===========================================
#===========================================
# Cluster extraction functions
#===========================================
#===========================================


def extract_clusters_from_thresh(T,XYZ,th,k=18):
    """
    Extract clusters from statistical map
    above specified threshold
    In:  T      (p)     statistical map
         XYZ    (3,p)   voxels coordinates
         th     <float> threshold
         k      <int>   the number of neighbours considered. (6,18 or 26)
    Out: labels (p)     cluster labels
    """
    labels = -np.ones(len(T),int)
    I = np.where(T >= th)[0]
    if len(I)>0:
        SupraThreshXYZ = XYZ[:, I]
        CC_label = wgraph_from_3d_grid(SupraThreshXYZ.T, k).cc()
        labels[I] = CC_label
    return labels



def max_dist(XYZ,I,J):
    """
    Maximum distance between two set of points
    In:  XYZ (3,p) voxels coordinates
         I   (q)   index of points
         J   (r)   index of points
    Out: d <float>
    """
    if min(min(np.shape(I)), min(np.shape(J))) == 0:
        return 0
    else:
        # Square distance matrix
        D = np.sum(np.square(XYZ[:,I].reshape(3, len(I), 1) - XYZ[:, J].reshape(3, 1, len(J))), axis=0)
        return np.sqrt((D).max())


def extract_clusters_from_diam(T,XYZ,th,diam,k=18):
    """
    Extract clusters from a statistical map
    under diameter constraint
    and above given threshold
    In:  T      (p)     statistical map
         XYZ    (3,p)   voxels coordinates
         th     <float> minimum threshold
         diam   <int>   maximal diameter (in voxels)
         k      <int>   the number of neighbours considered. (6,18 or 26)
    Out: labels (p)     cluster labels

    Comment by alexis-roche, September 15th 2012: this function was
    originally developed by Merlin Keller in an attempt to generalize
    classical cluster-level analysis by subdividing clusters in blobs
    with limited diameter (at least, this is my understanding). This
    piece of code seems to have remained very experimental and its
    usefulness in real-world neuroimaging image studies is still to be
    demonstrated.
    """
    CClabels = extract_clusters_from_thresh(T,XYZ,th,k)
    nCC = CClabels.max() + 1
    labels = -np.ones(len(CClabels),int)
    # Calls _extract_clusters_from_diam, a recursive function, and
    # catches an exception if maximum recursion depth is reached
    try:
        labels = _extract_clusters_from_diam(labels, T, XYZ, th, diam, k,
                                             nCC, CClabels)
    except RuntimeError:
        warnings.warn('_extract_clusters_from_diam did not converge')
    return labels


def _extract_clusters_from_diam(labels, T, XYZ, th, diam, k,
                                nCC, CClabels):
    """ 
    This recursive function modifies the `labels` input array.
    """
    clust_label = 0
    for i in xrange(nCC):
        #print "Searching connected component ", i, " out of ", nCC
        I = np.where(CClabels==i)[0]
        extCC = len(I)
        if extCC <= (diam+1)**3:
            diamCC = max_dist(XYZ,I,I)
        else:
            diamCC = diam+1
        if diamCC <= diam:
            labels[I] = np.zeros(extCC,int) + clust_label
            #print "cluster ", clust_label, ", diam = ", diamCC
            #print "ext = ", len(I), ", diam = ", max_dist(XYZ,I,I)
            clust_label += 1
        else:
            # build the field
            p = len(T[I])
            F = field_from_graph_and_data(
                wgraph_from_3d_grid(XYZ[:, I].T, k), np.reshape(T[I],(p,1)))
            # compute the blobs
            idx, parent,label = F.threshold_bifurcations(0,th)
            nidx = np.size(idx)
            height = np.array([np.ceil(np.sum(label == i) ** (1./3)) 
                               for i in np.arange(nidx)])
            #root = nidx-1
            root = np.where(np.arange(nidx)==parent)[0]
            # Can constraint be met within current region?
            Imin = I[T[I]>=height[root]]
            extmin = len(Imin)
            if extmin <= (diam+1)**3:
                dmin = max_dist(XYZ,Imin,Imin)
            else:
                dmin = diam+1
            if dmin <= diam:# If so, search for the largest cluster meeting the constraint
                Iclust = Imin # Smallest cluster
                J = I[T[I]<height[root]] # Remaining voxels
                argsortTJ = np.argsort(T[J])[::-1] # Sorted by decreasing T values
                l = 0
                L = np.array([J[argsortTJ[l]]],int)
                diameter = dmin
                new_diameter = max(dmin,max_dist(XYZ,Iclust,L))
                while new_diameter <= diam:
                    #print "diameter = " + str(new_diameter)
                    #sys.stdout.flush()
                    Iclust = np.concatenate((Iclust,L))
                    diameter = new_diameter
                    #print "diameter = ", diameter
                    l += 1
                    L = np.array([J[argsortTJ[l]]],int)
                    new_diameter = max(diameter,max_dist(XYZ,Iclust,L))
                labels[Iclust] = np.zeros(len(Iclust),int) + clust_label
                #print "cluster ", clust_label, ", diam = ", diameter
                #print "ext = ", len(Iclust), ", diam = ", max_dist(XYZ,Iclust,Iclust)
                clust_label += 1
            else:# If not, search inside sub-regions
                #print "Searching inside sub-regions "
                Irest = I[T[I]>height[root]]
                rest_labels = extract_clusters_from_diam(T[Irest],XYZ[:,Irest],th,diam,k)
                rest_labels[rest_labels>=0] += clust_label
                clust_label = rest_labels.max() + 1
                labels[Irest] = rest_labels
    return labels


def extract_clusters_from_graph(T, G, th):
    """
    This returns a label vector of same size as T,
    defining connected components for subgraph of
    weighted graph G containing vertices s.t. T >= th
    """
    labels = np.zeros(len(T), int) - 1
    I = T >= th
    nlabels = I.sum()
    if nlabels > 0:
        labels[I] = G.subgraph(I).cc()
    return labels


#======================================
#======================================
# Useful functions
#======================================
#======================================



def sorted_values(a):
    """
    Extract list of distinct sortedvalues from an array
    """
    if len(a) == 0:
        return []
    else:
        m = min(a)
        L = [m]
        L.extend( sorted_values(a[a>m]) )
        return L



def onesample_stat(Y, V, stat_id, base=0.0, axis=0, Magics=None, niter=DEF_NITER):
    """
    Wrapper for os_stat and os_stat_mfx
    """
    if stat_id.find('_mfx')<0: 
        return os_stat(Y, stat_id, base, axis, Magics)
    else:
        return os_stat_mfx(Y, V, stat_id, base, axis, Magics, niter)

def twosample_stat(Y1, V1, Y2, V2, stat_id, axis=0, Magics=None, niter=DEF_NITER):
    """
    Wrapper for ts_stat and ts_stat_mfx
    """
    if stat_id.find('_mfx')<0: 
        return ts_stat(Y1, Y2, stat_id, axis, Magics)
    else:
        return ts_stat_mfx(Y1, V1, Y2, V2, stat_id, axis, Magics, niter)

#=================================================
#=================================================
# Compute cluster and region summary statistics
#=================================================
#=================================================



def compute_cluster_stats(Tvalues, labels, random_Tvalues, cluster_stats=["size","Fisher"]):
    """
    size_values, Fisher_values = compute_cluster_stats(Tvalues, labels, random_Tvalues, cluster_stats=["size","Fisher"])
    Compute summary statistics in each cluster
    In:  see permutation_test_onesample class docstring
    Out: size_values   Array of size nclust, or None if "size" not in cluster_stats
         Fisher_values Array of size nclust, or None if "Fisher" not in cluster_stats
    """
    nclust = max(labels)+1
    if nclust == 0:
        if "size" in cluster_stats:
            size_values =  np.array([0])
        else:
            size_values = None
        if "Fisher" in cluster_stats:
            Fisher_values = np.array([0])
        else:
            Fisher_values = None
    else:
        if "size" in cluster_stats:
            size_values = np.zeros(nclust,int)
        else:
            size_values = None
        if "Fisher" in cluster_stats:
            Fisher_values = np.zeros(nclust,float)
            ndraws = len(random_Tvalues)
            pseudo_p_values = 1 - np.searchsorted(random_Tvalues,Tvalues)/float(ndraws)
        else:
            Fisher_values = None
    for i in xrange(nclust):
        I = np.where(labels==i)[0]
        if "size" in cluster_stats:
            size_values[i] = len(I)
        if "Fisher" in cluster_stats:
            Fisher_values[i] = -np.sum(np.log(pseudo_p_values[I]))
    return size_values, Fisher_values



def compute_region_stat(Tvalues, labels, label_values, random_Tvalues):
    """
    Fisher_values = compute_region_stat(Tvalues, labels, label_values, random_Tvalues)
    Compute summary statistics in each cluster
    In:  see permutation_test_onesample class docstring
    Out: Fisher_values Array of size nregions
    """
    Fisher_values = np.zeros(len(label_values),float)
    pseudo_p_values = 1 - np.searchsorted(random_Tvalues,Tvalues)/float(len(random_Tvalues))
    for i in xrange(len(label_values)):
        I = np.where(labels==label_values[i])[0]
        Fisher_values[i] = -np.sum(np.log(pseudo_p_values[I]))
    return Fisher_values

def peak_XYZ(XYZ, Tvalues, labels, label_values):
    """
    Returns (3, n_labels) array of maximum T values coordinates for each label value
    """
    C = np.zeros((3, len(label_values)), int)
    for i in xrange(len(label_values)):
        I = np.where(labels == label_values[i])[0]
        C[:, i] = XYZ[:, I[np.argmax(Tvalues[I])]]
    return C

#======================================
#======================================
# Generic permutation test class
#======================================
#======================================

class permutation_test(object):
    """
    This generic permutation test class contains the calibration method
    which is common to the derived classes permutation_test_onesample and 
    permutation_test_twosample (as well as other common methods)
    """
    #=======================================================
    # Permutation test calibration of summary statistics
    #=======================================================
    def calibrate(self, nperms=DEF_NPERMS, clusters=None, 
                  cluster_stats=["size","Fisher"], regions=None, 
                  region_stats=["Fisher"], verbose=False):
        """
        Calibrate cluster and region summary statistics using permutation test

        Parameters
        ----------
        nperms : int, optional    
            Number of random permutations generated.
            Exhaustive permutations are used only if nperms=None,
            or exceeds total number of possible permutations
            
        clusters : list [(thresh1,diam1),(thresh2,diam2),...], optional
            List of cluster extraction pairs: (thresh,diam).  *thresh* provides
            T values threshold, *diam* is the maximum cluster diameter, in
            voxels.  Using *diam*==None yields classical suprathreshold
            clusters.
            
        cluster_stats : list [stat1,...], optional
            List of cluster summary statistics id (either 'size' or 'Fisher')
            
        regions : list [Labels1,Labels2,...] 
            List of region labels arrays, of size (p,) where p is the number 
            of voxels
            
        region_stats : list [stat1,...], optional
            List of cluster summary statistics id (only 'Fisher' supported 
            for now)
            
        verbose : boolean, optional
            "Chatterbox" mode switch

        Returns
        -------
        voxel_results : dict 
            A dictionary containing the following keys: ``p_values`` (p,)
            Uncorrected p-values.``Corr_p_values`` (p,) Corrected p-values,
            computed by the Tmax procedure.  ``perm_maxT_values`` (nperms)
            values of the maximum statistic under permutation.
        cluster_results : list [results1,results2,...] 
            List of permutation test results for each cluster extraction pair. 
            These are dictionaries with the following keys "thresh", "diam", 
            "labels", "expected_voxels_per_cluster", 
            "expected_number_of_clusters", and "peak_XYZ" if XYZ field is 
            nonempty and for each summary statistic id "S": "size_values", 
            "size_p_values", "S_Corr_p_values", "perm_size_values", 
            "perm_maxsize_values"
        region_results :list [results1,results2,...] 
            List of permutation test results for each region labels arrays. 
            These are dictionaries with the following keys: "label_values", 
            "peak_XYZ" (if XYZ field nonempty) and for each summary statistic 
            id "S": "size_values", "size_p_values", "perm_size_values", 
            "perm_maxsize_values"
        """
        # Permutation indices
        if self.nsamples ==1:
            n, p = self.data.shape[self.axis], self.data.shape[1-self.axis]
            max_nperms = 2**n
        elif self.nsamples == 2:
            n1,p = self.data1.shape[self.axis], self.data1.shape[1-self.axis]
            n2 = self.data2.shape[self.axis]
            max_nperms = sm.comb(n1+n2,n1,exact=1)
            data = np.concatenate((self.data1,self.data2), self.axis)
            if self.vardata1 != None:
                vardata = np.concatenate((self.vardata1,self.vardata2), self.axis)
        if nperms == None or nperms >= max_nperms:
            magic_numbers = np.arange(max_nperms)
        else:
            #magic_numbers = np.random.randint(max_nperms,size=nperms)
            # np.random.randint does not handle longint!
            # So we use the following hack instead:
            magic_numbers = np.random.uniform(max_nperms,size=nperms)
        # Initialize cluster_results
        cluster_results = []
        if clusters != None:
            for (thresh,diam) in clusters:
                if diam == None:
                    if self.XYZ == None:
                        labels = extract_clusters_from_graph(self.Tvalues,self.G,thresh)
                    else:
                        labels = extract_clusters_from_thresh(self.Tvalues,self.XYZ,thresh)
                else:
                    labels = extract_clusters_from_diam(self.Tvalues,self.XYZ,thresh,diam)
                results = {"thresh" : thresh, "diam" : diam, "labels" : labels}
                size_values, Fisher_values = compute_cluster_stats(self.Tvalues, labels, self.random_Tvalues, cluster_stats)
                nclust = labels.max() + 1
                results["expected_voxels_per_thresh"] = 0.0
                results["expected_number_of_clusters"] = 0.0
                if self.XYZ != None:
                    results["peak_XYZ"] = peak_XYZ(self.XYZ, self.Tvalues, labels, np.arange(nclust))
                if "size" in cluster_stats:
                    results["size_values"] = size_values
                    results["perm_size_values"] = []
                    results["perm_maxsize_values"] = np.zeros(len(magic_numbers),int)
                if "Fisher" in cluster_stats:
                    results["Fisher_values"] = Fisher_values
                    results["perm_Fisher_values"] = []
                    results["perm_maxFisher_values"] = np.zeros(len(magic_numbers),float)
                cluster_results.append( results )
        # Initialize region_results
        region_results = []
        if regions != None:
            for labels in regions:
                label_values = sorted_values(labels)
                nregions = len(label_values)
                results = { "label_values" : label_values }
                if self.XYZ != None:
                    results["peak_XYZ"] = peak_XYZ(self.XYZ, self.Tvalues, labels, label_values)
                if "Fisher" in region_stats:
                    results["Fisher_values"] = compute_region_stat(self.Tvalues, labels, label_values, self.random_Tvalues)
                    results["perm_Fisher_values"] = np.zeros((nregions,len(magic_numbers)),float)
                    results["Fisher_p_values"] = np.zeros(nregions,float)
                    results["Fisher_Corr_p_values"] = np.zeros(nregions,float)
                region_results.append( results )
        # Permutation test
        p_values = np.zeros(p,float)
        Corr_p_values = np.zeros(p,float)
        nmagic = len(magic_numbers)
        perm_maxT_values = np.zeros(nmagic, float)
        for j in xrange(nmagic):
            m = magic_numbers[j]
            if verbose:
                print "Permutation", j+1, "out of", nmagic
            # T values under permutation
            if self.nsamples == 1:
                #perm_Tvalues = onesample_stat(self.data, self.vardata, self.stat_id, self.base, self.axis, np.array([m]), self.niter).squeeze()
                rand_sign = (np.random.randint(2,size=n)*2-1).reshape(n,1)
                rand_data = rand_sign*self.data
                if self.vardata == None:
                    rand_vardata = None
                else:
                    rand_vardata = rand_sign*self.vardata
                perm_Tvalues = onesample_stat(rand_data, rand_vardata, self.stat_id, self.base, self.axis, None, self.niter).squeeze()
            elif self.nsamples == 2:
                perm_Tvalues = twosample_stat(self.data1, self.vardata1, self.data2, self.vardata2, self.stat_id, self.axis, np.array([m]), self.niter).squeeze()
                rand_perm = np.random.permutation(np.arange(n1+n2))
                rand_data1 = data[:n1]
                rand_data2 = data[n1:]
                if self.vardata1 == None:
                    rand_vardata1 = None
                    rand_vardata2 = None
                else:
                    rand_vardata1 = vardata[:n1]
                    rand_vardata2 = vardata[n1:]
            # update p values
            p_values += perm_Tvalues >= self.Tvalues
            Corr_p_values += max(perm_Tvalues) >= self.Tvalues
            perm_maxT_values[j] = max(perm_Tvalues)
            # Update cluster_results
            if clusters != None:
                for i in xrange(len(clusters)):
                    thresh, diam = clusters[i]
                    if diam == None:
                        if self.XYZ == None:
                            perm_labels = extract_clusters_from_graph(perm_Tvalues,self.G,thresh)
                        else:
                            perm_labels = extract_clusters_from_thresh(perm_Tvalues,self.XYZ,thresh)
                    else:
                        perm_labels = extract_clusters_from_diam(perm_Tvalues,self.XYZ,thresh,diam)
                    perm_size_values, perm_Fisher_values = compute_cluster_stats(perm_Tvalues, perm_labels, self.random_Tvalues, cluster_stats)
                    perm_nclust = labels.max() + 1
                    cluster_results[i]["expected_voxels_per_thresh"] += perm_size_values.sum()/float(nclust)
                    cluster_results[i]["expected_number_of_clusters"] += nclust
                    if "size" in cluster_stats:
                        cluster_results[i]["perm_size_values"][:0] = perm_size_values
                        cluster_results[i]["perm_maxsize_values"][j] = max(perm_size_values)
                    if "Fisher" in cluster_stats:
                        cluster_results[i]["perm_Fisher_values"][:0] = perm_Fisher_values
                        cluster_results[i]["perm_maxFisher_values"][j] = max(perm_Fisher_values)
            # Update region_results
            if regions != None:
                for i in xrange(len(regions)):
                    labels = regions[i]
                    label_values = region_results[i]["label_values"]
                    nregions = len(label_values)
                    if "Fisher" in region_stats:
                        perm_Fisher_values = compute_region_stat(perm_Tvalues, labels, label_values, self.random_Tvalues)
                        region_results[i]["perm_Fisher_values"][:,j] = perm_Fisher_values
        # Compute p-values for clusters summary statistics
        if clusters != None:
            for i in xrange(len(clusters)):
                if "size" in cluster_stats:
                    cluster_results[i]["perm_size_values"] = np.array(cluster_results[i]["perm_size_values"])
                    cluster_results[i]["perm_size_values"].sort()
                    cluster_results[i]["perm_maxsize_values"].sort()
                    cluster_results[i]["size_p_values"] = 1 - np.searchsorted(cluster_results[i]["perm_size_values"], cluster_results[i]["size_values"])/float(cluster_results[i]["expected_number_of_clusters"])
                    cluster_results[i]["size_Corr_p_values"] = 1 - np.searchsorted(cluster_results[i]["perm_maxsize_values"], cluster_results[i]["size_values"])/float(nmagic)
                if "Fisher" in cluster_stats:
                    cluster_results[i]["perm_Fisher_values"] = np.array(cluster_results[i]["perm_Fisher_values"])
                    cluster_results[i]["perm_Fisher_values"].sort()
                    cluster_results[i]["perm_maxFisher_values"].sort()
                    cluster_results[i]["Fisher_p_values"] = 1 - np.searchsorted(cluster_results[i]["perm_Fisher_values"], cluster_results[i]["Fisher_values"])/float(cluster_results[i]["expected_number_of_clusters"])
                    cluster_results[i]["Fisher_Corr_p_values"] = 1 - np.searchsorted(cluster_results[i]["perm_maxFisher_values"], cluster_results[i]["Fisher_values"])/float(nmagic)
                cluster_results[i]["expected_voxels_per_thresh"] /= float(nmagic)
                cluster_results[i]["expected_number_of_clusters"] /= float(nmagic)
        # Compute p-values for regions summary statistics
        if regions != None:
            for i in xrange(len(regions)):
                if "Fisher" in region_stats:
                    sorted_perm_Fisher_values = np.sort(region_results[i]["perm_Fisher_values"],axis=1)
                    label_values = region_results[i]["label_values"]
                    nregions = len(label_values)
                    # Compute uncorrected p-values
                    for j in xrange(nregions):
                        region_results[i]["Fisher_p_values"][j] = 1 - np.searchsorted(sorted_perm_Fisher_values[j],region_results[i]["Fisher_values"][j])/float(nmagic)
                    #Compute corrected p-values
                    perm_Fisher_p_values = np.zeros((nregions,nmagic),float)
                    for j in xrange(nregions):
                        I = np.argsort(region_results[i]["perm_Fisher_values"][j])
                        perm_Fisher_p_values[j][I] = 1 - np.arange(1,nmagic+1)/float(nmagic)
                    perm_min_Fisher_p_values = np.sort(perm_Fisher_p_values.min(axis=0))
                    region_results[i]["Fisher_Corr_p_values"] = 1 - np.searchsorted(-perm_min_Fisher_p_values,-region_results[i]["Fisher_p_values"])/float(nmagic)
        voxel_results = {'p_values':p_values/float(nmagic), 
                         'Corr_p_values':Corr_p_values/float(nmagic),
                         'perm_maxT_values':perm_maxT_values}
        return voxel_results, cluster_results, region_results



    def height_threshold(self, pval):
        """
        Return the uniform height threshold matching a given
        permutation-based P-value.
        """
        tvals = self.random_Tvalues
        ndraws = tvals.size
        idx = np.ceil(ndraws*(1-pval))
        if idx >= ndraws:
            return np.inf
        candidate = tvals[idx]
        if tvals[max(0, idx-1)]<candidate:
            return candidate
        idx = np.searchsorted(tvals, candidate, 'right')
        if idx >= ndraws:
            return np.inf
        return tvals[idx]


    def pvalue(self, Tvalues=None):
        """
        Return uncorrected voxel-level pseudo p-values.
        """
        if Tvalues == None:
            Tvalues = self.Tvalues
        return 1 - np.searchsorted(self.random_Tvalues, Tvalues)/float(self.ndraws)
        

    def zscore(self, Tvalues=None):
        """
        Return z score corresponding to the uncorrected
        voxel-level pseudo p-value.
        """
        if Tvalues == None: 
            Tvalues = self.Tvalues
        return zscore(self.pvalue(Tvalues))
 

#======================================
#======================================
# One sample permutation test class
#======================================
#======================================


class permutation_test_onesample(permutation_test):
    """
    Class derived from the generic permutation_test class.
    Inherits the calibrate method
    """
    
    def __init__(self, data, XYZ, axis=0, vardata=None, 
                 stat_id=DEF_STAT_ONESAMPLE, base=0.0, niter=DEF_NITER,
                 ndraws=DEF_NDRAWS):
        """
        Initialize permutation_test_onesample instance,
        compute statistic values in each voxel and under permutation
        In:  data                data array
             XYZ                 voxels coordinates
                 axis    <int>       Subject axis in data
             vardata             variance (same shape as data)
                         optional (if None, mfx statistics cannot be used)
             stat_id <char>      choice of test statistic
                                 (see onesample.stats for a list of possible stats)
             base    <float>     mean signal under H0
             niter   <int>       number of iterations of EM algorithm
             ndraws  <int>       Number of generated random t values
        Out:
             self.Tvalues        voxelwise test statistic values
             self.random_Tvalues sorted statistic values in random voxels and under random
                                 sign permutation
        """
        # Create data fields
        n,p = data.shape[axis], data.shape[1-axis]
        self.data = data
        self.stat_id = stat_id
        self.XYZ = XYZ
        self.axis = axis
        self.vardata = vardata
        self.niter = niter
        self.base = base
        self.ndraws = ndraws
        self.Tvalues = onesample_stat(data, vardata, stat_id, base, axis, Magics=None, niter=niter).squeeze()
        self.nsamples = 1
        # Compute statistic values in random voxels and under random permutations
        # Use a self.verbose flag for this output?
        #print "Computing average null distribution of test statistic..."
        self.random_Tvalues = np.zeros(ndraws,float)
        # Random voxel selection
        I = np.random.randint(0,p,size=ndraws)
        if axis == 0:
            rand_data = data[:,I]
            if vardata == None:
                rand_vardata = None
            else:
                rand_vardata = vardata[:,I]
        else:
            rand_data = data[I]
            if vardata == None:
                rand_vardata = None
            else:
                rand_vardata = vardata[I]
        # Random sign permutation
        rand_sign = (np.random.binomial(1,0.5,size = n*ndraws)*2-1).reshape(n,ndraws)
        if axis == 1:
            rand_sign = rand_sign.transpose()
        rand_data *= rand_sign
        self.random_Tvalues = onesample_stat(rand_data, rand_vardata, stat_id, base, axis).squeeze()
        self.random_Tvalues.sort()



#==================================================================
#==================================================================
# One sample permutation test class with arbitrary graph structure
#==================================================================
#==================================================================


class permutation_test_onesample_graph(permutation_test):
    """
    Class derived from the generic permutation_test class.
    Inherits the calibrate method
    """
    
    def __init__(self,data,G,axis=0,vardata=None,stat_id=DEF_STAT_ONESAMPLE,base=0.0,niter=DEF_NITER,ndraws=DEF_NDRAWS):
        """
        Initialize permutation_test_onesample instance,
        compute statistic values in each voxel and under permutation
        In:  data                data array
             G                   weighted graph (each vertex corresponds to a voxel)
             axis    <int>       Subject axis in data
             vardata             variance (same shape as data)
                         optional (if None, mfx statistics cannot be used)
             stat_id <char>      choice of test statistic
                                 (see onesample.stats for a list of possible stats)
             base    <float>     mean signal under H0
             niter   <int>       number of iterations of EM algorithm
             ndraws  <int>       Number of generated random t values
        Out:
             self.Tvalues        voxelwise test statistic values
             self.random_Tvalues sorted statistic values in random voxels and under random
                                 sign permutation
        """
        # Create data fields
        n,p = data.shape[axis], data.shape[1-axis]
        self.data = data
        self.stat_id = stat_id
        self.XYZ = None
        self.G = G
        self.axis = axis
        self.vardata = vardata
        self.niter = niter
        self.base = base
        self.ndraws = ndraws
        self.Tvalues = onesample_stat(data, vardata, stat_id, base, axis, Magics=None, niter=niter).squeeze()
        self.nsamples = 1
        # Compute statistic values in random voxels and under random permutations
        # Use a self.verbose flag for this output?
        #print "Computing average null distribution of test statistic..."
        self.random_Tvalues = np.zeros(ndraws,float)
        # Random voxel selection
        I = np.random.randint(0,p,size=ndraws)
        if axis == 0:
            rand_data = data[:,I]
            if vardata == None:
                rand_vardata = None
            else:
                rand_vardata = vardata[:,I]
        else:
            rand_data = data[I]
            if vardata == None:
                rand_vardata = None
            else:
                rand_vardata = vardata[I]
        # Random sign permutation
        rand_sign = (np.random.binomial(1,0.5,size = n*ndraws)*2-1).reshape(n,ndraws)
        if axis == 1:
            rand_sign = rand_sign.transpose()
        rand_data *= rand_sign
        self.random_Tvalues = onesample_stat(rand_data, rand_vardata, stat_id, base, axis).squeeze()
        self.random_Tvalues.sort()


#======================================
#======================================
# Two sample permutation test class
#======================================
#======================================


class permutation_test_twosample(permutation_test):
    """
    Class derived from the generic permutation_test class.
    Inherits the calibrate method
    """
    def __init__(self,data1,data2,XYZ,axis=0,vardata1=None,vardata2=None,stat_id=DEF_STAT_TWOSAMPLE,niter=DEF_NITER,ndraws=DEF_NDRAWS):
        """
        Initialize permutation_test_twosample instance,
        compute statistic values in each voxel and under permutation
        In:  data1, data2        data arrays
             XYZ                 voxels coordinates
                 axis    <int>       Subject axis in data
             vardata1, vardata2  variance (same shape as data)
                         optional (if None, mfx statistics cannot be used)
             stat_id <char>      choice of test statistic
                                 (see onesample.stats for a list of possible stats)
             niter   <int>       number of iterations of EM algorithm
             ndraws  <int>       Number of generated random t values
        Out:
             self.Tvalues        voxelwise test statistic values
             self.random_Tvalues sorted statistic values in random voxels and under random
                                 sign permutation
        """
        # Create data fields
        n1,p = data1.shape[axis], data1.shape[1-axis]
        n2 = data2.shape[axis]
        self.data1 = data1
        self.data2 = data2
        self.stat_id = stat_id
        self.XYZ = XYZ
        self.axis = axis
        self.vardata1 = vardata1
        self.vardata2 = vardata2
        self.niter = niter
        self.ndraws = ndraws
        self.Tvalues = twosample_stat(data1, vardata1, data2, vardata2, stat_id, axis, Magics=None, niter=niter).squeeze()
        self.nsamples = 2
        # Compute statistic values in random voxels and under random permutations
        # Use a self.verbose flag for this output?
        #print "Computing average null distribution of test statistic..."
        self.random_Tvalues = np.zeros(ndraws,float)
        # Random voxel selection
        I = np.random.randint(0,p,size=ndraws)
        if axis == 0:
            perm_data = np.zeros((n1+n2,ndraws),float)
            perm_data[:n1] = data1[:,I]
            perm_data[n1:] = data2[:,I]
            if vardata1 != None:
                perm_vardata = np.zeros((n1+n2,ndraws),float)
                perm_vardata[:n1] = vardata1[:,I]
                perm_vardata[n1:] = vardata2[:,I]
        else:
            perm_data = np.zeros((ndraws,n1+n2),float)
            perm_data[:,:n1] = data1[I]
            perm_data[:,n1:] = data2[I]
            if vardata1 != None:
                perm_vardata = np.zeros((ndraws, n1+n2),float)
                perm_vardata[:,:n1] = vardata1[I]
                perm_vardata[:,n1:] = vardata2[I]
        rand_perm = np.array([np.random.permutation(np.arange(n1+n2)) for i in xrange(ndraws)]).transpose()
        ravel_rand_perm = rand_perm*ndraws + np.arange(ndraws).reshape(1,ndraws)
        if axis == 0:
            perm_data = perm_data.ravel()[ravel_rand_perm.ravel()].reshape(n1+n2,ndraws)
            if vardata1 != None:
                perm_vardata = perm_vardata.ravel()[ravel_rand_perm.ravel()].reshape(n1+n2,ndraws)
        else:
            perm_data = (perm_data.transpose().ravel()[ravel_rand_perm.ravel()].reshape(n1+n2,ndraws)).transpose()
            if vardata1 != None:
                perm_vardata = (perm_vardata.transpose().ravel()[ravel_rand_perm.ravel()].reshape(n1+n2,ndraws)).transpose()
        perm_data1 = perm_data[:n1]
        perm_data2 = perm_data[n1:]
        if vardata1 == None:
            perm_vardata1 = None
            perm_vardata2 = None
        else:
            perm_vardata1 = perm_vardata[:n1]
            perm_vardata2 = perm_vardata[n1:]
        self.random_Tvalues = twosample_stat(perm_data1, perm_vardata1, perm_data2, perm_vardata2, stat_id, axis).squeeze()
        self.random_Tvalues.sort()
