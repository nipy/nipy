"""
This script just create a list of subject matching a certain criterion,
fetches and read their activation images for a certain contrast.
Then the dataset is subject to jacknife subampling ('splitting'),
each subsample being analysed independently.
A reproducibility measure is then derived;

This is a temporeary script: it should provides functions
for a future 'reproducibility analysis' modumes

Bertrand Thirion, 2009.
"""

import nifti
import numpy as np
import os.path as op
from read_profil import read_profil
import nipy.neurospin.graph as fg


# ---------------------------------------------------------
# ----- cluster handling functions ------------------------
# ---------------------------------------------------------
#from roc import cluster_threshold, get_cluster_position_from_thresholded_map

def cluster_threshold(map,ijk,th,csize):
    """
    perform a thresholding of a map

    INPUT:
    - map: array of shape(nbvox)
    - ijk: array of shape(nbvox,3):
    the set of associated coordinates
    - th (float): cluster-forming threshold
    - cisze (int): cluster size threshold

    OUTPUT:
    -binary=array of shape (nvox): the thresholded map
    """
    ithr = np.nonzero(map>th)[0]
    binary = np.zeros(np.size(map)).astype('i')
    
    if np.size(ithr)>0:
        G = fg.WeightedGraph(np.size(ithr))
        G.from_3d_grid(ijk[ithr,:],18)
        
        # get the connected components
        label = G.cc()+1 
        binary[ithr] = label
        
        #remove the small clusters
        for i in range(label.max()+1):
            ji = np.nonzero(label==i)[0]
            if np.size(ji)<csize: binary[ji]=0

        binary = (binary>0)
    return binary


def get_cluster_position_from_thresholded_map(smap, ijk, coord, thr=3.0, csize=10):
    """
    the clusters above thr of size greater than csize in
    18-connectivity are computed

    INPUT:
    - smap : array of shape (nbvox)
    - ijk array of shape(nbvox,anat_dim) grid coordinates
    - coord: array of shape (nbvox,anatdim) physical ccordinates
    - thr=3.0 cluster-forming threshold
    - cisze=10: cluster size threshold

    output:
    - positions arrau of shape(k,anat_dim)
    the cluster positions in physical coordinates
    where k= number of clusters
    
    NOTE: if no such cluster exists, None is returned
    """

    # if no supra-threshold voxel, return
    ithr = np.nonzero(smap>thr)[0]
    if np.size(ithr)==0:
        return None

    # first build a graph
    g = fg.WeightedGraph(np.size(ithr))
    g.from_3d_grid(ijk[ithr,:],18)
    
    # get the connected components
    label = g.cc()
    baryc = []
    for i in range(label.max()+1):
        ji = np.nonzero(label==i)[0]
        if np.size(ji)>=csize:
            idx = ithr[ji]
            baryc.append(np.mean(coord[idx],0))

    if len(baryc)==0:
        return None

    baryc = np.vstack(baryc)
    return baryc


# -------------------------------------------------------
# ---------- The main functions -----------------------------
# -------------------------------------------------------



def splitgroup(nbsubj,groupsize):
    """
    Split the proposed group into disjoint subgroups
    INPUT:
    - nbsubj (int) the number of subjects to be split
    - groupsize(int) the size of each subbgroup
    OUPUT:
    - samples: a list of nb_subgroups arrey containing
    the indexes of the subjects in each sungroup
    """
    subgroups = int(np.floor(nbsubj/groupsize))
    rperm = np.argsort(np.random.rand(nbsubj))
    samples= [rperm[i*groupsize:(i+1)*groupsize] for i in range(subgroups)]
    return samples

def ttest(x):
    """
    returns the t-test for each row of the data x
    """
    import fff2.group.onesample as fos
    t = fos.stat(x.T,id='student',axis=0)
    return np.squeeze(t)
    #t = x.mean(1)/x.std(1)*np.sqrt(x.shape[1]-1)
    #return t

def fttest(x,vx):
    """
    returns a combined ('fixed') t-test of the data
    """
    n = x.shape[1]
    t = x/np.sqrt(vx)
    t = t.mean(1)*np.sqrt(n)
    return t
    
def mfx_ttest(x,vx):
    """
    returns the mixed effects t-test for each row of the data x
    and the associated variance vx
    """
    import fff2.group.onesample as fos
    t = fos.stat_mfx(x.T,vx.T,id='student_mfx',axis=0)
    return np.squeeze(t)

def voxel_thresholded_ttest(x,threshold):
    """
    returns a binary map of the ttest>threshold
    """
    t = ttest(x)
    return t>threshold

def statistics_from_position(target,data,sigma=1.0):
    """
    return a couple statistics charcterizing how close data is from
    target

    INPUT
    - target: rray of shape(nt,anat_dim) the target positions
    or None
    - data: array of shape(nd,anat_dim) the data position
    or None
    - sigma=1.0 (float): a distance that say how good good is 

    OUTPUT:
    - sensitivity (float): how well the targets are fitted
    by the data  in [0,1] interval
    1 is good
    0 is bad
    """
    from fff2.eda.dimension_reduction import Euclidian_distance as ed
    if data==None:
        if target==None:
            return 1.
        else:
            return 0.
    if target==None:
        return 0.
    
    dmatrix = ed(data,target)/sigma
    sensitivity = dmatrix.min(0)
    sensitivity = np.exp(-0.5*sensitivity**2)
    sensitivity = np.mean(sensitivity)
    return sensitivity

def voxel_reproducibility(data,vardata,groupsize,xyz,method='rfx',niter=0,verbose=0,**kwargs):
    """
    return a measure of voxel-level reproducibility
    of activation patterns

    INPUT:
    - data: array of shape (nvox,nsubj)
    the input data from which everything is computed
    - vardata: the corresponding variance information
    (same size) 
    - groupsize (int): the size of each subrgoup to be studied
    - threshold (float): binarization threshold
    (makes sense only if method==rfx)
    - method='rfx' inference method under study
    or 'crfx'
    - niter=0: number of iterations. potentially used to store
    intermediate data
    - verbose=0 : verbosity mode
    OUPUT:
    - kappa (float): the desired  reproducibility index
    """
    nbsubj = data.shape[1]
    nvox = data.shape[0]
    samples = splitgroup(nbsubj,groupsize)
    subgroups = len(samples)
    rmap = np.zeros(nvox)
    for i in range(subgroups):
        x = data[:,samples[i]]
        vx = vardata[:,samples[i]]
        if method=='rfx':
            threshold = kwargs['threshold']
            rmap += voxel_thresholded_ttest(x,threshold)        
        if method=='crfx':
            csize = kwargs['csize']
            threshold = kwargs['threshold']
            rfx = ttest(x)
            rmap += cluster_threshold(rfx,xyz,threshold,csize)>0
        if method=='cffx':
            csize = kwargs['csize']
            threshold = kwargs['threshold']
            ffx = fttest(x,vx)
            rmap += cluster_threshold(ffx,xyz,threshold,csize)>0
        if method=='cmfx':
            csize = kwargs['csize']
            threshold = kwargs['threshold']
            rfx = mfx_ttest(x,vx)
            rmap += cluster_threshold(rfx,xyz,threshold,csize)>0
        if method not in['rfx','crfx','cmfx','cffx']:
            raise ValueError, 'unknown method'

    import two_binomial_mixture as mtb
    MB = mtb.TwoBinomialMixture()
    MB.estimate_parameters(rmap,subgroups)
    if verbose:
        h = np.array([np.sum(rmap==i) for i in range(subgroups+1)])
        MB.show(h)
    return MB.kappa()


def cluster_reproducibility(data,vardata,groupsize,xyz,coord,sigma, method='crfx', niter=0,verbose=0,**kwargs):
    """
    return a measure of cluster-level reproducibility
    of activation patterns
    (i.e. how far clusters are from each other)

    INPUT:
    - data: array of shape (nvox,nsubj)
    the input data from which everything is computed
    - vardata: array of shape (nvox,nsubj)
    the variance of the data that is also available
    - groupsize (int): the size of each subrgoup to be studied
    - xyz array of shape (nvox,3) providing the grid ccordinates
    of the voxels
    - coord: array of shape (nvox,3) that provides the
    corresponding physical coordinates
    - sigma (float): parameter that encodes how far far is
    - threshold (float): binarization threshold
    (makes sense only if method==rfx)
    - method='rfx' inference method under study
    'rfx' or 'crfx'
    - niter = 0: (int) iteration number
    this is used to save intermediate results
    - verbose=0 : verbosity mode
    """
    tiny = 1.e-15
    nbsubj = data.shape[1]
    samples = splitgroup(nbsubj,groupsize)
    subgroups = len(samples)
    if subgroups==1:
        return 1.
    all_pos = []
    for i in range(subgroups):
        x = data[:,samples[i]]
        vx = vardata[:,samples[i]]
        tx = x/(tiny+np.sqrt(vx))
        if method=='crfx':
            rfx = ttest(x)
            csize = kwargs['csize']
            threshold = kwargs['threshold']
            pos = get_cluster_position_from_thresholded_map\
                  (rfx, xyz, coord, threshold, csize)
            all_pos.append(pos)
        if method=='cmfx':
            mfx = mfx_ttest(x,vx)
            csize = kwargs['csize']
            threshold = kwargs['threshold']
            pos = get_cluster_position_from_thresholded_map\
                  (mfx, xyz, coord, threshold, csize)
            all_pos.append(pos)
        if method=='cffx':
            ffx = fttest(x,vx)
            csize = kwargs['csize']
            threshold = kwargs['threshold']
            pos = get_cluster_position_from_thresholded_map\
                  (ffx, xyz, coord, threshold, csize)
            all_pos.append(pos)
        if method=='bsa':
            afname = kwargs['afname']
            header = kwargs['header']
            theta = kwargs['theta']
            dmax = kwargs['dmax']
            ths = kwargs['ths']
            thq = kwargs['thq']
            smin = kwargs['smin']
            afname = afname+'_%02d_%04d.pic'%(niter,i)
            pos = coord_bsa(xyz, coord, tx, header, theta, dmax,
                            ths, thq, smin,afname)
            all_pos.append(pos)

    score = 0
    for i in range(subgroups):
        for j in range(i):
            score += statistics_from_position(all_pos[i],all_pos[j],sigma)
            score += statistics_from_position(all_pos[j],all_pos[i],sigma)
            
    score /= (subgroups*(subgroups-1))
    return score



# -------------------------------------------------------
# ---------- BSA stuff -----------------------------
# -------------------------------------------------------

def coord_bsa(xyz, coord, betas, header, theta=3., dmax =  5., ths = 0, thq = 0.5, smin = 0,afname='/tmp/af.pic'):
    """
    main function for  performing bsa on a set of images
    """
    import fff2.spatial_models.bayesian_structural_analysis as bsa
    import fff2.graph.field as ff
    import  pickle
    
    nbvox = np.shape(xyz)[0]

    # create the field strcture that encodes image topology
    Fbeta = ff.Field(nbvox)
    Fbeta.from_3d_grid(xyz.astype(np.int),18)

    # volume density
    voxsize =  header['pixdim'][1:4] # fragile !
    # or np.absolute(np.diag(header['sform'])[:3]) ?
    g0 = 1.0/(np.prod(voxsize)*nbvox)

    crmap,AF,BF,p = bsa.compute_BSA_simple (Fbeta,betas,coord,dmax,xyz,header,thq, smin,ths, theta,g0,verbose=0)
    if AF==None:
        return None
    pickle.dump(AF, open(afname, 'w'), 2)
    afcoord = AF.discrete_to_roi_features('position')
    return afcoord

