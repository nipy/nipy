"""
The main routine of this package that aims at performing the
extraction of ROIs from multisubject dataset using the localization.
This has been puclished in Thirion et al. Structural Analysis of fMRI
Data Revisited: Improving the Sensitivity and Reliability of fMRI
Group Studies.  IEEE TMI 2007

Author : Bertrand Thirion, 2006-2008
"""

#autoindent 

import numpy as np
import numpy.random as nr


# import nipy.neurospin.clustering.clustering as fc
import nipy.neurospin.graph.BPmatch as BPmatch
import nipy.neurospin.graph.field as ff
import nipy.neurospin.graph.graph as fg
from nipy.neurospin.spatial_models import hroi 
from nipy.neurospin.clustering.hierarchical_clustering import \
    Average_Link_Graph_segment

class landmark_regions(hroi.NROI):
    """
    This class is intended to represent a set of inter-subject regions
    besides the standard multiple ROI features, its has
    features 'subjects_ids' and 'individual_positions' to describe
    from which subjects at which position it is found.
    """
    def __init__(self,k,parents=None,header=None,id=None,subj=None,coord=None):
        """
        Building the landmark_region

        INPUT
        - k (int): number of nodes/landmarks included into it
        nb :
        - parents = None: array of shape(self.k) describing the
        hierarchical relationship
        if None parents = np.arange(k) is sued instead
        - header (temporary): space-defining image header
        to embed the structure in an image space
        - subj=None: k-length list of subjects
        (these correspond to ROI feature)
        - coord=None; k-length list of  coordinate arrays
        CAVEAT:
        discrete is used here for subject identification
        """
        k = int(k)
        if k<1: raise ValueError, "cannot create an empty LR"
        if parents==None:
            parents = np.arange(k)
        hroi.NROI.__init__(self,parents,header,discrete=subj,id=id)
        self.set_discrete_feature('position',coord)
        self.subj = self.discrete

    def centers(self):
        """
        c = self.centers()
        returns the average of the coordinates for each region
        """
        centers = self.discrete_to_roi_features('position')
        return centers

    def homogeneity(self):
        """ returns the mean distance between points within each LR
        """
        import nipy.neurospin.eda.dimension_reduction  as dr 
        coord = self.discrete_features['position']
        size = self.get_size()
        h = np.zeros(self.k)
        for k in range(self.k):
             edk = dr.Euclidian_distance(coord[k]) 
             h[k] = edk.sum()/(size[k]*(size[k]-1))
        return h
             
    def HPD(self,k,cs,pval = 0.95,dmax=1.0):
        """
        i = self.HPD(cs,dmax = 10,pval = 0.95)
        Sample the postreior density of being in k
        on a grid defined by cs, assuming that the roi is an ellipsoid
        INPUT:
        - cs: an array of shape(n,dim) a set of input coordinates
         - pval=0.95 cutoff for the CR
         - dmax=1.0 : an upper bound for the spatial variance
        to avoid degenerate variance
        OUPUT:
        - hpd array of shape(n) that yields the value
        """
        if k>self.k:
            raise ValueError, 'wrong region index'
        
        coord = self.discrete_features['position'][k]
        centers = self.discrete_to_roi_features('position')
        dim = centers.shape[1]
        
        if cs.shape[1]!=dim:
            raise ValueError, "incompatible dimensions"
        
        dx = coord-centers[k]
        covariance = np.dot(np.transpose(dx),dx)/coord.shape[0]
        import numpy.linalg as L
        U,S,V = L.svd(covariance,0)
        sqrtS = np.sqrt(1/np.maximum(S,dmax**2))
        dx = cs-centers[k]
        dx = np.dot(dx,U)
        dx = np.dot(dx,np.diag(sqrtS))
        delta = np.sum(dx**2,1)
        lcst = -np.log(2*np.pi)*dim/2+(np.log(sqrtS)).sum()
        hpd = np.exp(lcst-delta/2)

        import scipy.special as sp
        gamma = 2*sp.erfinv(pval)**2
        hpd[delta>gamma]=0
        return hpd

    def map_label(self,cs,pval = 0.95,dmax=1.):
        """
        i = self.map_label(cs,pval = 0.95,dmax=1.0)
        Sample the set of landmark regions
        on the proposed coordiante set cs, assuming a Gaussian shape
        INPUT:
        - cs: an array of size(n*p) a set of input coordinates
        - dmax=10 : an upper bound for the spatial variance
        to avoid degenerate variance
        - pval=0.95 cutoff for the CR
        OUPUT:
        i = set of entries of the coordinates that are within the cr
        """
        label = -np.ones(cs.shape[0])
        if self.k>0:
            aux = -np.ones((cs.shape[0],self.k))
            for k in range(self.k):
                aux[:,k] = self.HPD(k,cs,pval,dmax)

            maux = np.max(aux,1)
            label[maux>0] = np.argmax(aux,1)[maux>0]
        return label

    def show(self):
        """function to print basic information on self
        """
        centers = self.discrete_to_roi_features('position')
        homogeneity = self.homogeneity()
        for i in range(self.k):
            print i, np.unique(self.subj[i]), homogeneity[i], centers[i]

def build_LR(BF,ths=0):
    """
    Given a list of hierarchical ROIs, and an associated labelling, this
    creates an Amer structure wuch groups ROIs with the same label.
    INPUT:
    - BF is the list of hierarchical ROIs.
    it is assumd that each list corresponds to one subject
    the ROIs are supposed to be labelled
    - ths=0 defines the condition (c):
    A label should be present in ths subjects in order to be valid
    OUTPUT:
    - LR : a list of Amers, each of which describing
    a cross-subject set of ROIs
    - newlabel :  a relabelling of the individual ROIs,
    similar to u, which converts old indexes to new ones

    NOTE:
    if no LR can be created then LR=None as an output argument
    """
    nbsubj = np.size(BF)
    subj = [s*np.ones(BF[s].k) for s in range(nbsubj) if BF[s]!=None]
    subj = np.concatenate(subj).astype(np.int)
    u = [BF[s].get_roi_feature('label') for s in range(nbsubj) if BF[s]!=None]
    u = np.squeeze(np.concatenate(u))

    if np.size(u)==0: return None,None
    nrois = np.size(subj)
    intrasubj = np.concatenate([np.arange(BF[s].k) for s in range(nbsubj) if BF[s]!=None])
   
    coords = []
    subjs = []
    
    # LR-defining algorithm 
    Mu = int(u.max()+1)
    valid = np.zeros(Mu).astype(np.int)
    
    for i in range(Mu):
        j = np.nonzero(u==i)
        j = np.reshape(j,np.size(j))
        q = 0
        if np.size(j)>1:
            q = np.size(np.unique(subj[j]))
            
        if  (q>ths):
            valid[i]=1
            sj = np.size(j)
            coord = np.zeros((sj,3),'d')
            for a in range(sj):
                sja = subj[j[a]]
                isja = intrasubj[j[a]]
                coord[a,:] = BF[sja].get_roi_feature('position')[isja]
            coords.append(coord)
            subjs.append(subj[j])

    maplabel = -np.ones(Mu).astype(np.int)
    maplabel[valid>0] = np.cumsum(valid[valid>0])-1
    k = np.sum(valid)
    
    # relabel the ROIs
    for s in range(nbsubj):
        if BF[s]!=None:
            us = BF[s].get_roi_feature('label')
            us[us>-1] = maplabel[us[us>-1]]
            BF[s].set_roi_feature('label',us)
            header = BF[s].header

    if k>0:
        # create the object
        LR = landmark_regions(k,header=header,subj=subjs,coord=coords)  
    else:
        LR=None
    return LR,maplabel




def clean_density_redraw(BFLs,dmax,xyz,pval = 0.05,verbose=0,dev=0,nrec=5,nsamples=10):
    """
    Computation of the positions where there is a significant
    spatial accumulation of pointwise ROIs. The significance is taken with
    respect to a uniform distribution of the ROIs.
    INPUT:
    - BFLs : a list of ROIs hierarchies, putatively describing ROIs
    from different subjects
    CAVEAT 1: The nested ROI (NROI) must have a 'position' feature defined
    beforehand
    CAVEAT 2: the structure is edited and modified by this function
    - dmax : the kernel width (std) for the spatial density estimator
    - xyz : a set of coordinates on which the test is perfomed
    it should be written as (nbitems,dimension) array
    - pval=0.05: corrected p-value for the significance of the test
    Importantly, the p-value is corrected only for the number of ROIs
    per subject
    - verbose=0: verbosity mode
    - dev=0. If dev=1, a different technique is used for density estimation
    (WIP)
    - nrec=5: number of recursions in the test: When some regions fail to be
    significant at one step, the density is recomputed, and the test is
    performed again and so on
    19/02/07: new version without the monotonic exclusion heuristic 
    """
    nbsubj = np.size(BFLs)
    sqdmax = 2*dmax*dmax
    nvox = xyz.shape[0]
    nlm = np.zeros(nbsubj)
    for s in range(nbsubj):
        if BFLs[s]!=None:
             nlm[s] = BFLs[s].get_k()
        
    if verbose>0: print nlm
    Nlm = np.sum(nlm)
    nlm0 = 2*nlm
    q = 0
    BFLc = [None for s in range(nbsubj)]
    for s in range(nbsubj):
        if BFLs[s]!=None:
            BFLc[s] = BFLs[s].copy()
            
    while np.sum((nlm0-nlm)**2)>0:
        nlm0 = nlm.copy()
        
        if dev==0:
            weight = compute_density(BFLc,xyz,dmax)
        else:
            weight = compute_density_dev(BFLc,xyz,dmax)

        sweight = np.sum(weight,1)
        ssw = np.sort(sweight)
        thg = ssw[round((1-pval)*nvox)]

        # make surrogate data
        if dev==0:
            surweight = compute_surrogate_density(BFLc,xyz,dmax,nsamples)
        else:
            surweight = compute_surrogate_density_dev(BFLc,xyz,dmax,nsamples)
        
        srweight = np.sum(surweight,1)
        srw = np.sort(srweight)
        
        thf = srw[int((1-min(pval,1))*nvox*nsamples)]
        mnlm = max(1,float(Nlm)/nbsubj)
        imin = min(nvox*nsamples-1,int((1.-pval/mnlm)*nvox*nsamples))
        thcf = srw[imin]
        if verbose: print thg,thf,thcf
        
        if q<1:
            if verbose>1:
                fig_density(sweight,surweight,pval,nlm)
                    
        for s in range(nbsubj):
            if nlm[s]>0:
                w1 = srweight-surweight[:,s]
                sw1 = np.sort(w1)
                imin = min(int((1.-pval/nlm[s])*nvox*nsamples),nvox*nsamples-1)
                th1 = sw1[imin]
                w1 = sweight-weight[:,s]
                BFLc[s] = BFLs[s].copy()
                targets = w1[BFLc[s].roi_features['seed']]
                valid = (targets>th1)
                BFLc[s].clean(np.ravel(valid))#
                nlm[s] = BFLc[s].get_k()
        
        
        Nlm = sum(nlm);
        q = q+1
        if verbose>0: print nlm
        if q==1:
            b = pval/nlm.mean()
            a = np.sum(sweight[sweight>thf])/np.sum(sweight)
        if q>nrec:
            break
        if Nlm==0:
            break

    print q,nlm0,nlm

    for s in range(nbsubj):
        if BFLs[s]!=None:
            BFLs[s]=BFLc[s].copy()
    return a,b

def clean_density(BFLs,dmax,xyz,pval = 0.05,verbose=0,dev=0,nrec=5,nsamples=10):
    """
    Computation of the positions where there is a significant
    spatial accumulation of pointwise ROIs. The significance is taken with
    respect to a uniform distribution of the ROIs.
    INPUT:
    - BFLs : a list of ROIs hierarchies, putatively describing ROIs
    from different subjects
    CAVEAT 1: The nested ROI (NROI) must have a 'position' feature defined
    beforehand
    CAVEAT 2: the structure is edited and modified by this function
    - dmax : the kernel width (std) for the spatial density estimator
    - xyz : a set of coordinates on which the test is perfomed
    it should be written as (nbitems,dimension) array
    - pval=0.05: corrected p-value for the significance of the test
    Importantly, the p-value is corrected only for the number of ROIs
    per subject
    - verbose=0: verbosity mode
    - dev=0. If dev=1, a different technique is used for density estimation
    (WIP)
    - nrec=5: number of recursions in the test: When some regions fail to be
    significant at one step, the density is recomputed, and the test is
    performed again and so on
    """
    nbsubj = np.size(BFLs)
    sqdmax = 2*dmax*dmax
    nvox = xyz.shape[0]
    nlm = np.array([BFLs[s].get_k() for s in range(nbsubj)]).astype(np.int)
 
    if verbose>0: print nlm
    Nlm = np.sum(nlm)
    Nlm0 = 2*Nlm
    q = 0
    while Nlm0>Nlm:
        Nlm0 = Nlm
        if dev==0:
            weight = compute_density(BFLs,xyz,dmax)
        else:
            weight = compute_density_dev(BFLs,xyz,dmax)

        sweight = np.sum(weight,1)
        ssw = np.sort(sweight)
        thg = ssw[round((1-pval)*nvox)]

        # make surrogate data
        if dev==0:
            surweight = compute_surrogate_density(BFLs,xyz,dmax,nsamples)
        else:
            surweight = compute_surrogate_density_dev(BFLs,xyz,dmax,nsamples)
        
        srweight = np.sum(surweight,1)
        srw = np.sort(srweight)
        if verbose>0:
            thf = srw[int((1-min(pval,1))*nvox*nsamples)]
            mnlm = max(1,float(Nlm)/nbsubj)
            imin = min(nvox*nsamples-1,int((1.-pval/mnlm)*nvox*nsamples))
            # print pval,mnlm, pval/mnlm, 1.-pval/mnlm,np.size(srw),imin
            thcf = srw[imin]
            print thg,thf,thcf
        
        if q<1:
            if verbose>1:
                fig_density(sweight,surweight,pval,nlm)
                    
        for s in range(nbsubj):
            if nlm[s]>0:
                w1 = srweight-surweight[:,s]
                sw1 = np.sort(w1)
                imin = min(int((1.-pval/nlm[s])*nvox*nsamples),nvox*nsamples-1)
                th1 = sw1[imin]
                w1 = sweight-weight[:,s]
                targets = w1[BFLs[s].get_seed()]
                valid = (targets>th1)
                BFLs[s].clean(valid)
                nlm[s] = BFLs[s].get_k()
            
        Nlm = sum(nlm);
        q = q+1
        if verbose>0:
            print nlm
        if q==1:
            b = pval/nlm.mean()
            a = np.sum(sweight[sweight>thf])/np.sum(sweight)
        if q>nrec:
            break
        if Nlm==0:
            break
    return a,b

def fig_density(sweight,surweight,pval,nlm):
    """
    Plot the histogram of sweight across the image
    and the thresholds implied by the surrogate model (surweight)
    """
    import matplotlib.pylab as MP
    # compute some thresholds
    nlm = nlm.astype('d')
    srweight = np.sum(surweight,1)
    srw = np.sort(srweight)
    nitem = np.size(srweight)
    thf = srw[int((1-min(pval,1))*nitem)]
    mnlm = max(1,nlm.mean())
    imin = min(nitem-1,int((1.-pval/mnlm)*nitem))
    
    thcf = srw[imin]
    h,c = np.histogram(sweight,100)
    I = h.sum()*(c[1]-c[0])
    h = h/I
    h0,c0 = np.histogram(srweight,100)
    I0 = h0.sum()*(c0[1]-c0[0])
    h0 = h0/I0
    MP.figure(1)
    MP.plot(c,h)
    MP.plot(c0,h0)
    MP.legend(('true histogram','surrogate histogram'))
    MP.plot([thf,thf],[0,0.8*h0.max()])
    MP.text(thf,0.8*h0.max(),'p<0.2, uncorrected')
    MP.plot([thcf,thcf],[0,0.5*h0.max()])
    MP.text(thcf,0.5*h0.max(),'p<0.05, corrected')
    MP.savefig('/tmp/histo_density.eps')
    MP.show()


def compute_density(BFLs,xyz,dmax):
    """
    Computation of the density of the BFLs points in the xyz volume
    dmax is a scale parameter
    """
    nvox = xyz.shape[0]
    nbsubj = np.size(BFLs)
    sqdmax = 2*dmax*dmax
    weight = np.zeros((nvox,nbsubj),'d')
    nlm =np.zeros(nbsubj).astype('int')
    for s in range(nbsubj):
        if BFLs[s]!=None:
            nlm[s] = BFLs[s].k
 
    for s in range(nbsubj):
        if nlm[s]>0:
            coord = BFLs[s].get_roi_feature('position')
            for i in range(nlm[s]):
                dxyz = xyz - coord[i,:]
                dw = np.exp(-np.sum(dxyz**2,1)/sqdmax)
                weight[:,s] = weight[:,s] + dw
    return weight

                    
def compute_density_dev(BFLs,xyz,dmax):
    """
    Computation of the density of the BFLs points in the xyz volume
    dmax is a scale parameter
    """
    import nipy.neurospin.utils.smoothing.smoothing as smoothing
    nvox = xyz.shape[0]
    nbsubj = np.size(BFLs)
    weight = np.zeros((nvox,nbsubj),'d')
    nlm = np.array([BFLs[s].k for s in range(nbsubj)])
    for s in range(nbsubj):
        if nlm[s]>0:
            weight[BFLs[s].seed,s] = 1
    weight = smoothing.cartesian_smoothing(np.transpose(xyz),weight,dmax)
    weight = weight*(2*np.pi*dmax*dmax)**1.5
    return weight

def compute_surrogate_density(BFLs,xyz,dmax,nsamples=1):
    """
    Cross-validated estimation of random samples of the uniform distributions
    INPUT:
    - BFLs : a list of sets of ROIs the list length, nsubj, is taken
    as the number of subjects
    - xyz (gs,3) array: a sampling grid to estimate
    spatial distribution
    - dmax kernel width of the density estimator
    - nsamples=1: number of surrogate smaples returned
    OUTPUT:
    - surweight: a (gs*nsamples,nsubj) array of samples
    """
    nvox = xyz.shape[0]
    nbsubj = np.size(BFLs)
    sqdmax = 2*dmax*dmax
    nlm = np.array([BFLs[s].k for s in range(nbsubj)])
    surweight = np.zeros((nvox*nsamples,nbsubj),'d')
    for it in range(nsamples):
        for s in range(nbsubj):
            if nlm[s]>0:
                js = (nvox*nr.rand(nlm[s])).astype(np.int)
                for i in range(nlm[s]):         
                    dxyz = xyz-xyz[js[i],:]
                    dw = np.exp(-np.sum(dxyz*dxyz,1)/sqdmax)
                    surweight[nvox*it:nvox*(it+1),s] = surweight[nvox*it:nvox*(it+1),s] + dw
    return surweight

def compute_surrogate_density_dev(BFLs,xyz,dmax,nsamples=1):
    """
    caveat: does not work for nsamples>1
    This function computes a surrogate density using graph diffusion techniques
    """
    import nipy.neurospin.utils.smoothing.smoothing as smoothing
    nvox = xyz.shape[0]
    nbsubj = np.size(BFLs)
    nlm = np.array([BFLs[s].k for s in range(nbsubj)])
    aux = np.zeros((nvox,nsamples*nbsubj),'d')
    for s in range(nbsubj):
        if nlm[s]>0:
            for it in range(nsamples):
                js = (nvox*nr.rand(nlm[s])).astype(np.int)
                aux[js,s*nsamples+it] = 1

    aux = smoothing.cartesian_smoothing(np.transpose(xyz),aux,dmax)
    
    surweight = np.zeros((nvox*nsamples,nbsubj),'d')
    for s in range(nbsubj):
        surweight[:,s] = np.reshape(aux[:,s*nsamples:(s+1)*nsamples],(nvox*nsamples))

    surweight = surweight*(2*np.pi*dmax*dmax)**1.5
    return surweight    

def hierarchical_asso(BF,dmax):
    """
    Compting an association graph of the ROIs defined across different subjects
    INPUT:
    - BF a list of ROI hierarchies, one for each subject
    - dmax : spatial scale used xhen building associtations
    OUPUT:
    - G a graph that represent probabilistic associations between all
    cross-subject pairs of regions. Note that the probabilities are normalized
    on a within-subject basis.
    """
    nbsubj = np.size(BF)
    nlm = np.array([BF[i].k for i in range(nbsubj)])
    cnlm = np.hstack(([0],np.cumsum(nlm)))
    if cnlm[nbsubj]==0:
        Gcorr = []
        return Gcorr

    eA = []
    eB = []
    eD = []
    for s in range(nbsubj):
        if (BF[s].k>0):
            for t in range(s):
                if (BF[t].k>0):
                    cs =  BF[s].get_roi_feature('position')
                    ct = BF[t].get_roi_feature('position')
                    Gs = BF[s].make_graph()
                    Gs.symmeterize()
                    Gs = Gs.adjacency()
                    Gt = BF[t].make_graph()
                    Gt.symmeterize()
                    Gt = Gt.adjacency()
                    ea,eb,ed = BPmatch.BPmatch(cs,ct, Gs,dmax)
                    if np.size(ea)>0:
                        eA = np.hstack((eA,ea+cnlm[s]))
                        eB = np.hstack((eB,eb+cnlm[t]))
                        eD = np.hstack((eD,ed))

                    ea,eb,ed = BPmatch.BPmatch(ct,cs, Gt,dmax)
                    if np.size(ea)>0:
                        eA = np.hstack((eA,ea+cnlm[t]))
                        eB = np.hstack((eB,eb+cnlm[s]))
                        eD = np.hstack((eD,ed))
        
    if np.size(eA)>0:
        edges = np.transpose([eA,eB]).astype(np.int)
        Gcorr = fg.WeightedGraph(cnlm[nbsubj],edges,eD)
    else:
        Gcorr = []
    return Gcorr



def RD_cliques(Gc,bstochastic=1):
    """
    Replicator dynamics graph segmentation: python implementation
    INPUT :
    - Gc graph to be segmented
    - bstochastic=1 stochastic initialization of the graph
    OUPUT:
    - labels : array of size V, the number of vertices of Gc
    the labelling of the vertices that represent the segmentation
    """
    V = Gc.V
    labels = -np.ones((V))
    i=0
    eps = 1.e-6
    nu = []
    Fc = ff.Field(Gc.V,Gc.edges,Gc.weights)
    while (np.sum(labels>-1)<V):
        if bstochastic:
            u = nr.rand(V)
        else:
            u = np.ones(V)

        u[labels>-1]=0
        w = np.zeros(V)
        q = 0
        while (np.sum((u-w)*(u-w))>eps):
            w = u.copy()
            Fc.set_field(u)
            Fc.diffusion()
            v = np.reshape(Fc.field,(V))
            u = u*v
            su = np.sum(u)
            if (su==0):
                u = np.zeros(V)
                break
            else:
                u = u/su
            q = q+1
            
            if q>1000:
                break

        Vl = V-sum(labels>-1)
        if Vl==1:
            Vl = 0.5
        
        j = np.nonzero(u>1./Vl)
        j = np.reshape(j,np.size(j))
        labels[j] = i
        nu.append(-np.size(j))
        i = i+1
        if np.size(j)==0:
            break
        if np.sum(u*u)==0:
            break
        
    nu = np.array(nu)
    inu = nu.argsort()
    j = np.nonzero(labels>-1)
    j = np.reshape(j,np.size(j))
    labels[j] = inu[labels[j]]
    j = np.nonzero(labels==-1);
    j = np.reshape(j,np.size(j))
    ml = labels.max()+1
    
    if np.size(j)>0:
        labels[j] =  np.arange(ml,ml+np.size(j))

    return labels

def merge(Gc,labels):
    """
    Given a first labelling of the graph Gc, this function
    builds a reduced graph by merging the vertices according to the labelling
    INPUT:
    - Gc the input graph
    - labels : array of size V, the number of vertices of Gc
    the labelling of the vertices that represent the segmentation
    OUTPUT:
    - labels : the new labelling after further merging
    - Gr the reduced graph after merging
    """
    V = Gc.V
    q = labels.max()+1
    P = np.zeros((V,q))
    for v in range(V):
        P[v,labels[v]]=1

    Fc = ff.Field(Gc.V,Gc.edges,Gc.weights)
    Fc.set_field(P)
    Fc.diffusion()
    Q = Fc.field

    b = np.dot(np.transpose(Q),P) 
    b = np.transpose(np.transpose(b)/sum(b,1))
    b = b-np.diag(np.diag(b))
    ib,jb = np.where(b)
    
    if np.size(ib)>0:
        kb = b[ib,jb]
        Gr = fg.WeightedGraph(q,np.transpose([ib,jb]),kb) 
        # w = RD_cliques(Gr)
        w = Gr.cliques()
        labels = w[labels]
    else:
        Gr = []
        
    return labels, Gr
    

def segment_graph_rd(Gc,nit = 1,verbose=0):
    """
    This function prefoms a hard segmentation of the graph Gc
    using a replicator dynamics approach.
    The clusters obtained in the first pass are further merged
    during a second pass, based on a reduced graph
    INPUT:
    - Gc : the graph to be segmented
    OUTPUT:
    - u : array of size V, the number of vertices of Gc
    the labelling of the vertices that represent the segmentation
    """
    u = []
    if (Gc.E>0):
        # print Gc.E
        u = Gc.cliques()
        if verbose>0:
            print u.max()
        Gr = Gc
        for i in range(nit):
            u,Gr = merge(Gr,u)
            if verbose>0:
                print u.max()
    return u


def Compute_Amers (Fbeta, Beta, xyz ,header, tal,dmax = 10., thr=3.0, ths = 0,pval=0.2,verbose=0):
    """
     This is the main function for contrsucting the BFLs
     INPUT
     - Fbeta : field structure that contains the spatial nodes of the dataset
     - Beta: functional data matrix of size (nbnodes,nbsubj)
     - xyz: 
     - tal: spatial coordinates of the nodes (e.g. MNI coords)
     - dmax=10.: spatial relaxation allowed in the preocedure
     - thr = 3.0: thrshold at the first-level
     - ths = 0, number of subjects to validate a BFL
     - pval = 0.2 : significance p-value for the spatial inference
     OUPUT
     - crmap: the map of CR of the activated clusters in the common space
     - AF : group level BFLs
     - BFLs: list of first-level nested ROIs
     - Newlabel: labelling of the individual ROIs
    """
    BFLs = []
    LW = [] 
    nbsubj = Beta.shape[1]
    nvox = Beta.shape[0]
    for s in range(nbsubj):
        beta = np.reshape(Beta[:,s],(nvox,1))
        Fbeta.set_field(beta)
        bfls = hroi.NROI_from_watershed(Fbeta,header,xyz,refdim=0,th=thr)
 
        if bfls!=None:
            bfls.compute_discrete_position()
            bfls.discrete_to_roi_features('position','average')
            
        #bfls.make_feature(tal,'position','mean')
        BFLs.append(bfls)

    # clean_density(BFLs,dmax,tal,pval,verbose=1,dev=0,nrec=5)
    clean_density_redraw(BFLs,dmax,tal,pval,verbose=0,dev=0,nrec=1,nsamples=10)
    
    Gc = hierarchical_asso(BFLs,dmax)
    Gc.weights = np.log(Gc.weights)-np.log(Gc.weights.min())
    if verbose:
        print Gc.V,Gc.E,Gc.weights.min(),Gc.weights.max()
    
    # building cliques
    #u = segment_graph_rd(Gc,1)
    u,cost = Average_Link_Graph_segment(Gc,0.1,Gc.V*1.0/nbsubj)

    # relabel the BFLs
    q = 0
    for s in range(nbsubj):
        BFLs[s].set_roi_feature('label',u[q:q+BFLs[s].k])
        q += BFLs[s].k
    
    LR,mlabel = build_LR(BFLs,ths)
    if LR!=None:
        crmap = LR.map_label(tal,pval = 0.95,dmax=2*dmax)
        
    return crmap, LR, BFLs 

