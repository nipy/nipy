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


# import neuroimaging.neurospin.clustering.clustering as fc
import neuroimaging.neurospin.graph.BPmatch as BPmatch
import neuroimaging.neurospin.graph.field as ff
import neuroimaging.neurospin.graph.graph as fg
import neuroimaging.neurospin.graph.hroi as hroi
from neuroimaging.neurospin.clustering.hierarchical_clustering import \
     Average_Link_Graph_segment


class Amers:
    """
    This class is intended to represent inter-subject regions
    its members are:
    - k: (int) the number of regions involved
    - subj: array of int of size k, an identifier from where ('the subjects')
    the regions come from
    - idx : array of int of size k, an index related to each region,
    typically a seed voxel of the region in a certain representation
    - coord: array of double of size k*p, where p is a certain space dimension
    representing coordinates of the regions in a certain
    unspecified, but common system
    """

    def __init__(self, k, subj=None, idx=None, coord=None):
        self.k = int(k)

        OK = 0
        if k>0:
            OK = (np.size(subj)==k)
            OK = OK &(np.size(idx)==k)
            OK = OK &(coord.shape[0]==k)

        if np.size(subj)==k:
            self.subj =  subj.astype('i')
                
        if np.size(idx)==k:
            self.idx =  idx.astype('i')
                        
        if coord.shape[0]==k:
            self.coord = coord
                
        if OK==0:
            raise ValueError, "incorrect objects have been provided"
            self.idx = None
            self.subj = None
            self.coord = None

    def homogeneity(self):
        """
        d = self.homogeneity()
        OUTPUT:
        d (float)
        returns the mean distance between the coordinates of the regions   
        """
        if self.k>0:
            if self.k==1: return 0
            tcoord = np.transpose(self.coord)
            
            D = BPmatch.EDistance(tcoord,tcoord)
            d = D.sum()/(self.k*(self.k-1))
            return d

    def center(self):
        """
        c = self.center()
        returns the average of the coordinates
        """
        if self.k>0:
           return (np.mean(self.coord,0))

    def confidence_region(self,cs,dmax = 10,pval = 0.95):
        """
        i = self.confidence_region(cs,dmax = 10,pval = 0.95)
        Sample the pval-confidence region of  AF
        on the proposed coordiante set cs, assuming a Gaussian shape
        INPUT:
        - cs: an array of size(n*p) a set of input coordinates
        - dmax=10 : an upper bound for the spatial variance
        to avoid degenerate variance
        - pval=0.95 cutoff for the CR
        OUPUT:
        i = set of entries of the coordinates that are within the cr
        """
        if cs.shape[1]!=self.coord.shape[1]:
            raise ValueError, "incompatible dimensions"
        if self.k>1:
            center = np.mean(self.coord,0)
            dx = self.coord-center
            covariance = np.dot(np.transpose(dx),dx)/(self.k-1)
            import numpy.linalg as L
            U,S,V = L.svd(covariance,0)
            sqrtS = np.sqrt(1/np.maximum(S,dmax))
            dx = cs-center
            dx = np.dot(dx,U)
            dx = np.dot(dx,np.diag(sqrtS))
            delta = np.sum(dx**2,1)
            import scipy.special as SP
            gamma = 2*SP.erfinv(pval)**2
    
            i = np.nonzero(delta<gamma)
            i = np.reshape(i,np.size(i))
            score = delta[i]
            return i,score

def clean_density_redraw(BFLs,dmax,xyz,pval = 0.05,verbose=0,dev=0,nrec=5,nsamples=10):
    """
    Computation of the positions where there is a significant
    spatial accumulation of pointwise ROIs. The significance is taken with
    respect to a uniform distribution of the ROIs.
    INPUT:
    - BFLs : a list of ROIs hierarchies, putatively describing ROIs
    from different subjects
    CAVEAT 1: The ROI_Hierarchy must have a 'position' feature defined
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
    Sess = np.size(BFLs)
    sqdmax = 2*dmax*dmax
    nvox = xyz.shape[0]
    nlm = np.array([BFLs[s].get_k() for s in range(Sess)]).astype('i')
 
    if verbose>0: print nlm
    Nlm = np.sum(nlm)
    nlm0 = 2*nlm
    q = 0
    BFLc = [BFLs[s].copy() for s in range(Sess)]
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
        if verbose>0:
            thf = srw[int((1-min(pval,1))*nvox*nsamples)]
            mnlm = max(1,float(Nlm)/Sess)
            imin = min(nvox*nsamples-1,int((1.-pval/mnlm)*nvox*nsamples))
            # print pval,mnlm, pval/mnlm, 1.-pval/mnlm,np.size(srw),imin
            thcf = srw[imin]
            print thg,thf,thcf
        
        if q<1:
            if verbose>1:
                fig_density(sweight,surweight,pval,nlm)
                    
        for s in range(Sess):
            if nlm[s]>0:
                w1 = srweight-surweight[:,s]
                sw1 = np.sort(w1)
                imin = min(int((1.-pval/nlm[s])*nvox*nsamples),nvox*nsamples-1)
                th1 = sw1[imin]
                w1 = sweight-weight[:,s]
                BFLc[s] = BFLs[s].copy()
                targets = w1[BFLc[s].get_seed()]
                valid = (targets>th1)
                BFLc[s].clean(valid)
                nlm[s] = BFLc[s].get_k()
        
        
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

    print q,nlm0,nlm

    for s in range(Sess): BFLs[s]=BFLc[s].copy()
    return a,b

def clean_density(BFLs,dmax,xyz,pval = 0.05,verbose=0,dev=0,nrec=5,nsamples=10):
    """
    Computation of the positions where there is a significant
    spatial accumulation of pointwise ROIs. The significance is taken with
    respect to a uniform distribution of the ROIs.
    INPUT:
    - BFLs : a list of ROIs hierarchies, putatively describing ROIs
    from different subjects
    CAVEAT 1: The ROI_Hierarchy must have a 'position' feature defined
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
    Sess = np.size(BFLs)
    sqdmax = 2*dmax*dmax
    nvox = xyz.shape[0]
    nlm = np.array([BFLs[s].get_k() for s in range(Sess)]).astype('i')
 
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
            mnlm = max(1,float(Nlm)/Sess)
            imin = min(nvox*nsamples-1,int((1.-pval/mnlm)*nvox*nsamples))
            # print pval,mnlm, pval/mnlm, 1.-pval/mnlm,np.size(srw),imin
            thcf = srw[imin]
            print thg,thf,thcf
        
        if q<1:
            if verbose>1:
                fig_density(sweight,surweight,pval,nlm)
                    
        for s in range(Sess):
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
    Sess = np.size(BFLs)
    sqdmax = 2*dmax*dmax
    weight = np.zeros((nvox,Sess),'d')
    nlm = np.array([BFLs[s].k for s in range(Sess)])
    for s in range(Sess):
        if nlm[s]>0:
            coord = BFLs[s].get_ROI_feature('coord')
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
    import neuroimaging.neurospin.utils.smoothing.smoothing as smoothing
    nvox = xyz.shape[0]
    Sess = np.size(BFLs)
    weight = np.zeros((nvox,Sess),'d')
    nlm = np.array([BFLs[s].k for s in range(Sess)])
    for s in range(Sess):
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
    Sess = np.size(BFLs)
    sqdmax = 2*dmax*dmax
    nlm = np.array([BFLs[s].k for s in range(Sess)])
    surweight = np.zeros((nvox*nsamples,Sess),'d')
    for it in range(nsamples):
        for s in range(Sess):
            if nlm[s]>0:
                js = (nvox*nr.rand(nlm[s])).astype('i')
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
    import neuroimaging.neurospin.utils.smoothing.smoothing as smoothing
    nvox = xyz.shape[0]
    Sess = np.size(BFLs)
    nlm = np.array([BFLs[s].k for s in range(Sess)])
    aux = np.zeros((nvox,nsamples*Sess),'d')
    for s in range(Sess):
        if nlm[s]>0:
            for it in range(nsamples):
                js = (nvox*nr.rand(nlm[s])).astype('i')
                aux[js,s*nsamples+it] = 1

    aux = smoothing.cartesian_smoothing(np.transpose(xyz),aux,dmax)
    
    surweight = np.zeros((nvox*nsamples,Sess),'d')
    for s in range(Sess):
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
    Sess = np.size(BF)
    nlm = np.array([BF[i].k for i in range(Sess)])
    cnlm = np.hstack(([0],np.cumsum(nlm)))
    if cnlm[Sess]==0:
        Gcorr = []
        return Gcorr

    eA = []
    eB = []
    eD = []
    for s in range(Sess):
        if (BF[s].k>0):
            for t in range(s):
                if (BF[t].k>0):
                    cs =  BF[s].get_ROI_feature('coord')
                    ct = BF[t].get_ROI_feature('coord')
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
        edges = np.transpose([eA,eB]).astype('i')
        Gcorr = fg.WeightedGraph(cnlm[Sess],edges,eD)
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

def Build_Amers(BF,u,ths=0):
    """
    Given a list of hierarchical ROIs, and an associated labelling, this
    creates an Amer structure wuch groups ROIs with the same label.
    INPUT:
    - BF is the list of hierarchical ROIs.
    it is assumd that each list corresponds to one subject
    - u is a labelling array of size the total number of ROIs
    - ths=0 defines the condition (c):
    A label should be present in ths subjects in order to be valid
    OUTPUT:
    - AF : a list of Amers, each of which describing a cross-subject set of ROIs
    - newlabel :  a relabelling of the individual ROIs, similar to u, which discards
    labels that do not fulfill the condition (c)
    """
    Sess = np.size(BF)
    Nlm = np.size(u)

    subj = np.concatenate([s*np.ones(BF[s].k,'i') for s in range(Sess)])
    nrois = np.size(subj)
    if nrois != Nlm:
        raise ValueError, "incompatiable estimates of the number of regions"
    intrasubj = np.concatenate([np.arange(BF[s].k) for s in range(Sess)])
    newlabel = -np.ones(np.size(u),'i')
    AF = []
    nl = 0
    if np.size(u)>0:
        Mu = u.max()+1
        for i in range(Mu):
            j = np.nonzero(u==i)
            j = np.reshape(j,np.size(j))
            #j.sort()
            #q = 1
            q = 0
            if np.size(j)>1:
                #nj = np.size(j)-1
                #q = 1+ np.sum(subj[j[np.arange(nj)+1]]-subj[j[np.arange(nj)]]>0)
                q = np.size(np.unique(subj[j]))
            
            if  (q>ths):
                newlabel[j] = nl
                sj = np.size(j)
                idx = np.zeros(sj)
                coord = np.zeros((sj,3),'d')
                for a in range(sj):
                    sja = subj[j[a]]
                    isja = intrasubj[j[a]]
                    idx[a] = BF[sja].seed[isja]
                    coord[a,:] = BF[sja].get_ROI_feature('coord')[isja]

                amers = Amers(sj, subj[j], idx,coord)
                AF.append(amers)
                nl = nl+1
                
    return AF,newlabel

def Compute_Amers (Fbeta,Beta, tal,dmax = 10., thr=3.0, ths = 0,pval=0.2):
    """
     This is the main function for contrsucting the BFLs
     INPUT
     - Fbeta : field structure that contains the spatial nodes of the dataset
     - Beta: functional data matrix of size (nbnodes,nbsubj)
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
    Sess = Beta.shape[1]
    nvox = Beta.shape[0]
    for s in range(Sess):
        beta = np.reshape(Beta[:,s],(nvox,1))
        Fbeta.set_field(beta)
        
        if thr<beta.max():
            idx,depth, major,label = Fbeta.custom_watershed(0,thr)
        else:
            idx = None
            depth = None
            major = None
            label = -np.ones(nvox)
    
        if idx==None:
            k = 0
        else:
            k = np.size(idx)

        bfls = hroi.ROI_Hierarchy(k,idx,major,label)
        bfls.make_feature(tal,'coord','mean')
        BFLs.append(bfls)

    # clean_density(BFLs,dmax,tal,pval,verbose=1,dev=0,nrec=5)
    clean_density_redraw(BFLs,dmax,tal,pval,verbose=1,dev=0,nrec=1,nsamples=10)
    
    Gc = hierarchical_asso(BFLs,dmax)
    Gc.weights = np.log(Gc.weights)-np.log(Gc.weights.min())
    print Gc.V,Gc.E,Gc.weights.min(),Gc.weights.max()
    
    # building cliques
    #u = segment_graph_rd(Gc,1)
    u,cost = Average_Link_Graph_segment(Gc,0.1,Gc.V*1.0/Sess)

    AF,newlabel = Build_Amers(BFLs,u,ths)

    crmap = np.zeros(nvox)
    gscore =  np.inf*np.ones(nvox)  
    for i in range(np.size(AF)):
        print i, AF[i].k, AF[i].homogeneity(), AF[i].center()
        j,score = AF[i].confidence_region(tal)
        lscore = np.inf*np.ones(nvox)
        lscore[j] = score 
        crmap[gscore>lscore]=i+1
        gscore = np.minimum(gscore,lscore)
        #crmap[j] = i+1
        
    return crmap, AF, BFLs,newlabel

