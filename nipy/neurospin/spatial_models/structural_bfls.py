# vi: set ft=python sts=4 ts=4 sw=4 et:
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
""" 
The main routine of this module aims at performing the
extraction of ROIs from multisubject dataset using the localization.

This has been published in Thirion et al. Structural Analysis of fMRI
Data Revisited: Improving the Sensitivity and Reliability of fMRI
Group Studies.  IEEE TMI 2007

Author : Bertrand Thirion, 2006-2010
"""

#autoindent 

import numpy as np
import numpy.random as nr
from scipy import stats

import nipy.neurospin.graph.BPmatch as BPmatch
import nipy.neurospin.graph.field as ff
import nipy.neurospin.graph.graph as fg
from nipy.neurospin.spatial_models import hroi 
from nipy.neurospin.clustering.hierarchical_clustering import \
    average_link_graph_segment

class LandmarkRegions(object):
    """
    This class is intended to represent a set of inter-subject regions
    It should inherit from some abstract multiple ROI class,
    not implemented yet.
    """
    def __init__(self, domain, k, indiv_coord, subj, id=''):
        """
        Building the landmark_region

        Parameters
        ----------
        domain: ROI instance
                defines the spatial context of the SubDomains
        k: int, the number of regions considered
        subj: k-length list of subjects
              (these correspond to ROI feature)
        indiv_coord:  k-length list of arrays, optional,
                      coordinates of the nodes in some embedding space. 
        id: string, optional, identifier
        """
        self.domain = domain
        self.k = int(k)
        self.id = id
        self.features = {}
        self.set_feature('position', indiv_coord)
        self.set_feature('subjects', subj)
        
    def set_feature(self, fid, data):
        """
        """
        if len(data) != self.k:
            raise ValueError, 'data should have length k'
        self.features.update({fid:data})

    def get_feature(self, fid):
        return self.features[fid]
        
    def centers(self):
        """returns the average of the coordinates for each region
        """
        pos = self.get_feature('position')
        centers = np.array([np.mean(pos[k], 0) for k in range(self.k)])
        return centers

    def homogeneity(self):
        """ returns the mean distance between points within each LR
        """
        from nipy.neurospin.eda.dimension_reduction import Euclidian_distance
        
        coord = self.get_features('position')
        h = np.zeros(self.k)
        for k in range(self.k):
            pk = coord[k]
            sk = pk.shape[0]
            edk = Euclidian_distance(pk) 
            h[k] = edk.sum() / (sk * (sk-1))
        return h

    def density (self, k, coord=None, dmax=1., dof=10):
        """
        Posterior density of component k

        Parameters
        ----------
        k: int, less or equal to self.k
           reference component
        coord: array of shape(n, self.dom.em_dim), optional
            a set of input coordinates
        dmax: float, optional
              regularizaing constant for the variance estimation
        dof: float, optional,
             strength of the regulaization

        Returns
        -------
        pd: array of shape(n)
            the posterior density that has been computed
        delta: array of shape(n)
               the quadratic term in the gaussian model

        Fixme
        -----
        instead of dof/dmax, use Raftery's regularization
        """
        from scipy.linalg import svd
        
        if k>self.k:
            raise ValueError, 'wrong region index'
        
        pos = self.get_feature('position')[k]
        center = pos.mean(0)
        dim = self.domain.em_dim
        
        if coord==None:
            coord = self.domain.coord
            
        if coord.shape[1]!= dim: 
            raise ValueError, "incompatible dimensions"

        n_points = pos.shape[0]
        dx = pos-center
        covariance = np.dot(dx.T, dx) / n_points
        U, S, V = svd(covariance,0)
        S = (n_points*S + dmax**2 * np.ones(dim)*dof)/(n_points + dof)
        sqrts = 1. / np.sqrt(S)
        dx = coord - center
        dx = np.dot(dx, U)
        dx = np.dot(dx, np.diag(sqrts))
        delta = np.sum(dx**2,1)
        lcst = - np.log(2*np.pi)*dim/2+(np.log(sqrts)).sum()
        pd = np.exp(lcst-delta/2)
        return pd, delta
        
    def hpd(self, k, coord=None, pval=0.95, dmax=1.0):
        """
        Sample the posterior probability of being in k
        on a grid defined by cs, assuming that the roi is an ellipsoid
        
        Parameters
        ----------
        k: int, less or equal to self.k
           reference component
        coord: array of shape(n,dim), optional
               a set of input coordinates
        pval: float<1, optional,
              cutoff for the CR
        dmax=1.0: an upper bound for the spatial variance
                  to avoid degenerate variance
        
        Returns
        -------
        hpd array of shape(n) that yields the value
        """
        hpd, delta = self.density (k, coord, dmax)
        
        import scipy.special as sp
        gamma = 2*sp.erfinv(pval)**2
        #
        #--- all the following is to solve the equation
        #--- erf(x/sqrt(2))-x*exp(-x**2/2)/sqrt(pi/2) = alpha
        #--- should better be put elsewhere
        #
        def dicho_solve_lfunc(alpha,eps = 1.e-7):
            if alpha>1:
                raise ValueError, "no solution for alpha>1"
            if alpha>1-1.e-15:
                return np.infty
            if alpha<0:
                raise ValueError, "no solution for alpha<0" 
            if alpha<1.e-15:
                return 0
    
            xmin = sp.erfinv(alpha)*np.sqrt(2)
            xmax = 2*xmin
            while lfunc(xmax)<alpha:
                xmax*=2
                xmin*=2
            return (dichomain_lfunc(xmin,xmax,eps,alpha))

        def dichomain_lfunc(xmin, xmax, eps, alpha):
            x =  (xmin+xmax)/2
            if xmax<xmin+eps:
                return x
            else:
                if lfunc(x)>alpha:
                    return dichomain_lfunc(xmin, x, eps, alpha)
                else:
                    return dichomain_lfunc(x, xmax, eps, alpha)
        
        def lfunc(x):
            return sp.erf(x/np.sqrt(2))-x*np.exp(-x**2/2)/np.sqrt(np.pi/2)

        gamma = dicho_solve_lfunc(pval)**2
        hpd[delta>gamma]=0
        return hpd

    def map_label(self, coord=None, pval=0.95, dmax=1.):
        """
        Sample the set of landmark regions
        on the proposed coordiante set cs, assuming a Gaussian shape
        
        Parameters
        ----------
        coord: array of shape(n,dim), optional,
               a set of input coordinates
        pval: float in [0,1]), optional
              cutoff for the CR, i.e.  highest posterior density threshold
        dmax: an upper bound for the spatial variance
                to avoid degenerate variance
    
        Returns
        -------
        label: array of shape (n): the posterior labelling
        """
        if coord==None:
            coord = self.domain.coord
        label = -np.ones(coord.shape[0])
        if self.k>0:
            aux = -np.ones((coord.shape[0], self.k))
            for k in range(self.k):
                aux[:,k] = self.hpd(k, coord, pval, dmax)

            maux = np.max(aux,1)
            label[maux>0] = np.argmax(aux,1)[maux>0]
        return label

    def show(self):
        """function to print basic information on self
        """
        centers = self.centers()
        subj = self.get_feature('subjects')
        prevalence = self.roi_prevalence()
        print "index", "prevalence", "mean_position", "individuals"
        for i in range(self.k):
            print i, prevalence[i], centers[i], np.unique(subj[i])


    def roi_confidence(self, ths=0, fid='confidence'):
        """
        assuming that a certain feature fid field has been set 
        as a discrete feature,
        this creates an approximate p-value that states 
        how confident one might 
        that the LR is defined in at least ths individuals
        if conficence is not defined as a discrete_feature,
        it is assumed to be 1.
        
        Parameters
        ----------
        ths: integer that yields the representativity threshold 
        
        Returns
        -------
        pvals: array of shape self.k
               the p-values corresponding to the ROIs
        """
        pvals = np.zeros(self.k)
        subj = self.get_feature('subjects')
        if self.features.has_key(fid)==False:
            # the feature has not been defined
            print 'using per ROI subject counts'
            for j in range(self.k):
                pvals[j] = np.size(np.unique(subj[j]))
            pvals = pvals>ths + 0.5*(pvals==ths)
        else:
            for j in range(self.k):
                subjj = subj[j]
                conf = self.get_features(fid)[j]
                mp = 0.
                vp = 0.
                for ls in np.unique(subjj):
                    lmj = 1-np.prod(1-conf[subjj==ls])
                    lvj = lmj*(1-lmj)
                    mp = mp+lmj
                    vp = vp+lvj
                    # If noise is too low the variance is 0: ill-defined:
                    vp = max(vp, 1e-14)
                    
                pvals[j] = stats.norm.sf(ths, mp, np.sqrt(vp))
        return pvals

    def roi_prevalence(self, fid='confidence'):
        """
        assuming that fid='confidence' field has been set 
        as a discrete feature,
        this creates the expectancy of the confidence measure
        i.e. expected numberof  detection of the roi in the observed group
             
        Returns
        -------
        confid: array of shape self.k
               the population_prevalence
        """
        confid = np.zeros(self.k)
        subj = self.get_feature('subjects')
        if self.features.has_key(fid)==False:
            for j in range(self.k):
                subjj = subj[j]
                confid[j] = np.size(np.unique(subjj))
        else:
            for j in range(self.k):
                subjj = subj[j]
                conf = self.get_feature(fid)[j]
                for ls in np.unique(subjj):
                    lmj = 1-np.prod(1-conf[subjj==ls])
                    confid[j] += lmj
        return confid
               
    def weighted_feature_density(self, feature):
        """
        Given a set of feature values, produce a weighted feature map,
        where roi-levle features are mapped smoothly based on the density
        of the components 
        
        Parameters
        ----------
        feature: array of shape (self.k),
                 the information to map

        Returns
        -------
        wsm: array of shape(self.shape)
        """
        if np.size(feature)!=self.k:
            raise ValueError, 'Incompatible feature dimension'

        cs = self.domain.coord
        aux = np.zeros((cs.shape[0], self.k))
        for k in range(self.k):
            aux[:, k], _ = self.density(k, cs)
            
        wsum = np.dot(aux, feature)
        return wsum
    
    def prevalence_density(self):
        """
        returns a weighted map of self.prevalence
             
        Returns
        -------
        wp: array of shape(n_samples)
        """
        return self.weighted_feature_density(self.roi_prevalence())


def build_LR(bf, thq=0.95, ths=0, dmax=1., verbose=0):
    """
    Given a list of hierarchical ROIs, and an associated labelling, this
    creates an Amer structure wuch groups ROIs with the same label.
    
    Parameters
    ----------
    bf : list of nipy.neurospin.spatial_models.hroi.Nroi instances
       it is assumd that each list corresponds to one subject
       each NROI is assumed to have the roi_features
       'position', 'label' and 'posterior_proba' defined
    thq=0.95, ths=0 defines the condition (c):
                   (c) A label should be present in ths subjects
                   with a probability>thq
                   in order to be valid
    dmax: float optional,
          regularizing constant that defines a prior on the region extent
    
    Results
    -------
    LR : an structural_bfls.LR instance, describing a cross-subject set of ROIs
       if inference yields a null results, LR is set to None
    newlabel: a relabelling of the individual ROIs, similar to u,
              which discards
              labels that do not fulfill the condition (c)

    Fixme
    -----
    Should be merged with sbf.build_LR
    """
    dim = bf[0].domain.em_dim
    
    # prepare various variables to ease information manipulation
    nbsubj = np.size(bf)
    subj = np.concatenate([s*np.ones(bf[s].k, np.int)
                           for s in range(nbsubj)])
    u = np.concatenate([bf[s].get_roi_feature('label')
                        for s in range(nbsubj)if bf[s].k>0])
    u = np.squeeze(u)
    if bf[0].roi_features.has_key('prior_proba'):
        conf =  np.concatenate([bf[s].get_roi_feature('prior_proba')
                                for s in range(nbsubj)if bf[s].k>0])
    else:
        conf = np.ones(u.size)
    intrasubj = np.concatenate([np.arange(bf[s].k)
                                for s in range(nbsubj) ])
    
    coords = []
    subjs = []
    pps = []
    n_labels = int(u.max()+1)
    valid = np.zeros(n_labels).astype(np.int)
    
    # do some computation to find which regions are worth reporting
    for i in np.unique(u[u>-1]):
        mp = 0.
        vp = 0.
        subjj = subj[u==i]
        for ls in np.unique(subjj):
            lmj = 1-np.prod(1-conf[(u==i)*(subj==ls)])
            lvj = lmj*(1-lmj)
            mp = mp + lmj
            vp = vp + lvj
            
        # If noise is too low the variance is 0: ill-defined:
        vp = max(vp, 1e-14)

        # if above threshold, get some information to create the LR
        if verbose:
            print i, valid.sum(), ths, mp, thq

        if stats.norm.sf(ths, mp, np.sqrt(vp))>thq:         
            sj = np.size(subjj)
            coord = np.zeros((sj, dim))
            for (k, s, a) in zip(intrasubj[u==i], subj[u==i], range(sj)):
                coord[a, :] = bf[s].get_roi_feature('position')[k]

            valid[i] = 1
            coords.append(coord)
            subjs.append(subjj)   
            pps.append(conf[u==i])

    # relabel the ROIs
    maplabel = -np.ones(n_labels).astype(np.int)
    maplabel[valid>0] = np.cumsum(valid[valid>0])-1
    for s in range(nbsubj):
        if bf[s].k>0:
            us = bf[s].get_roi_feature('label')
            us[us>-1] = maplabel[us[us>-1]]
            bf[s].set_roi_feature('label',us)

    # create the landmark regions structure
    k = np.sum(valid)
    LR = LandmarkRegions(bf[0].domain, k, indiv_coord=coords, subj=subjs)
    LR.set_feature('confidence', pps)
    
    return LR, maplabel

#########################################################################
# Structural BFLs
#########################################################################


def _clean_density_redraw(BFLs, dmax, xyz, pval=0.05, verbose=0, 
                          nrec=5, nsamples=10):
    """
    Computation of the positions where there is a significant
    spatial accumulation of pointwise ROIs. The significance is taken with
    respect to a uniform distribution of the ROIs.
    
    Parameters
    ----------
    BFLs: list of  nipy.neurospin.spatial_models.hroi.HierrachicalROI
          instances describing ROIs from different subjects
    dmax: float, the kernel width (std) for the spatial density estimator
    xyz (nbitems,dimension) array
        set of coordinates on which the test is perfomed
    pval: float, in [0,1], optional,
          corrected p-value for the significance of the test
    verbose=0: verbosity mode
    nrec=5: number of recursions in the test: When some regions fail to be
            significant at one step, the density is recomputed, 
            and the test is performed again and so on
                
    Note
    ----
    The Hierarchical ROI instances in BFLs must have a 'position' feature
        defined beforehand
    BFLs is edited and modified by this function
    The p-value is corrected only for the number of ROIs
        per subject
    """
    nbsubj = np.size(BFLs)
    nvox = xyz.shape[0]
    nlm = np.array([BFLs[s].k for s in range(nbsubj)])

    if verbose>0: print nlm
    Nlm = np.sum(nlm)
    nlm0 = 2*nlm
    q = 0
    BFLc = [BFLs[s].copy() for s in range(nbsubj)]

    while np.sum((nlm0-nlm)**2)>0:
        nlm0 = nlm.copy()        
        weight = _compute_density(BFLc, xyz, dmax)
        sweight = np.sum(weight,1)
        ssw = np.sort(sweight)
        thg = ssw[round((1-pval)*nvox)]

        # make surrogate data
        surweight = _compute_surrogate_density(BFLc, xyz, dmax, nsamples)
        srweight = np.sum(surweight,1)
        srw = np.sort(srweight)
        
        thf = srw[int((1-min(pval,1))*nvox*nsamples)]
        mnlm = max(1, float(Nlm)/nbsubj)
        imin = min(nvox*nsamples-1, int((1.-pval/mnlm)*nvox*nsamples))
        thcf = srw[imin]
        if verbose: print thg, thf, thcf
        
        if q<1:
            if verbose>1:
                _fig_density(sweight,surweight,pval,nlm)
                    
        for s in range(nbsubj):
            if nlm[s]>0:
                w1 = srweight-surweight[:,s]
                sw1 = np.sort(w1)
                imin = min(int((1-pval/nlm[s])*nvox*nsamples), nvox*nsamples-1)
                th1 = sw1[imin]
                w1 = sweight-weight[:,s]
                BFLc[s] = BFLs[s].copy()
                targets = w1[BFLc[s].roi_features['seed']]
                valid = (targets>th1)
                BFLc[s].select(np.ravel(valid))#
                nlm[s] = BFLc[s].k
                
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

    BFLs = [BFLc[s].copy() for s in range(nbsubj)]

    return a, b

def _clean_density(BFLs, dmax, xyz, pval=0.05, verbose=0, nrec=5, nsamples=10):
    """
    Computation of the positions where there is a significant
    spatial accumulation of pointwise ROIs. The significance is taken with
    respect to a uniform distribution of the ROIs.

    Parameters
    ----------
    BFLs : List of  nipy.neurospin.spatial_models.hroi.Nroi instances
          describing ROIs from different subjects
    dmax (float): the kernel width (std) for the spatial density estimator
    xyz (nbitems,dimension) array
        set of coordinates on which the test is perfomed
    pval: float, in [0,1], optional,
          corrected p-value for the  significance of the test
    verbose=0: verbosity mode
    nrec=5: number of recursions in the test: When some regions fail to be
            significant at one step, the density is recomputed, 
            and the test is performed again and so on
                
    Note
    ----
    Caveat 1: The NROI instances in BFLs must have a 'position' feature 
           defined beforehand
    Caveat 2: BFLs is edited and modified by this function
    The p-value is corrected only for the number of ROIs
        per subject
    """
    nbsubj = np.size(BFLs)
    nvox = xyz.shape[0]
    nlm = np.array([BFLs[s].k for s in range(nbsubj)]).astype(np.int)
 
    if verbose>0: print nlm
    Nlm = np.sum(nlm)
    Nlm0 = 2*Nlm
    q = 0
    while Nlm0>Nlm:
        Nlm0 = Nlm
        weight = _compute_density(BFLs,xyz,dmax)

        sweight = np.sum(weight,1)
        ssw = np.sort(sweight)
        thg = ssw[round((1-pval)*nvox)]

        # make surrogate data
        surweight = _compute_surrogate_density(BFLs,xyz,dmax,nsamples)
        srweight = np.sum(surweight,1)
        srw = np.sort(srweight)
        if verbose>0:
            thf = srw[int((1-min(pval,1))*nvox*nsamples)]
            mnlm = max(1,float(Nlm)/nbsubj)
            imin = min(nvox*nsamples-1,int((1.-pval/mnlm)*nvox*nsamples))
            thcf = srw[imin]
            print thg, thf, thcf
        
        if q<1:
            if verbose>1:
                _fig_density(sweight,surweight,pval,nlm)
                    
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
                nlm[s] = BFLs[s].k
            
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

def _fig_density(sweight, surweight, pval, nlm):
    """
    Plot the histogram of sweight across the image
    and the thresholds implied by the surrogate model (surweight)
    """
    import matplotlib.pylab as mp
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
    mp.figure(1)
    mp.plot(c,h)
    mp.plot(c0,h0)
    mp.legend(('true histogram','surrogate histogram'))
    mp.plot([thf,thf],[0,0.8*h0.max()])
    mp.text(thf,0.8*h0.max(),'p<0.2, uncorrected')
    mp.plot([thcf,thcf],[0,0.5*h0.max()])
    mp.text(thcf,0.5*h0.max(),'p<0.05, corrected')
    mp.savefig('/tmp/histo_density.eps')
    mp.show()


def _compute_density(BFLs, xyz, dmax):
    """
    Computation of the density of the BFLs points in the xyz volume
    dmax is a scale parameter
    """
    nvox = xyz.shape[0]
    nbsubj = np.size(BFLs)
    sqdmax = 2*dmax*dmax
    weight = np.zeros((nvox,nbsubj),'d')
    for s in range(nbsubj):
        if BFLs[s] is not None:
            coord = BFLs[s].get_roi_feature('position')
            for i in range(BFLs[s].k):
                dxyz = xyz - coord[i,:]
                dw = np.exp(-np.sum(dxyz**2,1)/sqdmax)
                weight[:,s] += dw
    return weight


def _compute_surrogate_density(BFLs, xyz, dmax, nsamples=1):
    """
    Cross-validated estimation of random samples of the uniform distributions
    
    Parameters
    ----------
    BFLs : a list of sets of ROIs the list length, nsubj, is taken
         as the number of subjects
    xyz (gs,3) array: a sampling grid to estimate
        spatial distribution
    dmax kernel width of the density estimator
    nsamples=1: number of surrogate smaples returned
    
    Returns
    -------
    surweight: a (gs*nsamples,nsubj) array of samples
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
                    dw = np.exp(-np.sum(dxyz*dxyz, 1)/sqdmax)
                    surweight[nvox*it:nvox*(it+1), s] += dw
    return surweight


def _hierarchical_asso(BF,dmax):
    """
    Compting an association graph of the ROIs defined across different subjects
    
    Parameters
    ----------
    BF : List of  nipy.neurospin.spatial_models.hroi.Nroi instances
          describing ROIs from different subjects
    dmax (float): spatial scale used xhen building associtations
    
    Returns
    -------
    G a graph that represent probabilistic associations between all
      cross-subject pairs of regions.
    
    Note that the probabilities are normalized
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
        edges = np.transpose([eA, eB]).astype(np.int)
        Gcorr = fg.WeightedGraph(cnlm[nbsubj], edges, eD)
    else:
        Gcorr = []
    return Gcorr



def RD_cliques(Gc, bstochastic=1):
    """
    Replicator dynamics graph segmentation: python implementation
    
    Parameters
    ----------
    Gc: nipy.neurospin.graph.WeightedGraph instance,
        the graph to be segmented
    bstochastic=1 stochastic initialization of the graph
    
    Returns
    -------
    labels : array of shape V, the number of vertices of Gc
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

def merge(Gc, labels):
    """
    Given a first labelling of the graph Gc, this function
    builds a reduced graph by merging the vertices according to the labelling

    Parameters
    ----------
    Gc nipy.neurospin.graph.graph.WeightedGraph instance
    labels array of shape V, the number of vertices of Gc
           the labelling of the vertices that represent the segmentation

    Returns
    -------
    labels array of shape V, the new labelling after further merging
    Gr the reduced graph after merging
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

    b = np.dot(Q.T,P) 
    b = (b.T/sum(b,1)).T
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
    

def segment_graph_rd(Gc, nit=1,verbose=0):
    """
    Hard segmentation of the graph Gc
    using a replicator dynamics approach.
    The clusters obtained in the first pass are further merged
    during a second pass, based on a reduced graph
    
    Parameters
    ----------
    Gc : nipy.neurospin.graph.graph.WeightedGraph instance
       the graph to be segmented
    
    Returns
    -------
    u : array of shape Gc.V labelling of the vertices 
      that represents the segmentation
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


def Compute_Amers(dom, lbeta, dmax=10., thr=3.0, ths=0, pval=0.2, verbose=0):
    """
    This is the main function for building the BFLs

    Parameters
    ----------
    dom : StructuredDomain instance,
          generic descriptor of the space domain
    lbeta: an array of shape (nbnodes, subjects):
           the multi-subject statistical maps
    dmax:float, optional, spatial relaxation allowed in the preocedure
    thr: float, optional, threshold at the first-level
    ths: float, optional,  number of subjects to validate a BFL
    pval: float, optional  : significance p-value for the spatial inference

    Returns
    -------
    crmap: array of shape (nnodes):
           the resulting group-level labelling of the space
    AF : a instance of LandmarkRegions that describes the ROIs found
        in inter-subject inference
        If no such thing can be defined LR is set to None
    BFLs:  List of  nipy.neurospin.spatial_models.hroi.Nroi instances
        representing individual ROIs
    Newlabel: labelling of the individual ROIs
    """
    BFLs = []
    nbsubj = lbeta.shape[1]
    nvox = lbeta.shape[0]

    from nipy.neurospin.graph.field import field_from_coo_matrix_and_data 
    Fbeta = field_from_coo_matrix_and_data(dom.topology, lbeta[:, 0])

    for s in range(nbsubj):
        beta = np.reshape(lbeta[:,s],(nvox,1))
        Fbeta.set_field(beta)
        bfls = hroi.HROI_from_watershed(dom, beta, threshold=thr)
 
        if bfls.k>0:
            bfls.make_feature('position', dom.coord)
            pos = bfls.representative_feature('position', 'mean')
            bfls.set_roi_feature('position', pos)
    
        BFLs.append(bfls)
    
    _clean_density_redraw(BFLs, dmax, dom.coord, pval, verbose=0,
                          nrec=1, nsamples=10)
    
    Gc = _hierarchical_asso(BFLs,dmax)
    Gc.weights = np.log(Gc.weights)-np.log(Gc.weights.min())
    if verbose:
        print Gc.V, Gc.E, Gc.weights.min(), Gc.weights.max()
    
    # building cliques
    u,cost = average_link_graph_segment(Gc, 0.1, Gc.V*1.0/nbsubj)

    # relabel the BFLs
    q = 0
    for s in range(nbsubj):
        BFLs[s].set_roi_feature('label',u[q:q+BFLs[s].k])
        q += BFLs[s].k
    
    LR, mlabel = build_LR(BFLs, ths=ths)
    if LR!=None:
        crmap = LR.map_label(dom.coord, pval=0.95, dmax=dmax)
        
    return crmap, LR, BFLs 


