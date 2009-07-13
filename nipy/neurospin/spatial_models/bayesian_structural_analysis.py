"""
The main routine of this package that aims at performing the
extraction of ROIs from multisubject dataset using the localization
and activation strength of extracted regions.  This has been puclished
in Thirion et al. High level group analysis of FMRI data based on
Dirichlet process mixture models, IPMI 2007

Author : Bertrand Thirion, 2006-2009
"""

import numpy as np
import scipy.stats as st

import structural_bfls as sbf
import nipy.neurospin.graph.graph as fg
from nipy.neurospin.spatial_models import hroi 

import nipy.neurospin.clustering.clustering as fc
from nipy.neurospin.graph import BPmatch
from nipy.neurospin.clustering.hierarchical_clustering import Average_Link_Graph_segment
import nipy.neurospin.utils.emp_null as en

#------------------------------------------------------------------
#---------------- Auxiliary functions -----------------------------
#------------------------------------------------------------------


def hierarchical_asso(bfl,dmax):
    """
    Compting an association graph of the ROIs defined
    across different subjects

    INPUT:
    ------
    - bfl a list of ROI hierarchies, one for each subject
    - dmax : spatial scale used when building associtations

    OUTPUT:
    ------
    - G a graph that represent probabilistic associations between all
    cross-subject pairs of regions.
    Note that the probabilities are normalized
    on a within-subject basis.
    """
    nbsubj = np.size(bfl)
    nlm = np.zeros(nbsubj)
    for i in range(nbsubj):
        if bfl[i]!=None:
            nlm[i] = bfl[i].k

    cnlm = np.hstack(([0],np.cumsum(nlm)))
    if cnlm.max()==0:
        gcorr = []
        return gcorr

    gea = []
    geb = []
    ged = []
    for s in range(nbsubj):
        if (bfl[s]!=None):
            for t in range(s):
                if (bfl[t]!=None):
                    cs =  bfl[s].get_roi_feature('position')
                    ct = bfl[t].get_roi_feature('position')
                    Gs = bfl[s].make_forest()
                    Gs.anti_symmeterize()
            
                    Gt = bfl[t].make_forest()
                    Gt.anti_symmeterize()

                    #ea,eb,ed = BPmatch.BPmatch_slow_asym(cs,ct, Gs,Gt,dmax)
                    ea,eb,ed = BPmatch.BPmatch_slow_asym_dev(cs,ct, Gs,Gt,dmax)
                    if np.size(ea)>0:
                        gea = np.hstack((gea,ea+cnlm[s]))
                        geb = np.hstack((geb,eb+cnlm[t]))
                        ged = np.hstack((ged,ed))

                    #ea,eb,ed = BPmatch.BPmatch_slow_asym_dev(ct,cs, Gt,Gs,dmax)
                    ea,eb,ed = BPmatch.BPmatch_slow_asym_dev(ct,cs, Gt,Gs,dmax)
                    if np.size(ea)>0:
                        gea = np.hstack((gea,ea+cnlm[t]))
                        geb = np.hstack((geb,eb+cnlm[s]))
                        ged = np.hstack((ged,ed))

    if np.size(gea)>0:
        edges = np.transpose([gea,geb]).astype(np.int)
        gcorr = fg.WeightedGraph(cnlm[nbsubj],edges,ged)
    else:
        gcorr = []
    return gcorr

def _clean_size_(bf,smin=0):
    """
    This function cleans the nested ROI structure
    by merging small regions into their parent
    bf = _clean_size_(bf,smin)

    INPUT:
    ------
    - bf the hroi.NROI to be cleaned
    - smin=0 the minimal size for ROIs

    OUTPUT:
    -------
    - bf the cleaned  hroi.NROI
    """
    k = 2* bf.get_k()
    if k>0:
        while k>bf.get_k():
            k = bf.get_k()
            size = bf.compute_size()
            bf.merge_ascending(size>smin,None)
            bf.merge_descending(None)
            size = bf.compute_size()
            bf.clean(size>smin)
            bf.check()
    return bf

def _clean_size_and_connectivity_(bf,Fbeta,smin=0):
    """
    This function cleans the nested ROI structure
    by merging small regions into their parent
    bf = _clean_size_and_connectivity_(bf,Fbeta,smin)
    and by checking the simple connectivity of the areas in the hierarchy

    INPUT:
    ------
    - bf the hroi.NROI to be cleaned
    - Fbeta: fff.field class, the underlying field of data
    - smin=0 the minimal size for ROIs

    OUTPUT:
    -------
    - bf the cleaned  hroi.NROI
    NOTE : it may be slow
    """
    bf = _clean_size_(bf,smin)
    if bf.k<1: return bf
    #import time
    #t1 = time.time()

    for i in range(bf.k):
        l = bf.subtree(i)
        valid = np.zeros(bf.k)
        valid[l]=1
        vvalid = np.zeros(Fbeta.V)
        vvalid[bf.label>-1] = valid[bf.label[bf.label>-1]]
        vvalid = 1-vvalid
        if np.sum(vvalid)>0:
            g = Fbeta.subgraph(vvalid)
            iv = np.nonzero(vvalid)[0]
            u = g.cc()
            if u.max()>0:
                mccs = np.size(g.main_cc())
                for j in range(u.max()+1):
                    if np.sum(u==j)<mccs:
                        bf.label[iv[u==j]]=i
    #t2 = time.time()
    bf = _clean_size_(bf,smin)
    return bf

def make_crmap(AF,coord,verbose=0):
    """
    crmap = make_crmap(AF,coord)
    Compute the spatial map associated with the AF
    i.e. the confidence interfval for the position of
    the different landmarks
    
    INPUT:
    ------
    - AF the list of group-level landmarks regions
    - coord: array of shape(nvox,3): the position of the reference points

    OUTPUT:
    -----
    - crmap: array of shape(nvox)
    """
    nvox = coord.shape[0]
    crmap = -np.ones(nvox)
    gscore =  np.inf*np.ones(nvox)    
    print np.size(AF)
    for i in range(np.size(AF)):
        if verbose:
            print i, AF[i].k, AF[i].homogeneity(), AF[i].center()
        j,score = AF[i].confidence_region(coord)
        lscore = np.inf*np.ones(nvox)
        lscore[j] = score 
        crmap[gscore>lscore]=i
        gscore = np.minimum(gscore,lscore)
    return crmap

def infer_LR(bf,thq=0.95,ths=0,verbose=0):
    """
    Given a list of hierarchical ROIs, and an associated labelling, this
    creates an Amer structure wuch groups ROIs with the same label.
    
    INPUT:
    ------
    - bf is the list of NROIs (Nested ROIs).
    it is assumd that each list corresponds to one subject
    each NROI is assumed to have the roi_features
    'position', 'label' and 'posterior_proba' defines
    - thq=0.95,ths=0 defines the condition (c):
    (c) A label should be present in ths subjects
    with a probability>thq
    in order to be valid
    
    OUTPUT:
    -------
    - LR : a LR instance, describing a cross-subject set of ROIs
    if inference yields a null results, LR is set to None
    - newlabel :  a relabelling of the individual ROIs, similar to u,
    which discards
    labels that do not fulfill the condition (c)
    """
    # prepare various variables to ease information manipulation
    nbsubj = np.size(bf)
    subj = np.concatenate([s*np.ones(bf[s].k, np.int)
                           for s in range(nbsubj) if bf[s]!=None])
    nrois = np.size(subj)
    u = np.concatenate([bf[s].get_roi_feature('label')
                        for s in range(nbsubj)if bf[s]!=None])
    u = np.squeeze(u)
    conf =  np.concatenate([bf[s].get_roi_feature('prior_proba')
                            for s in range(nbsubj) if bf[s]!=None])
    intrasubj = np.concatenate([np.arange(bf[s].k)
                                for s in range(nbsubj) if bf[s]!=None])
    
    if np.size(u)==0:  return None,None

    coords = []
    subjs=[]
    pps = []
    Mu = int(u.max()+1)
    valid = np.zeros(Mu).astype(np.int)

    # do some computation to find which regions are worth reporting
    for i in range(Mu):
        j = np.nonzero(u==i)
        j = np.reshape(j,np.size(j))
        mp = 0.
        vp = 0.
        if np.size(j)>1:
            subjj = subj[j]
            for ls in np.unique(subjj):
                lmj = 1-np.prod(1-conf[(u==i)*(subj==ls)])
                lvj = lmj*(1-lmj)
                mp = mp+lmj
                vp = vp+lvj
        # If noise is too low the variance is 0: ill-defined:
        vp = max(vp, 1e-14)

        # if above threshold, get some information to create the LR
        if st.norm.sf(ths,mp,np.sqrt(vp)) >thq:         
            if verbose:
                print valid.sum(),ths,mp,thq, st.norm.sf(ths,mp,np.sqrt(vp))
            valid[i]=1
            sj = np.size(j)
            idx = np.zeros(sj)
            coord = np.zeros((sj,3), np.float)
            for a in range(sj):
                sja = subj[j[a]]
                isja = intrasubj[j[a]]
                coord[a,:] = bf[sja].get_roi_feature('position')[isja]

            coords.append(coord)
            subjs.append(subj[j])   
            pps.append(conf[j])

    maplabel = -np.ones(Mu).astype(np.int)
    maplabel[valid>0] = np.cumsum(valid[valid>0])-1
       
    # relabel the ROIs
    for s in range(nbsubj):
        if bf[s]!=None:
            us = bf[s].get_roi_feature('label')
            us[us>-1] = maplabel[us[us>-1]]
            bf[s].set_roi_feature('label',us)
            header = bf[s].header

    # create the landmark regions structure
    k = np.sum(valid)
    if k>0:
        LR = sbf.landmark_regions(k,header=header,subj=subjs,coord=coords)
        LR.set_discrete_feature('confidence', pps)
    else:
        LR=None
    return LR,maplabel


#------------------------------------------------------------------
#------------------- main functions ----------------------------------
#------------------------------------------------------------------



def compute_BSA_ipmi(Fbeta,lbeta, coord,dmax, xyz, header, thq=0.5,
                     smin=5, ths=0, theta=3.0, g0=1.0, bdensity=0, verbose=0):
    """
    Compute the  Bayesian Structural Activation patterns
    with approach described in IPMI'07 paper

    INPUT:
    ------
    - Fbeta : an fff field class describing the spatial relationships
    in the dataset (nbnodes nodes)
    - lbeta: an array of size (nbnodes, subjects) with functional data
    - coord: spatial coordinates of the nodes
    - thq = 0.5: posterior significance threshold
    - smin = 5: minimal size of the regions to validate them
    - theta = 3.0: first level threshold
    - g0 = 1.0 : constant values of the uniform density
    over the volume of interest
    - bdensity=0 if bdensity=1, the variable p in ouput contains
    the likelihood of the data under H1 on the set of input nodes

    OUTPUT:
    -------
    - crmap: resulting group map
    - AF: list of inter-subject related ROIs
    - bf: List of individual ROIs
    - u: labelling of the individual ROIs
    - p: likelihood of the data under H1 over some sampling grid

    NOTE:
    -----
    This is historically the first version, but probably not the  most optimal
    It should not be changed for historical reason
    """
    bf = []
    gfc = []
    gf0 = []
    sub = []
    gc = []
    nbsubj = lbeta.shape[1]
    nvox = lbeta.shape[0]

    # intra-subject part: compute the blobs
    # and assess their significance
    for s in range(nbsubj):
        beta = np.reshape(lbeta[:,s],(nvox,1))
        Fbeta.set_field(beta)
        nroi = hroi.NROI_from_field(Fbeta,header,xyz,refdim=0,
                                    th=theta,smin=smin)
        bf.append(nroi)
        
        if nroi!=None:
            sub.append(s*np.ones(nroi.k))
            # find some way to avoid coordinate averaging
            bfm = nroi.discrete_to_roi_features('activation','average')
            nroi.compute_discrete_position()
            bfc = nroi.discrete_to_roi_features('position',
                                                'cumulated_average')            
            gfc.append(bfc)

            # get some prior on the significance of the regions
            beta = np.reshape(beta,(nvox))
            beta = beta[beta!=0]

            # use a Gamma-Gaussian Mixture Model
            bfp  = en.Gamma_Gaussian_fit(beta,bfm,verbose)
            bf0 = bfp[:,1]

            gf0.append(bf0)
                      
    crmap = -np.ones(nvox, np.int)
    u = []
    AF = []
    p = np.zeros(nvox)
    if len(sub)<1:
        return crmap,AF,bf,u,p

    # inter-subject analysis
    # use the DPMM (core part)
    sub = np.concatenate(sub).astype(np.int) 
    gfc = np.concatenate(gfc)
    gf0 = np.concatenate(gf0)
    p = np.zeros(np.size(nvox))
    g1 = g0
    dof = 0
    prior_precision =  1./(dmax*dmax)*np.ones((1,3), np.float)

    if bdensity:
        spatial_coords = coord
    else:
        spatial_coords = gfc

    p,q =  fc.fdp(gfc, 0.5, g0, g1,dof, prior_precision,
                  1-gf0, sub, 100, spatial_coords,10,1000)
    # inference
    valid = q>thq

    if verbose>1:
        import matplotlib.pylab as mp
        mp.figure()
        mp.plot(1-gf0,q,'.')
        print np.sum(valid),np.size(valid)

    # remove non-significant regions
    for s in range(nbsubj):
        bfs = bf[s]
        if bfs!=None:
            valids = valid[sub==s]
            valids = bfs.propagate_upward_and(valids)
            bfs.clean(valids)
            
        if bfs!=None:
            bfs.merge_descending()
            bfs.compute_discrete_position()
            bfs.discrete_to_roi_features('position','cumulated_average')

    # compute probabilitsic correspondences across subjects
    gc = hierarchical_asso(bf,np.sqrt(2)*dmax)

    if gc == []:
        return crmap,AF,bf,p

    # make hard clusters
    # choose one solution...
    #u = sbf.segment_graph_rd(gc,1)
    u,cost = Average_Link_Graph_segment(gc,0.2,gc.V*1.0/nbsubj)

    q = 0
    for s in range(nbsubj):
        if bf[s]!=None:
            bf[s].set_roi_feature('label',u[q:q+bf[s].k])
            q += bf[s].k
    
    LR,mlabel = sbf.build_LR(bf,ths)
    if LR!=None:
        crmap = LR.map_label(coord,pval=0.95,dmax=dmax)
    
    return crmap,LR,bf,p

#------------------------------------------------------------------
# --------------- dev part ----------------------------------------
# -----------------------------------------------------------------



def compute_BSA_dev (Fbeta, lbeta, coord, dmax,  xyz, header,
                     thq=0.9,smin=5, ths=0,theta=3.0, g0=1.0,
                     bdensity=0, verbose=0):
    """
    Compute the  Bayesian Structural Activation paterns

    INPUT:
    ------
    - Fbeta : an fff field class describing the spatial
    relationships in the dataset (nbnodes nodes)
    - lbeta: an array of size (nbnodes, subjects) with functional data
    - coord: spatial coordinates of the nodes
    - thq = 0.9: posterior significance threshold
    - smin = 5: minimal size of the regions to validate them
    - theta = 3.0: first level threshold
    - g0 = 1.0 : constant values of the uniform density
    over the volume of interest
    - bdensity=0 if bdensity=1, the variable p in ouput
    contains the likelihood of the data under H1 on the set of input nodes

    OUTPUT:
    -------
    - crmap: resulting group map
    - AF: list of inter-subject related ROIs
    - bf: List of individual ROIs
    - u: labelling of the individual ROIs
    - p: likelihood of the data under H1 over some sampling grid

    NOTE:
    -----
    This version is probably the best one to date
    the intra subject Gamma-Gaussian MM has been replaces by a Gaussian MM
    which is probably mroe robust
    """
    bf = []
    gfc = []
    gf0 = []
    sub = []
    gc = []
    nbsubj = lbeta.shape[1]
    nvox = lbeta.shape[0]

    # intra-subject analysis: get the blobs,
    # with their position and their significance
    for s in range(nbsubj):       
        # description in terms of blobs
        beta = np.reshape(lbeta[:,s],(nvox,1))
        Fbeta.set_field(beta)
        nroi = hroi.NROI_from_field(Fbeta,header,xyz,refdim=0,th=theta,smin=smin)
        bf.append(nroi)
        
        if nroi!=None:
            sub.append(s*np.ones(nroi.k))
            # find some way to avoid coordinate averaging
            bfm = nroi.discrete_to_roi_features('activation','average')

            # compute the region position
            nroi.compute_discrete_position()
            bfc = nroi.discrete_to_roi_features('position','cumulated_average')            
            gfc.append(bfc)

            # compute the prior proba of being null
            beta = np.squeeze(beta)
            beta = beta[beta!=0]
            alpha = 0.01
            prior_strength = 100
            bfp = en.three_classes_GMM_fit(beta, bfm, alpha,
                                        prior_strength,verbose)
            bf0 = bfp[:,1]
            gf0.append(bf0)
            
    crmap = -np.ones(nvox, np.int)
    u = []
    AF = []
    p = np.zeros(nvox)
    if len(sub)<1:
        return crmap,AF,bf,u,p

    # inter-subject analysis
    # use the DPMM (core part)
    sub = np.concatenate(sub).astype(np.int)
    gfc = np.concatenate(gfc)
    gf0 = np.concatenate(gf0)
    p = np.zeros(np.size(nvox))
    g1 = g0
    dof = 0
    prior_precision =  1./(dmax*dmax)*np.ones((1,3), np.int)

    if bdensity:
        spatial_coords = coord
    else:
        spatial_coords = gfc
            
    p,q =  fc.fdp(gfc, 0.5, g0, g1, dof,prior_precision, 1-gf0, sub, 100, spatial_coords,10,1000)
    valid = q>thq
    if verbose:
        import matplotlib.pylab as mp
        mp.figure()
        mp.plot(1-gf0,q,'.')    
        print np.sum(valid),np.size(valid)

    # remove non-significant regions
    for s in range(nbsubj):
        bfs = bf[s]
        if bfs!=None:
            valids = valid[sub==s]
            valids = bfs.propagate_upward_and(valids)
            bfs.clean(valids)
            bfs.merge_descending()
            
            # re-compute the region position
            bfs.compute_discrete_position()
            bfc = bfs.discrete_to_roi_features('position','cumulated_average')
            # Alan's choice
            #beta = np.reshape(lbeta[:,s],(nvox,1))
            #bfsc = coord[bfs.feature_argmax(beta)]
            #bfs.set_roi_feature(bfsc,'position')

    # compute a model of between-regions associations
    gc = hierarchical_asso(bf,np.sqrt(2)*dmax)

    # Infer the group-level clusters
    if gc == []:
        return crmap,AF,bf,p

    # either replicator dynamics or agglomerative clustering
    #u = sbf.segment_graph_rd(gc,1)
    u,cost = Average_Link_Graph_segment(gc,0.1,gc.V*1.0/nbsubj)

    q = 0
    for s in range(nbsubj):
        if bf[s]!=None:
            bf[s].set_roi_feature('label',u[q:q+bf[s].k])
            q += bf[s].k
    
    LR,mlabel = sbf.build_LR(bf,ths)
    if LR!=None:
        crmap = LR.map_label(coord,pval = 0.95,dmax=dmax)

    return crmap,LR,bf,p


def compute_BSA_simple(Fbeta, lbeta, coord, dmax, xyz, header=None,
                       thq=0.5, smin=5, ths=0, theta=3.0, g0=1.0,
                       verbose=0):
    """
    Compute the  Bayesian Structural Activation paterns - simplified version  

    INPUT:
    ------
    - Fbeta : an fff field class describing the spatial relationships
    in the dataset (nbnodes nodes)
    - lbeta: an array of size (nbnodes, subjects) with functional data
    - coord: spatial coordinates of the nodes
    - xyz: the grid coordinates of the field
    - header=None: the referential defining header
    - thq = 0.5: posterior significance threshold
    - smin = 5: minimal size of the regions to validate them
    - theta = 3.0: first level threshold
    - g0 = 1.0 : constant values of the uniform density
    over the volume of interest
    - bdensity=0 if bdensity=1, the variable p in ouput
    contains the likelihood of the data under H1 on the set of input nodes
    - verbose=0: verbosity mode

    OUTPUT:
    -------
    - crmap: resulting group map
    - LR: a instance of sbf.Landmrak_regions that describes the ROIs found
    in inter-subject inference
    If no such thing can be defined LR is set to None
    - bf: List of individual ROIs
    - p: likelihood of the data under H1 over some sampling grid

    NOTE:
    -----
    In that case, the DPMM is used to derive a spatial density of
    significant local maxima in the volume. Each terminal (leaf)
    region which is a posteriori significant enough is assigned to the
    nearest mode of theis distribution
    """
    bf = []
    gfc = []
    gf0 = []
    sub = []
    gc = []
    nbsubj = lbeta.shape[1]
    nvox = lbeta.shape[0]

    for s in range(nbsubj):
        
        # description in terms of blobs
        beta = np.reshape(lbeta[:,s],(nvox,1))
        Fbeta.set_field(beta)
        nroi = hroi.NROI_from_field(Fbeta,header,xyz,refdim=0,
                                    th=theta,smin=smin)
        bf.append(nroi)
        
        if nroi!=None:
            bfm = nroi.discrete_to_roi_features('activation','average')
            bfm = bfm[nroi.isleaf()]

            # get the regions position
            nroi.compute_discrete_position()
            bfc = nroi.discrete_to_roi_features('position','average')
            bfc = bfc[nroi.isleaf()]
            gfc.append(bfc)

            # compute the prior proba of being null
            beta = np.squeeze(beta)
            beta = beta[beta!=0]

            # use a GMM model...
            alpha = 0.01
            prior_strength = 100
            bfp = en.three_classes_GMM_fit(beta, bfm, alpha,
                                        prior_strength,verbose)
            bf0 = bfp[:,1]
            
            ## ... or the emp_null heuristic
            #enn = en.ENN(beta)
            #enn.learn()
            #bf0 = np.reshape(enn.fdr(bfm),np.size(bf0))
            
            gf0.append(bf0)
            sub.append(s*np.ones(np.size(bfm)))

    crmap = -np.ones(nvox, np.int)
    u = []
    LR = None
    p = np.zeros(nvox)
    if len(sub)<1:
        return crmap,LR,bf,p

    # prepare the DPMM
    sub = np.concatenate(sub).astype(np.int) 
    gfc = np.concatenate(gfc)
    gf0 = np.concatenate(gf0)
    g1 = g0
    prior_precision =  1./(dmax*dmax)*np.ones((1,3), np.float)
    dof = 100
    spatial_coords = coord
    burnin = 100
    nis = 1000
    nii = 100
    
    p,q =  fc.fdp(gfc, 0.5, g0, g1, dof,prior_precision, 1-gf0,
                  sub,burnin,spatial_coords,nis, nii)

    if verbose:
        import matplotlib.pylab as mp
        mp.figure()
        mp.plot(1-gf0,q,'.')
        h1,c1 = mp.histogram((1-gf0),bins=100)
        h2,c2 = mp.histogram(q,bins=100)
        mp.figure()
        # We use c1[:len(h1)] to be independant of the change in np.hist
        mp.bar(c1[:len(h1)],h1,width=0.005)
        mp.bar(c2[:len(h2)]+0.003,h2,width=0.005,color='r')
        print 'Number of candidate regions %i, regions found %i' % (
                    np.size(q), q.sum())
    
    Fbeta.set_field(p)
    idx,depth, major,label = Fbeta.custom_watershed(0,g0)

    # append some information to the hroi in each subject
    for s in range(nbsubj):
        bfs = bf[s]
        if bfs!=None:
            leaves = bfs.isleaf()
            us = -np.ones(bfs.k).astype(np.int)
            lq = np.zeros(bfs.k)
            lq[leaves] = q[sub==s]
            bfs.set_roi_feature('posterior_proba',lq)
            lq = np.zeros(bfs.k)
            lq[leaves] = 1-gf0[sub==s]
            bfs.set_roi_feature('prior_proba',lq)
                   
            idx = bfs.feature_argmax('activation')
            midx = [bfs.discrete_features['masked_index'][k][idx[k]] for k in range(bfs.k)]
            j = label[np.array(midx)]
            us[leaves] = j[leaves]

            # when parent regions has similarly labelled children,
            # include it also
            us = bfs.propagate_upward(us)
            bfs.set_roi_feature('label',us)
                        
    # derive the group-level landmarks
    # with a threshold on the number of subjects
    # that are represented in each one 
    LR,nl = infer_LR(bf,thq,ths,verbose=verbose)

    # make a group-level map of the landmark position
    crmap = -np.ones(np.shape(label))
    if nl!=None:
        aux = np.arange(label.max()+1)
        aux[0:np.size(nl)]=nl
        crmap[label>-1]=aux[label[label>-1]]
 
            
    return crmap,LR,bf,p
