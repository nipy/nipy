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
import fff2.graph.graph as fg
from fff2.graph import hroi 
from fff2.clustering import GGMixture
import fff2.clustering.clustering as fc
from fff2.graph import BPmatch
from fff2.clustering.hierarchical_clustering import Average_Link_Graph_segment

#------------------------------------------------------------------
#---------------- Auxiliary functions -----------------------------
#------------------------------------------------------------------



def hierarchical_asso(bfl,dmax):
    """
    Compting an association graph of the ROIs defined across different subjects
    INPUT:
    - bfl a list of ROI hierarchies, one for each subject
    - dmax : spatial scale used xhen building associtations
    OUPUT:
    - G a graph that represent probabilistic associations between all
    cross-subject pairs of regions. Note that the probabilities are normalized
    on a within-subject basis.
    """
    nbsubj = np.size(bfl)
    nlm = np.array([bfl[i].k for i in range(nbsubj)])
    cnlm = np.hstack(([0],np.cumsum(nlm)))
    if cnlm[nbsubj]==0:
        gcorr = []
        return gcorr

    gea = []
    geb = []
    ged = []
    for s in range(nbsubj):
        if (bfl[s].k>0):
            for t in range(s):
                if (bfl[t].k>0):
                    cs =  bfl[s].get_ROI_feature('coord')
                    ct = bfl[t].get_ROI_feature('coord')
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
    This function cleans the ROI_Hierarchy structure
    by merging small regions into their parent
    bf = _clean_size_(bf,smin)
    INPUT:
    - bf the hroi.ROI_Hierarchy to be cleaned
    - smin=0 the minimal size for ROIs
    OUTPUT:
    - bf the cleaned  hroi.ROI_Hierarchy
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
    This function cleans the ROI_Hierarchy structure
    by merging small regions into their parent
    bf = _clean_size_and_connectivity_(bf,Fbeta,smin)
    and by checking the simple connectivity of the areas in the hierarchy
    INPUT:
    - bf the hroi.ROI_Hierarchy to be cleaned
    - Fbeta: fff.field class, the underlying field of data
    - smin=0 the minimal size for ROIs
    OUTPUT:
    - bf the cleaned  hroi.ROI_Hierarchy
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

                    
def _GMM_priors_(beta,bfm,theta = 0,alpha=0.01,prior_strength = 100,verbose=0):
    """
    bfp = _GMM_priors_(beta,bfm,alpha=0.01,prior_strength = 100,verbose=0)
    Computing somr prior probabilities that the voxels of a certain map
    are in class disactivated, null or active
    INPUT:
    - beta array os shape (nvox): the map to be analysed
    - bfm array of shape(nbitems):
    the test values for which the p-value needs to be computed
    - theta = 0 the threshold above which the decision has to be made
    (normally, bfm>theta);
    if theta = -np.infty, then the method has a standard behaviour
    - alpha = 0.01 the prior weights of the positive and negative classes
    - prior_strength = 100 the confidence on the prior
    (should be compared to size(beta))
    - verbose=0 : verbosity mode
    OUPUT:
    bfp : array of shape (nbitems,3):
    the probability of each component in the MM for each test value
    """
    if np.size(bfm)==0:
        return None

    lnvox = np.size(beta)    
    from fff2.clustering.gmm import BGMM,grid_descriptor
    bgmm = BGMM(3,1,1)
    sbeta = np.sort(beta)
    mb0 = np.mean(sbeta[:alpha*lnvox])
    mb2 = np.mean(sbeta[(1-alpha)*lnvox:])
    prior_centers = np.reshape(np.array([mb0,0,mb2]),(3,1))
    prior_precision = np.ones((3,1))*1./prior_strength
    prior_dof = np.ones(3)*prior_strength
    prior_weights = np.array([alpha,1-2*alpha,alpha])*prior_strength
    prior_mean_scale = np.ones(3)*prior_strength
    bgmm.set_priors(prior_centers, prior_weights, prior_precision, prior_dof, prior_mean_scale)

    beta = np.reshape(beta,(lnvox,1))

    # only a few iterations are sufficient
    l = bgmm.Gibbs_estimate(beta,10)

    # estimate the prior weights
    gd = grid_descriptor(1) 
    gd.getinfo([beta.min(),beta.max()],100)
    gdm = gd.make_grid().squeeze()

    lj = bgmm.sample_on_data(gd.make_grid())
    lw = np.sum(lj[gdm>theta],0)
    bfp = lw/bgmm.weights*bgmm.sample_on_data(bfm)
    #bfp = bgmm.sample_on_data(bfm)

    if verbose>1:
        #bgmm.show(beta,gd,lj.sum(1))
        bgmm.show_components(beta,gd,lj)
    
    return bfp

def _GGM_priors_(beta,bfm,verbose=0):
    """
    bfp = _GGM_priors_(beta,bfm,verbose=0)
    Computing some prior probabilities that the voxels of a certain map
    are in class disactivated, null or active
    INPUT:
    - beta array os shape (nvox): the map to be analysed
    - bfm array of shape(nbitems):
    the test values for which the p-value needs to be computed
    - verbose=0 : verbosity mode
    OUPUT:
    bfp : array of shape (nbitems,3):
    the probability of each component in the MM for each test value
    """
    Ggg = GGMixture.GGGM()
    Ggg.init_fdr(beta)
    Ggg.estimate(beta,100,1.e-8,1.0,0)
    if verbose>1:
        # hyper-verbose mode
        Ggg.show(beta)
        Ggg.parameters()

    bfm = np.reshape(bfm,np.size(bfm))
    #?
    #bfp =  np.transpose(np.array(Ggg.posterior(bfm)))
    bfp = np.transpose(np.array(Ggg.component_likelihood(bfm)))
    return bfp

def make_crmap(AF,tal,verbose=0):
    """
    crmap = make_crmap(AF,tal)
    Compute the spatial map associated with the AF
    i.e. the confidence interfval for the position of
    the different landmarks
    INPUT:
    - AF the list of group-level landmarks regions
    - tal: array of shape(nvox,3): the position of the reference points
    OUPUT
    - crmap: array of shape(nvox)
    """
    nvox = tal.shape[0]
    crmap = -np.ones(nvox)
    gscore =  np.inf*np.ones(nvox)    
    print np.size(AF)
    for i in range(np.size(AF)):
        if verbose:
            print i, AF[i].k, AF[i].homogeneity(), AF[i].center()
        j,score = AF[i].confidence_region(tal)
        lscore = np.inf*np.ones(nvox)
        lscore[j] = score 
        crmap[gscore>lscore]=i
        gscore = np.minimum(gscore,lscore)
    return crmap

def infer_amers(BF,u,conf,thq=0.95,ths=0):
    """
    Given a list of hierarchical ROIs, and an associated labelling, this
    creates an Amer structure wuch groups ROIs with the same label.
    INPUT:
    - BF is the list of hierarchical ROIs.
    it is assumd that each list corresponds to one subject
    - u is a labelling array of size the total number of ROIs
    - ths=0 defines the condition (c):
    (c) A label should be present in ths subjects in order to be valid
    OUTPUT:
    - AF : a list of Amers, each of which describing a cross-subject set of ROIs
    - newlabel :  a relabelling of the individual ROIs, similar to u, which discards
    labels that do not fulfill the condition (c)
    """
    Sess = np.size(BF)
    Nlm = np.size(u)

    subj = np.concatenate([s*np.ones(BF[s].k, np.int) for s in range(Sess)])
    nrois = np.size(subj)
    if nrois != Nlm:
        raise ValueError, "incompatiable estimates of the number of regions"
    intrasubj = np.concatenate([np.arange(BF[s].k) for s in range(Sess)])
    newlabel = -np.ones(np.size(u), np.int)
    AF = []
    nl = 0
    if np.size(u)==0:  return AF,newlabel
    
    Mu = u.max()+1
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

        if st.norm.sf(ths,mp,np.sqrt(vp)) >thq:
            print nl,ths,mp,thq, st.norm.sf(ths,mp,np.sqrt(vp))
            newlabel[j] = nl
            sj = np.size(j)
            idx = np.zeros(sj)
            coord = np.zeros((sj,3), np.float)
            for a in range(sj):
                sja = subj[j[a]]
                isja = intrasubj[j[a]]
                idx[a] = BF[sja].seed[isja]
                coord[a,:] = BF[sja].get_ROI_feature('coord')[isja]

            amers = sbf.Amers(sj, subj[j], idx,coord)
            AF.append(amers)
            nl = nl+1
                    
    return AF,newlabel

#------------------------------------------------------------------
#------------------- main functions ----------------------------------
#------------------------------------------------------------------



def compute_BSA_ipmi(Fbeta,lbeta, tal,dmax, thq=0.5, smin=5, ths=0, 
                     theta=3.0, g0=1.0, bdensity=0, verbose=0):
    """
    Compute the  Bayesian Structural Activation paterns with approach described in IPMI'07 paper
    INPUT:
    - Fbeta : an fff field class describing the spatial relationships in the dataset (nbnodes nodes)
    - lbeta: an array of size (nbnodes, subjects) with functional data
    - tal: spatial coordinates of the nodes
    - thq = 0.5: posterior significance threshold
    - smin = 5: minimal size of the regions to validate them
    - theta = 3.0: first level threshold
    - g0 = 1.0 : constant values of the uniform density over the volume of interest
    - bdensity=0 if bdensity=1, the variable p in ouput contains the likelihood of the data under H1 on the set of input nodes
    OUTPUT:
    - crmap: resulting group map
    - AF: list of inter-subject related ROIs
    - BF: List of individual ROIs
    - u: labelling of the individual ROIs
    - p: likelihood of the data under H1 over some sampling grid
    NOTE:
    This is historically the first verion, but probably not the  most opimal
    It should not be changed for historical reason
    """
    BF = []
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
    
        if theta<beta.max():
            idx,height, parents,label = Fbeta.threshold_bifurcations(0,theta)
        else:
            idx = []
            parents = []
            label = -np.ones(np.shape(beta))
        
        k = np.size(idx)
        nroi = hroi.ROI_Hierarchy(k,idx, parents,label)      
        nroi = _clean_size_(nroi,smin)
        BF.append(nroi)

        # discard small regions
        # merge them into higher levels of the hierarchy
        if nroi.k>0:
            idx = nroi.get_seed()
            parents = nroi.get_parents()
            label = nroi.get_label()
            nroi.make_feature(beta, 'height','mean')
            bfm = nroi.get_ROI_feature('height')
            nroi.make_feature(tal.astype(np.float),'coord','cumulative_mean')
            bfc = nroi.get_ROI_feature('coord')

            gfc.append(bfc)

            # get some prior on the significance of the regions
            beta = np.reshape(beta,(nvox))
            beta = beta[beta!=0]

            # use a GGM
            bfp = _GGM_priors_(beta,bfm,verbose=0)
            bf0 = bfp[:,1]/np.sum(bfp,1)
            
            gf0.append(bf0)
            
            sub.append(s*np.ones(nroi.k))
            
    crmap = -np.ones(nvox, np.int)
    u = []
    AF = []
    p = np.zeros(nvox)
    if len(sub)<1:
        return crmap,AF,BF,u,p

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
        spatial_coords = tal
    else:
        spatial_coords = gfc

    p,q =  fc.fdp(gfc, 0.5, g0, g1,dof, prior_precision, 1-gf0, sub, 100, spatial_coords,10,1000)

    if verbose>1:
        import matplotlib.pylab as mp
        mp.figure()
        mp.plot(1-gf0,q,'.')
        #mp.show()

    # inference
    valid = q>thq
    if verbose:
        print np.sum(valid),np.size(valid)

    # remove non-significant regions
    for s in range(nbsubj):
        bf = BF[s]
        if bf.k>0:
            valids = valid[sub==s]
            valids = bf.propagate_AND_to_root(valids)
            bf.clean(valids)
            bf.merge_descending()
            bf.remove_feature('coord')
            bf.make_feature(tal.astype(np.float),'coord','cumulative_mean')

    # compute probabilitsic correspondences across subjects
    gc = hierarchical_asso(BF,np.sqrt(2)*dmax)

    # make hard clusters
    if gc != []:
        # choose one solution...
        #u = sbf.segment_graph_rd(gc,1)
        u,cost = Average_Link_Graph_segment(gc,0.2,gc.V*1.0/nbsubj)
            
        AF,u = sbf.Build_Amers(BF,u,ths) 
            
    # Finally make the ouput map
    crmap = make_crmap(AF,tal,verbose)
    
    return crmap,AF,BF,u,p

#-------------------------------------------------------------------------
# --------------- dev part _________________________----------------------
# ------------------------------------------------------------------------



def compute_BSA_dev (Fbeta, lbeta, tal, dmax, thq=0.9, smin=5, ths=0,
                     theta=3.0, g0=1.0, bdensity=0, verbose=0):
    """
    Compute the  Bayesian Structural Activation paterns
    INPUT:
    - Fbeta : an fff field class describing the spatial relationships in the dataset (nbnodes nodes)
    - lbeta: an array of size (nbnodes, subjects) with functional data
    - tal: spatial coordinates of the nodes
    - thq = 0.9: posterior significance threshold
    - smin = 5: minimal size of the regions to validate them
    - theta = 3.0: first level threshold
    - g0 = 1.0 : constant values of the uniform density over the volume of interest
    - bdensity=0 if bdensity=1, the variable p in ouput contains the likelihood of the data under H1 on the set of input nodes
    OUTPUT:
    - crmap: resulting group map
    - AF: list of inter-subject related ROIs
    - BF: List of individual ROIs
    - u: labelling of the individual ROIs
    - p: likelihood of the data under H1 over some sampling grid
    NOTE:
    This version is probably the best one to date
    the intra subject Gamma-Gaussian MM has been replaces by a Gaussian MM
    which is probably mroe robust
    """
    BF = []
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
    
        if theta<beta.max():
            idx,height, parents,label = Fbeta.threshold_bifurcations(0,theta)
        else:
            idx = []
            parents = []
            label = -np.ones(np.shape(beta))

        # remove the small regions
        k = np.size(idx)
        nroi = hroi.ROI_Hierarchy(k,idx, parents,label)      
        nroi = _clean_size_(nroi,smin)
        #nroi = _clean_size_and_connectivity_(nroi,Fbeta,smin)
        BF.append(nroi)
        
        if nroi.k>0:
            # find some way to avoid coordinate averaging
            idx = nroi.get_seed()
            parents = nroi.get_parents()
            label = nroi.get_label()
            nroi.make_feature(beta, 'height','mean')
            bfm = nroi.get_ROI_feature('height')
            nroi.make_feature(tal.astype(np.float),'coord','cumulative_mean')
            bfc = nroi.get_ROI_feature('coord')
            # Alan's choice
            #bfc = tal[nroi.argmax(beta)]
            #nroi.set_ROI_feature(bfc, 'coord')

            gfc.append(bfc)

            # compute the prior proba of being null
            beta = np.squeeze(beta)
            beta = beta[beta!=0]
            alpha = 0.01
            prior_strength = 100
            bfp = _GMM_priors_(beta,bfm,theta,alpha,prior_strength,verbose)
            bf0 = bfp[:,1]/np.sum(bfp,1)
            gf0.append(bf0)
            
            sub.append(s*np.ones(nroi.k))

    crmap = -np.ones(nvox, np.int)
    u = []
    AF = []
    p = np.zeros(nvox)
    if len(sub)<1:
        return crmap,AF,BF,u,p

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
        spatial_coords = tal
    else:
        spatial_coords = gfc
            
    p,q =  fc.fdp(gfc, 0.5, g0, g1, dof,prior_precision, 1-gf0, sub, 100, spatial_coords,10,1000)
    if verbose:
        import matplotlib.pylab as mp
        mp.figure()
        mp.plot(1-gf0,q,'.')
        #mp.show()

    valid = q>thq
    print np.sum(valid),np.size(valid)

    # remove non-significant region
    for s in range(nbsubj):
        bfs = BF[s]
        if bfs.k>0:
            valids = valid[sub==s]
            valids = bfs.propagate_AND_to_root(valids)
            bfs.clean(valids)
            bfs.merge_descending()
            bfs.remove_feature('coord')
            bfs.make_feature(tal.astype(np.float),'coord','cumulative_mean')
            # Alan's choice
            #beta = np.reshape(lbeta[:,s],(nvox,1))
            #bfsc = tal[bfs.argmax(beta)]
            #bfs.set_ROI_feature(bfsc,'coord')
                    
    gc = hierarchical_asso(BF,np.sqrt(2)*dmax)

    # Infer the group-level clusters
    if gc != []:
        # either replictor dynamics or agglomerative clustering
        #u = sbf.segment_graph_rd(gc,1)
        u,cost = Average_Link_Graph_segment(gc,0.1,gc.V*1.0/nbsubj)
                        
        AF,u = sbf.Build_Amers(BF,u,ths) 

    # make the group-level label map
    crmap = make_crmap(AF,tal,verbose)

    return crmap,AF,BF,u,p







def compute_BSA_simple(Fbeta, lbeta, tal, dmax, thq=0.5, smin=5, ths=0,
                        theta=3.0, g0=1.0, bdensity=0, verbose=0):
    """
    Compute the  Bayesian Structural Activation paterns - simplified version  
    INPUT:
    - Fbeta : an fff field class describing the spatial relationships in the dataset (nbnodes nodes)
    - lbeta: an array of size (nbnodes, subjects) with functional data
    - tal: spatial coordinates of the nodes
    - thq = 0.5: posterior significance threshold
    - smin = 5: minimal size of the regions to validate them
    - theta = 3.0: first level threshold
    - g0 = 1.0 : constant values of the uniform density over the volume of interest
    - bdensity=0 if bdensity=1, the variable p in ouput contains the likelihood of the data under H1 on the set of input nodes
    - verbose=1: verbosity mode
    OUTPUT:
    - crmap: resulting group map
    - AF: list of inter-subject related ROIs
    - BF: List of individual ROIs
    - u: labelling of the individual ROIs
    - p: likelihood of the data under H1 over some sampling grid
    NOTE:
    In that case, the DPMM is used to derive a spatial density of
    significant local maxima in the volume. Each terminal (leaf)
    region which is a posteriori significant enough is assigned to the
    nearest mode of theis distribution
    """
    BF = []
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
        nroi = Fbeta.generate_blobs(refdim=0,th=theta,smin = smin)
        BF.append(nroi) 
        
        if nroi.k>0:
            # find some way to avoid coordinate averaging
            nroi.make_feature(beta, 'height','mean')
            bfm = nroi.get_ROI_feature('height')[nroi.isleaf()]#---
            nroi.make_feature(tal.astype(np.float),'coord','cumulative_mean')
            bfc = nroi.get_ROI_feature('coord')[nroi.isleaf()]#---
            #
            gfc.append(bfc)

            # compute the prior proba of being null
            beta = np.squeeze(beta)
            beta = beta[beta!=0]

            # use a GMM model...
            alpha = 0.01
            prior_strength = 100
            bfp = _GMM_priors_(beta,bfm,theta,alpha,prior_strength,verbose)
            bf0 = bfp[:,1]/np.sum(bfp,1)
            #import scipy.stats as st
            #print s, st.norm.isf(bf0).mean(),
            # ... or the emp_null heuristic
            import fff2.utils.emp_null as en
            enn = en.ENN(beta)
            enn.learn()
            #bf0 = np.reshape(enn.fdr(bfm),np.size(bf0))
            
            gf0.append(bf0)
            sub.append(s*np.ones(np.size(bfm)))

    crmap = -np.ones(nvox, np.int)
    u = []
    AF = []
    p = np.zeros(nvox)
    if len(sub)<1:
        return crmap,AF,BF,u,p

    # prepare the DPMM
    sub = np.concatenate(sub).astype(np.int) 
    gfc = np.concatenate(gfc)
    gf0 = np.concatenate(gf0)
    g1 = g0
    prior_precision =  1./(dmax*dmax)*np.ones((1,3), np.float)
    dof = 100
    spatial_coords = tal
    burnin=100
    nis=100
    nii=1000

    p,q =  fc.fdp(gfc, 0.5, g0, g1, dof,prior_precision, 1-gf0, sub,burnin,spatial_coords,nis, nii)

    if verbose:
        import matplotlib.pylab as mp
        mp.figure()
        mp.plot(1-gf0,q,'.')
        h1,c1,p1 = mp.hist((1-gf0),bins=100)
        h2,c2,p2 = mp.hist(q,bins=100)
        mp.figure()
        # We use c1[:len(h1)] to be independant of the change in np.hist
        mp.bar(c1[:len(h1)],h1,width=0.005)
        mp.bar(c2[:len(h2)]+0.003,h2,width=0.005,color='r')
        mp.show()
        print 'Number of candidate regions %i, regions found %i' % (
                    np.size(q), q.sum())

    Fbeta.set_field(p)
    idx,depth, major,label = Fbeta.custom_watershed(0,g0)

    # label the blobs
    u = []
    qf = []
    for s in range(nbsubj):
        bfs = BF[s]
        bfs.remove_feature('coord')
        if bfs.k>0:
            leaves = bfs.isleaf()
            us = -np.ones(bfs.k).astype(np.int)
            lq = np.zeros(bfs.k)
            lq[leaves] = q[sub==s]
            beta = np.reshape(lbeta[:,s],(nvox,1))
            bfsc = tal[bfs.argmax(beta)]
            bfs.set_ROI_feature(bfsc,'coord')
            j = label[bfs.argmax(beta)]
            us[leaves] = j[leaves]
            us = bfs.propagate_upward(us)
            u.append(us)
            qf.append(lq)
            
    u = np.concatenate(u)
    qf = np.concatenate(qf)
    
    #valid = q>thq
    #if verbose: print np.sum(valid),np.size(valid),thq
   
    # derive the group-level landmarks
    # with a threshold on the number of subjects
    # that are represented in each one 
    AF,nl = infer_amers(BF,u,qf,thq,ths) 

    # make a group-level map of the landmark position
    #crmap = make_crmap(AF,tal,verbose)
    crmap = -np.ones(np.shape(label))
    q = 0
    for i in range(u.max()+1):
        if np.unique(nl[u==i])>-1:
            crmap[label==i]= np.unique(nl[u==i])            
            
    return crmap,AF,BF,nl,p

# ----------------------------------------------------------------
# ---------- Deprectaed stuff ------------------------------------
# ----------------------------------------------------------------


def _compute_BSA_simple_dep (Fbeta,lbeta, tal,dmax, thq=0.5, smin=5,ths = 0, theta=3.0, g0 = 1.0, bdensity=0,verbose=1):
    """
    Compute the  Bayesian Structural Activation paterns - simplified version  
    INPUT:
    - Fbeta : an fff field class describing the spatial relationships in the dataset (nbnodes nodes)
    - lbeta: an array of size (nbnodes, subjects) with functional data
    - tal: spatial coordinates of the nodes
    - thq = 0.5: posterior significance threshold
    - smin = 5: minimal size of the regions to validate them
    - theta = 3.0: first level threshold
    - g0 = 1.0 : constant values of the uniform density over the volume of interest
    - bdensity=0 if bdensity=1, the variable p in ouput contains the likelihood of the data under H1 on the set of input nodes
    - verbose=1: verbosity mode
    OUTPUT:
    - crmap: resulting group map
    - AF: list of inter-subject related ROIs
    - BF: List of individual ROIs
    - u: labelling of the individual ROIs
    - p: likelihood of the data under H1 over some sampling grid
    NOTE:
    In that case, the DPMM is used to derive a spatial density of
    significant local maxima in the volume. Each terminal (leaf)
    region which is a posteriori significant enough is assigned to the
    nearest mode of theis distribution
    """
    BF = []
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
        nroi = Fbeta.generate_blobs(refdim=0,th=theta,smin = smin)

        # consider only the rois associated with maxima
        # hence with no hierrachy
        #nroi = nroi.reduce_to_leaves()

        BF.append(nroi) 
        
        if nroi.k>0:
            # find some way to avoid coordinate averaging
            idx = nroi.get_seed()
            parents = nroi.get_parents()
            label = nroi.get_label()
            nroi.make_feature(beta, 'height','mean')
            bfm = nroi.get_ROI_feature('height')[nroi.isleaf()]#---
            nroi.make_feature(tal.astype(np.float),'coord','cumulative_mean')
            bfc = nroi.get_ROI_feature('coord')[nroi.isleaf()]#---
            #
            gfc.append(bfc)

            # compute the prior proba of being null
            beta = np.squeeze(beta)
            beta = beta[beta!=0]

            # use a GMM model...
            alpha = 0.01
            prior_strength = 100
            bfp = _GMM_priors_(beta,bfm,theta,alpha,prior_strength,verbose)
            bf0 = bfp[:,1]/np.sum(bfp,1)
            #import scipy.stats as st
            #print s, st.norm.isf(bf0).mean(),
            # ... or the emp_null heuristic
            import fff2.utils.emp_null as en
            enn = en.ENN(beta)
            enn.learn()
            #bf0 = np.reshape(enn.fdr(bfm),np.size(bf0))
            #print st.norm.isf(bf0).mean()
            
            gf0.append(bf0)
            sub.append(s*np.ones(np.size(bfm)))

    crmap = -np.ones(nvox, np.int)
    u = []
    AF = []
    p = np.zeros(nvox)
    if len(sub)<1:
        return crmap,AF,BF,u,p

    # prepare the DPMM
    sub = np.concatenate(sub).astype(np.int) 
    gfc = np.concatenate(gfc)
    gf0 = np.concatenate(gf0)
    g1 = g0
    prior_precision =  1./(dmax*dmax)*np.ones((1,3), np.float)
    dof = 100
    spatial_coords = tal
    burnin=100
    nis=100
    nii=1000

    p,q =  fc.fdp(gfc, 0.5, g0, g1, dof,prior_precision, 1-gf0, sub,burnin,spatial_coords,nis, nii)
    
    if verbose:
        import matplotlib.pylab as mp
        mp.figure()
        mp.plot(1-gf0,q,'.')

    # threshold the blobs according to their posterior probability
    # NB: this cooresponds to a BF of thq/(1-thq)
    valid = q>thq
    if verbose:
        print np.sum(valid),np.size(valid),thq

    # make a watershed of the posterior density map
    # in order to have the modes and assign each selected blob
    # to the nearest mode
    Fbeta.set_field(p)
    idx,depth, major,label = Fbeta.custom_watershed(0,g0)

    # label the blobs
    for s in range(nbsubj):
        bfs = BF[s]
        if bfs.k>0:
            valids = np.zeros(bfs.k,'bool')
            #valids = valid[sub==s] # ---
            valids[bfs.isleaf()] = valid[sub==s] # ---
            valids = bfs.propagate_AND_to_root(valids)
            bfs.clean(valids)
            bfs.merge_descending()
            bfs.remove_feature('coord')
            if bfs.k>0:
                leaves = bfs.isleaf()
                us = -np.ones(bfs.k).astype(np.int)
                beta = np.reshape(lbeta[:,s],(nvox,1))
                bfsc = tal[bfs.argmax(beta)]
                bfs.set_ROI_feature(bfsc,'coord')
                j = label[bfs.argmax(beta)]
                us[leaves] = j[leaves]
                us = bfs.propagate_upward(us)
                u.append(us)

    if len(u)==0:
        return crmap,AF,BF,u,p
    
    u = np.concatenate(u)

    # derive the group-level landmarks
    # with a threshold on the number of subjects
    # that are represented in each one 
    AF,u = sbf.Build_Amers(BF,u,ths) 

    # make a group-level map of the landmark position
    crmap = make_crmap(AF,tal,verbose)
    
    return crmap,AF,BF,u,p
