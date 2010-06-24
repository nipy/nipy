# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
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
from nipy.neurospin.clustering.hierarchical_clustering import\
     average_link_graph_segment
import nipy.neurospin.utils.emp_null as en

####################################################################
# Ancillary functions
####################################################################


def _hierarchical_asso(bfl,dmax):
    """
    Compting an association graph of the ROIs defined
    across different subjects

    Parameters
    ----------
    bfl a list of ROI hierarchies, one for each subject
    dmax : spatial scale used when building associtations

    Results
    -------
    G a graph that represent probabilistic associations between all
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

                    ea,eb,ed = BPmatch.BPmatch_slow_asym_dev(
                        cs, ct, Gs, Gt,dmax)
                    if np.size(ea)>0:
                        gea = np.hstack((gea,ea+cnlm[s]))
                        geb = np.hstack((geb,eb+cnlm[t]))
                        ged = np.hstack((ged,ed))

                    ea,eb,ed = BPmatch.BPmatch_slow_asym_dev(
                        ct, cs, Gt, Gs, dmax)
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

def infer_LR(bf, thq=0.95, ths=0, verbose=0):
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
    
    Results
    -------
    LR : a LR instance, describing a cross-subject set of ROIs
       if inference yields a null results, LR is set to None
    newlabel: a relabelling of the individual ROIs, similar to u,
              which discards
              labels that do not fulfill the condition (c)

    Fixme
    -----
    Should be merged with sbf.build_LR
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

    for s in range(nbsubj):
        if bf[s]is not None:
            dim = len(bf[s].shape)
    
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
                print valid.sum(),ths,mp,thq,\
                      st.norm.sf(ths,mp,np.sqrt(vp))
            valid[i]=1
            sj = np.size(j)
            idx = np.zeros(sj)
            coord = np.zeros((sj, dim), np.float)
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
            affine = bf[s].affine
            shape = bf[s].shape

    # create the landmark regions structure
    k = np.sum(valid)
    if k>0:
        LR = sbf.landmark_regions(k, affine=affine, shape=shape, subj=subjs,
                                  coord=coords)
        LR.set_discrete_feature('confidence', pps)
    else:
        LR = None
    return LR,maplabel

def _relabel_(label, nl=None):
    """
    Simple utilisity to relabel a pre-existing label vector
    
    Parameters
    ----------
    label: array of shape(n)
    nl: array of shape(p), where p<= label.max(), optional
        if None, the output is -1*np.ones(n)
    Returns
    -------
    new_label: array of shape (n)
    """
    if label.max()+1<np.size(nl):
        raise ValueError, 'incompatible values for label of nl'
    new_label = -np.ones(np.shape(label))
    if nl!=None:
        aux = np.arange(label.max()+1)
        aux[0:np.size(nl)] = nl
        new_label[label>-1] = aux[label[label>-1]]
    return new_label


#------------------------------------------------------------------
#------------------- main functions ----------------------------------
#------------------------------------------------------------------



def compute_BSA_ipmi(Fbeta, lbeta, coord, dmax, xyz, affine=np.eye(4), 
                     shape=None, thq=0.5, smin=5, ths=0, theta=3.0, g0=1.0,
                     bdensity=0, model="gam_gauss", verbose=0):
    """
    Compute the  Bayesian Structural Activation patterns
    with approach described in IPMI'07 paper

    Parameters
    ----------
    Fbeta :   nipy.neurospin.graph.field.Field instance
          an  describing the spatial relationships
          in the dataset. nbnodes = Fbeta.V
    lbeta: an array of shape (nbnodes, subjects):
           the multi-subject statistical maps
    coord array of shape (nnodes,3):
          spatial coordinates of the nodes
    xyz array of shape (nnodes,3):
        the grid coordinates of the field
    affine=np.eye(4), array of shape(4,4)
         coordinate-defining affine transformation
    shape=None, tuple of length 3 defining the size of the grid
        implicit to the discrete ROI definition 
    thq = 0.5 (float): posterior significance threshold should be in [0,1]
    smin = 5 (int): minimal size of the regions to validate them
    theta = 3.0 (float): first level threshold
    g0 = 1.0 (float): constant values of the uniform density
       over the (compact) volume of interest
    bdensity=0 if bdensity=1, the variable p in ouput
               contains the likelihood of the data under H1 
               on the set of input nodes
    model: string,
           model used to infer the prior p_values
           can be 'gamma_gauss' or 'gauss_mixture'
    verbose=0 : verbosity mode
    
    Results
    -------
    crmap: array of shape (nnodes):
           the resulting group-level labelling of the space
    LR: a instance of sbf.Landmrak_regions that describes the ROIs found
        in inter-subject inference
        If no such thing can be defined LR is set to None
    bf: List of  nipy.neurospin.spatial_models.hroi.Nroi instances
        representing individual ROIs
    p: array of shape (nnodes):
       likelihood of the data under H1 over some sampling grid
    
    Note
    ----
    This is historically the first version,
    but probably not the  most optimal
    It should not be changed for historical reason
    """
    nbsubj = lbeta.shape[1]
    nvox = lbeta.shape[0]

    bf, gf0, sub, gfc = compute_individual_regions(
        Fbeta, lbeta, coord, xyz, affine,  shape,  smin, theta,
        'gam_gauss', verbose)
    
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
    dof = 1000
    prior_precision =  1./(dmax*dmax)*np.ones((1,3))

    if bdensity:
        spatial_coords = coord
    else:
        spatial_coords = gfc

    #p,q =  fc.fdp(gfc, 0.5, g0, g1,dof, prior_precision,
    #              1-gf0, sub, 100, spatial_coords,10,1000)
    p,q =  dpmm(gfc, 0.5, g0, g1, dof, prior_precision,
                  1-gf0, sub, 100, spatial_coords, nis=300)

    # inference
    valid = q>thq

    # remove non-significant regions
    for s in range(nbsubj):
        bfs = bf[s]
        if bfs is not None:
            valids = -np.ones(bfs.V).astype('bool')
            valids[bfs.isleaf()] = valid[sub==s]
            valids = bfs.propagate_upward_and(valids)
            bfs.clean(valids)
            
        if bfs!=None:
            bfs.merge_descending()
            bfs.set_discrete_feature_from_index('position',coord)
            bfs.discrete_to_roi_features('position','cumulated_average')

    # compute probabilitsic correspondences across subjects
    gc = _hierarchical_asso(bf, np.sqrt(2)*dmax)

    if gc == []:
        return crmap,AF,bf,p

    # make hard clusters through clustering
    u, cost = average_link_graph_segment(gc, 0.2, gc.V*1.0/nbsubj)

    q = 0
    for s in range(nbsubj):
        if bf[s]!=None:
            bf[s].set_roi_feature('label', u[q:q+bf[s].k])
            q += bf[s].k
    
    LR, mlabel = sbf.build_LR(bf, ths)
    if LR is not None:
        crmap = LR.map_label(coord, pval=0.95, dmax=dmax)
    
    return crmap, LR, bf, p

#------------------------------------------------------------------
# --------------- dev part ----------------------------------------
# -----------------------------------------------------------------

def bsa_dpmm(Fbeta, bf, gf0, sub, gfc, coord, dmax, thq, ths, g0,verbose=0):
    """
    Estimation of the population level model of activation density using 
    dpmm and inference
    
    Parameters
    ----------
    Fbeta nipy.neurospin.graph.field.Field instance
          an  describing the spatial relationships
          in the dataset. nbnodes = Fbeta.V
    bf list of nipy.neurospin.spatial_models.hroi.Nroi instances
       representing individual ROIs
       let nr be the number of terminal regions across subjects
    gf0, array of shape (nr)
         the mixture-based prior probability 
         that the terminal regions are true positives
    sub, array of shape (nr)
         the subject index associated with the terminal regions
    gfc, array of shape (nr, coord.shape[1])
         the coordinates of the of the terminal regions
    coord: array of shape(sum(nr), dim)
           the coordinates of the regions
    dmax float>0:
         expected cluster std in the common space in units of coord
    thq = 0.5 (float in the [0,1] interval)
        p-value of the prevalence test
    ths=0, float in the rannge [0,nsubj]
        null hypothesis on region prevalence that is rejected during inference
    g0:float,
       constant value of the uniform density over the domain of interest
    verbose=0, verbosity mode

    Returns
    -------
    crmap: array of shape (nnodes):
           the resulting group-level labelling of the space
    LR: a instance of sbf.Landmark_regions that describes the ROIs found
        in inter-subject inference
        If no such thing can be defined LR is set to None
    bf: List of  nipy.neurospin.spatial_models.hroi.Nroi instances
        representing individual ROIs
    p: array of shape (nnodes):
       likelihood of the data under H1 over some sampling grid
    """
    nvox = coord.shape[0]
    n_subj = len(bf)
    
    crmap = -np.ones(nvox, np.int)
    u = []
    LR = None
    p = np.zeros(nvox)
    if len(sub)<1:
        return crmap,LR,bf,p

    sub = np.concatenate(sub).astype(np.int) 
    gfc = np.concatenate(gfc)
    gf0 = np.concatenate(gf0)
    
    # prepare the DPMM
    dim = coord.shape[1]
    g1 = g0
    prior_precision =  1./(dmax*dmax)*np.ones((1,dim))
    dof = 10
    burnin = 100
    nis = 300
    # nis = number of iterations to estimate p
    
    #nii = 100
    ## nii = number of iterations to estimate q
    #p,q =  fc.fdp(gfc, 0.5, g0, g1, dof, prior_precision, 1-gf0,
    #              sub, burnin, coord, nis, nii)
    
    p, q =  dpmm(gfc, 0.5, g0, g1, dof, prior_precision, 1-gf0,
               sub, burnin, coord, nis)
    
    if verbose:
        import matplotlib.pylab as mp
        mp.figure()
        mp.plot(1-gf0,q,'.')
        h1,c1 = mp.histogram((1-gf0),bins=100)
        h2,c2 = mp.histogram(q,bins=100)
        mp.figure()
        mp.bar(c1[:len(h1)],h1,width=0.005)
        mp.bar(c2[:len(h2)]+0.003,h2,width=0.005,color='r')
        print 'Number of candidate regions %i, regions found %i' % (
                    np.size(q), q.sum())
    
    Fbeta.set_field(p)
    _, _, _, label = Fbeta.custom_watershed(0, g0)

    # append some information to the hroi in each subject
    for s in range(n_subj):
        bfs = bf[s]
        if bfs!=None:
            leaves = bfs.isleaf()
            us = -np.ones(bfs.k).astype(np.int)
            lq = np.zeros(bfs.k)
            lq[leaves] = q[sub==s]
            bfs.set_roi_feature('posterior_proba', lq)
            lq = np.zeros(bfs.k)
            lq[leaves] = 1-gf0[sub==s]
            bfs.set_roi_feature('prior_proba',lq)
                   
            #idx = bfs.feature_argmax('activation')
            #midx = [bfs.discrete_features['index'][k][idx[k]]
            #        for k in range(bfs.k)]
            pos = bfs.roi_features['position']
            midx = [np.argmin(np.sum((coord-pos[k])**2,1))  for k in range(bfs.k)]
            j = label[np.array(midx)]
            us[leaves] = j[leaves]

            # when parent regions has similarly labelled children,
            # include it also
            us = bfs.propagate_upward(us)
            bfs.set_roi_feature('label',us)
                        
    # derive the group-level landmarks
    # with a threshold on the number of subjects
    # that are represented in each one 
    LR, nl = infer_LR(bf, thq, ths,verbose=verbose)

    # make a group-level map of the landmark position        
    crmap = _relabel_(label, nl)   
    
    return crmap, LR, bf, p

def dpmm(gfc, alpha, g0, g1, dof, prior_precision, gf1, sub, burnin,
         spatial_coords=None, nis=1000, co_clust=False, verbose=False):
    """
    Apply the dpmm analysis to the data in python
    """
    from nipy.neurospin.clustering.imm import MixedIMM
    dim = gfc.shape[1]
    migmm = MixedIMM(alpha, dim)
    migmm.set_priors(gfc)
    migmm.set_constant_densities(null_dens=g0, prior_dens=g1)
    migmm._prior_dof = dof
    migmm._prior_scale = np.diag(prior_precision[0]/dof)
    migmm._inv_prior_scale = [np.diag(1./(prior_precision[0]/dof))]
    migmm.sample(gfc, null_class_proba=1-gf1, niter=burnin, init=False,
                 kfold=sub)
    if verbose:
        print 'number of components: ', migmm.k

    #sampling
    if co_clust:
        like, pproba, co_clust =  migmm.sample(
            gfc, null_class_proba=1-gf1, niter=nis,
            sampling_points=spatial_coords, kfold=sub, co_clustering=co_clust)

        if verbose:
            print 'number of components: ', migmm.k
        
        return like, 1-pproba, co_clust
    else:
        like, pproba =  migmm.sample(
            gfc, null_class_proba=1-gf1, niter=nis,
            sampling_points=spatial_coords, kfold=sub, co_clustering=co_clust)
    if verbose:
        print 'number of components: ', migmm.k

    return like, 1-pproba


def bsa_dpmm2(Fbeta, bf, gf0, sub, gfc, coord, dmax, thq, ths, g0,verbose):
    """
    Estimation of the population level model of activation density using 
    dpmm and inference
    
    Parameters
    ----------
    Fbeta nipy.neurospin.graph.field.Field instance
          an  describing the spatial relationships
          in the dataset. nbnodes = Fbeta.V
    bf list of nipy.neurospin.spatial_models.hroi.Nroi instances
       representing individual ROIs
       let nr be the number of terminal regions across subjects
    gf0, array of shape (nr)
         the mixture-based prior probability 
         that the terminal regions are true positives
    sub, array of shape (nr)
         the subject index associated with the terminal regions
    gfc, array of shape (nr, coord.shape[1])
         the coordinates of the of the terminal regions
    dmax float>0:
         expected cluster std in the common space in units of coord
    thq = 0.5 (float in the [0,1] interval)
        p-value of the prevalence test
    ths=0, float in the rannge [0,n_subj]
        null hypothesis on region prevalence that is rejected during inference
    g0 = 1.0 (float): constant value of the uniform density
       over the (compact) volume of interest
    verbose=0, verbosity mode

    Returns
    -------
    crmap: array of shape (nnodes):
           the resulting group-level labelling of the space
    LR: a instance of sbf.Landmark_regions that describes the ROIs found
        in inter-subject inference
        If no such thing can be defined LR is set to None
    bf: List of  nipy.neurospin.spatial_models.hroi.Nroi instances
        representing individual ROIs
    Coclust: array of shape (nr,nr):
             co-labelling matrix that gives for each pair of cross_subject regions 
             how likely they are in the same class according to the model
             
    """
    nvox = coord.shape[0]
    n_subj = len(bf)
    
    crmap = -np.ones(nvox, np.int)
    LR = None
    p = np.zeros(nvox)
    if len(sub)<1:
        return crmap,LR,bf,p

    sub = np.concatenate(sub).astype(np.int) 
    gfc = np.concatenate(gfc)
    gf0 = np.concatenate(gf0)
    
    # prepare the DPMM
    g1 = g0
    prior_precision =  1./(dmax*dmax)*np.ones((1,3), np.float)
    dof = 100
    spatial_coords = coord
    burnin = 100
    nis = 300
    # nis = number of iterations to estimate q and co_clust
    nii = 100
    # number of iterations to estimate p
 
    #CoClust, q, p =  fc.fdp2(gfc, 0.5, g0, g1, dof, prior_precision, 1-gf0,
    #               sub, burnin, gfc, nis, nii)
    q, p, CoClust = dpmm(gfc, .5, g0, g1, dof, prior_precision, 1-gf0,
                         sub, burnin, nis=nis, co_clust=True)
    
    #qq = (CoClust>0.5).astype(np.float)
    #cg = fg.WeightedGraph(np.size(q))
    #cg.from_adjacency(qq)
    cg = fg.wgraph_from_coo_matrix(CoClust)
    u = cg.cc()
    u[p<g0] = u.max()+1+np.arange(np.sum(p<g0))

    if verbose:
        cg.show(gfc)
        
    # append some information to the hroi in each subject
    for s in range(n_subj):
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
       
            us[leaves] = u[sub==s]

            # when parent regions has similarly labelled children,
            # include it also
            us = bfs.propagate_upward(us)
            bfs.set_roi_feature('label',us)
                        
    # derive the group-level landmarks
    # with a threshold on the number of subjects
    # that are represented in each one 
    LR,nl = infer_LR(bf,thq,ths,verbose=verbose)

    # make a group-level map of the landmark position
    crmap = -np.ones(nvox)
    # not implemented at the moment
 
    return crmap, LR, bf, CoClust
    
        


def compute_BSA_simple(Fbeta, lbeta, coord, dmax, xyz, affine=np.eye(4), 
                              shape=None,
                       thq=0.5, smin=5, ths=0, theta=3.0, g0=1.0,
                       verbose=0):
    """
    Compute the  Bayesian Structural Activation paterns - simplified version  

    Parameters
    ----------
    Fbeta :  nipy.neurospin.graph.field.Field instance
          an  describing the spatial relationships
          in the dataset. nbnodes = Fbeta.V
    lbeta: an array of shape (nbnodes, subjects):
           the multi-subject statistical maps
    coord array of shape (nnodes,3):
          spatial coordinates of the nodes
    dmax float>0:
         expected cluster std in the common space in units of coord
    xyz array of shape (nnodes,3):
        the grid coordinates of the field
    affine=np.eye(4), array of shape(4,4)
         coordinate-defining affine transformation
    shape=None, tuple of length 3 defining the size of the grid
        implicit to the discrete ROI definition      
    thq = 0.5 (float):
        posterior significance threshold 
        should be in the [0,1] interval
    smin = 5 (int): minimal size of the regions to validate them
    theta = 3.0 (float): first level threshold
    g0 = 1.0 (float): constant values of the uniform density
       over the (compact) volume of interest
    verbose=0: verbosity mode

    Results
    -------
    crmap: array of shape (nnodes):
           the resulting group-level labelling of the space
    LR: a instance of sbf.Landmark_regions that describes the ROIs found
        in inter-subject inference
        If no such thing can be defined LR is set to None
    bf: List of  nipy.neurospin.spatial_models.hroi.Nroi instances
        representing individual ROIs
    p: array of shape (nnodes):
       likelihood of the data under H1 over some sampling grid

    Note
    ----
    In that case, the DPMM is used to derive a spatial density of
    significant local maxima in the volume. Each terminal (leaf)
    region which is a posteriori significant enough is assigned to the
    nearest mode of this distribution

    fixme
    -----
    The number of itertions should become a parameter
    """
   
    bf, gf0, sub, gfc = compute_individual_regions(
        Fbeta, lbeta, coord, xyz, affine, shape, smin, theta,
        'gauss_mixture', verbose)
    
    crmap, LR, bf, p = bsa_dpmm(Fbeta, bf, gf0, sub, gfc, coord, dmax, thq,
                                ths, g0, verbose)
    
    return crmap, LR, bf, p

def compute_BSA_simple_quick(Fbeta, lbeta, coord, dmax, xyz, affine=np.eye(4), 
                        shape=None, thq=0.5, smin=5, ths=0, theta=3.0, g0=1.0,
                       verbose=0):
    """
    Idem compute_BSA_simple, but this one does not estimate the full density
    (on small datasets, it can be much faster)  

    Parameters
    ----------
    Fbeta :  nipy.neurospin.graph.field.Field instance
          an  describing the spatial relationships
          in the dataset. nbnodes = Fbeta.V
    lbeta: an array of shape (nbnodes, subjects):
           the multi-subject statistical maps
    coord array of shape (nnodes,3):
          spatial coordinates of the nodes
    dmax float>0:
         expected cluster std in the common space in units of coord
    xyz array of shape (nnodes,3):
        the grid coordinates of the field
    affine=np.eye(4), array of shape(4,4)
         coordinate-defining affine transformation
    shape=None, tuple of length 3 defining the size of the grid
        implicit to the discrete ROI definition      
    thq = 0.5 (float):
        posterior significance threshold 
        should be in the [0,1] interval
    smin = 5 (int): minimal size of the regions to validate them
    theta = 3.0 (float): first level threshold
    g0 = 1.0 (float): constant values of the uniform density
       over the (compact) volume of interest
    verbose=0: verbosity mode

    Results
    -------
    crmap: array of shape (nnodes):
           the resulting group-level labelling of the space
    LR: a instance of sbf.Landmark_regions that describes the ROIs found
        in inter-subject inference
        If no such thing can be defined LR is set to None
    bf: List of  nipy.neurospin.spatial_models.hroi.Nroi instances
        representing individual ROIs
    coclust: array of shape (nr,nr):
        co-labelling matrix that gives for each pair of cross_subject regions 
        how likely they are in the same class according to the model
    """
    
    bf, gf0, sub, gfc = compute_individual_regions(
        Fbeta, lbeta, coord, xyz, affine,  shape,  smin, theta,
        'gauss_mixture', verbose)
    crmap, LR, bf, coclust = bsa_dpmm2(
        Fbeta, bf, gf0, sub, gfc, coord, dmax, thq, ths, g0, verbose)

    return crmap, LR, bf, coclust

def compute_individual_regions(Fbeta, lbeta, coord, xyz, affine=np.eye(4),
                               shape=None, smin=5, theta=3.0,
                               model='gauss_mixture', verbose=0, reshuffle=0):
    """
    Compute the  Bayesian Structural Activation paterns -
    with statistical validation

    Parameters
    ----------
    Fbeta:  nipy.neurospin.graph.field.Field instance
          an  describing the spatial relationships
          in the dataset. nbnodes = Fbeta.V
    lbeta: an array of shape (nbnodes, subjects)
           the multi-subject statistical maps
    coord: array of shape (nnodes, 3),
          spatial coordinates of the nodes
    xyz: array of shape (nnodes,3)
        the grid coordinates of the field
    affine=np.eye(4), array of shape(4, 4)
         coordinate-defining affine transformation
    shape=None, tuple of length 3 defining the size of the grid
        implicit to the discrete ROI definition      
    smin: int, optional
          minimal size of the regions to validate them
    theta: float, optional
           first level threshold
    model: string, optional,
           model that is used to provide priori significance
           can be 'gauss_mixture', 'gam_gauss' or 'emp_null'
    verbose=0: verbosity mode
    reshuffle=0: if nonzero, reshuffle the positions; this affects bf and gfc
    
    Returns
    -------
    bf list of nipy.neurospin.spatial_models.hroi.Nroi instances
       representing individual ROIs
       let nr be the number of terminal regions across subjects
    gf0, array of shape (nr)
         the mixture-based prior probability 
         that the terminal regions are true positives
    sub, array of shape (nr)
         the subject index associated with the terminal regions
    gfc, array of shape (nr, coord.shape[1])
         the coordinates of the of the terminal regions
    """
    bf = []
    gfc = []
    gf0 = []
    sub = []
    n_subj = lbeta.shape[1]
    nvox = lbeta.shape[0]

    for s in range(n_subj):
        # description in terms of blobs
        beta = np.reshape(lbeta[:,s], (nvox,1))
        Fbeta.set_field(beta)
        nroi = hroi.NROI_from_field(Fbeta, affine, shape, xyz, refdim=0,
                                    th=theta, smin=smin)
              
        if nroi!=None:
            nroi.set_discrete_feature_from_index('activation',beta)
            bfm = nroi.discrete_to_roi_features('activation','average')
            bfm = bfm[nroi.isleaf()]

            # get the regions position
            if reshuffle:
                nroi = nroi.reduce_to_leaves()
                ## randomize the positions
                ## by taking any local maximum of the image
                temp = np.argsort(np.random.rand(nvox))[:nroi.k]
                bfc = coord[temp]
                nroi.parents = np.arange(nroi.k)
                nroi.set_roi_feature('position',bfc)
            else:
                nroi.set_discrete_feature_from_index('position',coord)
                bfc = nroi.discrete_to_roi_features('position','average')
                bfc = bfc[nroi.isleaf()]
            gfc.append(bfc)
            
            # compute the prior proba of being null
            beta = np.squeeze(beta)

            # first remove 0's to avoid artefact in histogram fit
            beta = beta[beta!=0]

            if model=='gauss_mixture':
                alpha = 0.01
                prior_strength = 100
                fixed_scale = True
                bfp = en.three_classes_GMM_fit(
                    beta, bfm, alpha, prior_strength,verbose, fixed_scale)
                bf0 = bfp[:,1]
            elif model== 'emp_null':
                enn = en.ENN(beta)
                enn.learn()
                bf0 = np.reshape(enn.fdr(bfm),np.size(bf0))
            elif model=='gam_gauss':
                bfp  = en.Gamma_Gaussian_fit(beta, bfm, verbose)
                bf0 = bfp[:,1]
            else: raise ValueError, 'Unknown model'
            
            gf0.append(bf0)
            sub.append(s*np.ones(np.size(bfm)))

            nroi.set_roi_feature('label',np.arange(nroi.k))
        bf.append(nroi)    
    return bf, gf0, sub, gfc


def compute_BSA_loo(Fbeta, lbeta, coord, dmax, xyz, affine=np.eye(4), 
                    shape=None,  thq=0.5, smin=5, ths=0, theta=3.0, g0=1.0,
                    verbose=0):
    """
    Compute the  Bayesian Structural Activation paterns -
    with statistical validation

    Parameters
    ----------
    Fbeta :  nipy.neurospin.graph.field.Field instance
          an  describing the spatial relationships
          in the dataset. nbnodes = Fbeta.V
    lbeta: an array of shape (nbnodes, subjects):
           the multi-subject statistical maps
    coord array of shape (nnodes,3):
          spatial coordinates of the nodes
    dmax float>0:
         expected cluster std in the common space in units of coord
    xyz array of shape (nnodes,3):
        the grid coordinates of the field
    affine=np.eye(4), array of shape(4,4)
         coordinate-defining affine transformation
    shape=None, tuple of length 3 defining the size of the grid
        implicit to the discrete ROI definition      
    thq = 0.5 (float):
        posterior significance threshold 
        should be in the [0,1] interval
    smin = 5 (int): minimal size of the regions to validate them
    theta = 3.0 (float): first level threshold
    g0 = 1.0 (float): constant values of the uniform density
       over the (compact) volume of interest
    verbose=0: verbosity mode

    Results
    -------
    mll, float, the average cross-validated log-likelihood across subjects
    ml0, float the log-likelihood of the model under a global null hypothesis

    """
    n_subj = lbeta.shape[1]
    nvox = lbeta.shape[0]
    bf, gf0, sub, gfc = compute_individual_regions(
        Fbeta, lbeta, coord, xyz, affine,  shape,  smin, theta,
        'gauss_mixture', verbose)
    
    crmap = -np.ones(nvox, np.int)
    LR = None
    p = np.zeros(nvox)
    if len(sub)<1:
        return np.log(g0), np.log(g0)

    sub = np.concatenate(sub).astype(np.int) 
    gfc = np.concatenate(gfc)
    gf0 = np.concatenate(gf0)
    
    # prepare the DPMM
    g1 = g0
    prior_precision =  1./(dmax*dmax)*np.ones((1,3), np.float)
    dof = 100
    burnin = 100
    nis = 300
    nii = 100
    ll1 = []
    ll0 = []
    ll2 = []
    
    for s in range(n_subj):
        # 
        if np.sum(sub==s)>0:
            spatial_coords = gfc[sub==s]
            #p, q =  fc.fdp(
            #    gfc[sub!=s], 0.5, g0, g1, dof, prior_precision,
            #    1-gf0[sub!=s], sub[sub!=s], burnin, spatial_coords, nis, nii)
            p, q =  dpmm(
                gfc[sub!=s], 0.5, g0, g1, dof, prior_precision,
                1-gf0[sub!=s], sub[sub!=s], burnin, spatial_coords, nis)
            pp = gf0[sub==s]*g0 + p*(1-gf0[sub==s])
            ll2.append(np.mean(np.log(pp)))
            ll1.append(np.mean(np.log(p)))
            ll0.append(np.mean(np.log(g0)))

    ml0 = np.mean(np.array(ll0))
    ml1 = np.mean(np.array(ll1))
    mll = np.mean(np.array(ll2))
    if verbose: 
       print 'average cross-validated log likelihood'
       print 'null model: ', ml0,' alternative model: ', mll

    return mll, ml0
