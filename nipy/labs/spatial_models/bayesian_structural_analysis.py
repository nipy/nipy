# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
The main routine of this package that aims at performing the
extraction of ROIs from multisubject dataset using the localization
and activation strength of extracted regions.

This has been published in:
- Thirion et al. High level group analysis of FMRI data based on
Dirichlet process mixture models, IPMI 2007
- Thirion et al.
Accurate Definition of Brain Regions Position Through the
Functional Landmark Approach, MICCAI 2010

Author : Bertrand Thirion, 2006-2011
"""

import numpy as np
import scipy.stats as st

from .structural_bfls import build_LR
from nipy.algorithms.graph import wgraph_from_coo_matrix
from ...algorithms.statistics.empirical_pvalue import \
    NormalEmpiricalNull, three_classes_GMM_fit, Gamma_Gaussian_fit
from .hroi import HROI_as_discrete_domain_blobs

####################################################################
# Ancillary functions
####################################################################


def _relabel_(label, nl=None):
    """ Simple utilisity to relabel a pre-existing label vector

    Parameters
    ----------
    label: array of shape(n)
    nl: array of shape(p), where p<= label.max(), optional
        if None, the output is -1*np.ones(n)
    Returns
    -------
    new_label: array of shape (n)
    """
    if label.max() + 1 < np.size(nl):
        raise ValueError('incompatible values for label of nl')
    new_label = - np.ones(np.shape(label))
    if nl is not None:
        aux = np.arange(label.max() + 1)
        aux[0: np.size(nl)] = nl
        new_label[label > - 1] = aux[label[label > - 1]]
    return new_label


def signal_to_pproba(test, learn=None, method='prior', alpha=0.01, verbose=0):
    """Convert a set of z-values to posterior probabilities of not being active

    Parameters
    ----------
    test: array pf shape(n_samples),
           data that is assessed
    learn: array pf shape(n_samples), optional
           data to learn a mixture model
    method: string, optional, to be chosen within
            ['gauss_mixture', 'emp_null', 'gam_gauss', 'prior']
    alpha: float in the [0,1], optional,
           parameter that yields the prior probability that a region is active
           should be chosen close to 0
    """
    if method == 'gauss_mixture':
        prior_strength = 100
        fixed_scale = True
        bfp = three_classes_GMM_fit(
            learn, test, alpha, prior_strength, verbose, fixed_scale)
        bf0 = bfp[:, 1]
    elif method == 'emp_null':
        enn = NormalEmpiricalNull(learn)
        enn.learn()
        bf0 = np.reshape(enn.fdr(test), np.size(test))
    elif method == 'gam_gauss':
        bfp = Gamma_Gaussian_fit(learn, test, verbose)
        bf0 = bfp[:, 1]
    elif method == 'prior':
        y0 = st.norm.pdf(test)
        shape_, scale_ = 3., 2.
        y1 = st.gamma.pdf(test, shape_, scale=scale_)
        bf0 = np.ravel((1 - alpha) * y0 / (alpha * y1 + (1 - alpha) * y0))
    else:
        raise ValueError('Unknown method')
    return bf0


def compute_individual_regions(domain, lbeta, smin=5, theta=3.0,
                               method='gauss_mixture', verbose=0, reshuffle=0,
                               criterion='size', assign_val='weighted_mean'):
    """ Compute the individual regions that are real activation candidates

    Parameters
    ----------
    dom : StructuredDomain instance,
          generic descriptor of the space domain
    lbeta: an array of shape (nbnodes, subjects)
           the multi-subject statistical maps
    smin: int, optional
          minimal size of the regions to validate them
    theta: float, optional
           first level threshold
    method: string, optional,
           method that is used to provide priori significance
           can be 'prior', 'gauss_mixture', 'gam_gauss' or 'emp_null'
    verbose: verbosity mode, optional
    reshuffle: bool, otpional,
               if nonzero, reshuffle the positions; this affects bf and gfc
    criterion: string, optional,
               'size' or 'volume', thresholdding criterion
    assign_val: string, optional,
                to  be chosen in 'weighted mean', 'mean', 'min', 'max'
                heuristic to assigna  blob-level signal

    Returns
    -------
    bf list of nipy.labs.spatial_models.hroi.Nroi instances
       representing individual ROIs
       let nr be the number of terminal regions across subjects
    gf0, array of shape (nr)
         the mixture-based prior probability
         that the terminal regions are false positives
    sub, array of shape (nr)
         the subject index associated with the terminal regions
    gfc, array of shape (nr, coord.shape[1])
         the coordinates of the of the terminal regions

    Fixme
    -----
    Should allow for subject specific domains
    """
    bf = []
    gfc = []
    gf0 = []
    sub = []
    n_subj = lbeta.shape[1]
    nvox = lbeta.shape[0]

    for s in range(n_subj):
        # description in terms of blobs
        beta = np.reshape(lbeta[:, s], (nvox, 1))
        nroi = HROI_as_discrete_domain_blobs(
            domain, beta, threshold=theta, smin=smin)
        
        if nroi is not None and nroi.k > 0:
            bfm = nroi.representative_feature('signal', 'weighted mean')
            bfm = bfm[[nroi.select_id(id) for id in nroi.get_leaves_id()]]
            # get the regions position
            if reshuffle:
                nroi.reduce_to_leaves()
                ## randomize the positions
                ## by taking any local maximum of the image
                temp = np.argsort(np.random.rand(nvox))[:nroi.k]
                bfc = domain.coord[temp]
            else:
                mean_pos = np.asarray(
                    [np.mean(coords, 0) for coords in nroi.get_coord()])
                nroi.set_roi_feature('position', mean_pos)
                bfc = mean_pos[[nroi.select_id(id)
                                for id in nroi.get_leaves_id()]]
            gfc.append(bfc)

            # compute the prior proba of being null
            learn = np.squeeze(beta[beta != 0])
            bf0 = signal_to_pproba(bfm, learn, method)
            gf0.append(bf0)
            sub.append(s * np.ones(np.size(bfm)))
            
        bf.append(nroi)
    return bf, gf0, sub, gfc


def dpmm(gfc, alpha, g0, g1, dof, prior_precision, gf1, sub, burnin,
         spatial_coords=None, nis=1000, co_clust=False, verbose=False):
    """Apply the dpmm analysis to compute clusters from regions coordinates
    """
    from nipy.algorithms.clustering.imm import MixedIMM

    dim = gfc.shape[1]
    migmm = MixedIMM(alpha, dim)
    migmm.set_priors(gfc)
    migmm.set_constant_densities(null_dens=g0, prior_dens=g1)
    migmm._prior_dof = dof
    migmm._prior_scale = np.diag(prior_precision[0] / dof)
    migmm._inv_prior_scale_ = [np.diag(dof * 1. / (prior_precision[0]))]
    migmm.sample(gfc, null_class_proba=1 - gf1, niter=burnin, init=False,
                 kfold=sub)
    if verbose:
        print 'number of components: ', migmm.k

    # sampling
    if co_clust:
        like, pproba, co_clust = migmm.sample(
            gfc, null_class_proba=1 - gf1, niter=nis,
            sampling_points=spatial_coords, kfold=sub, co_clustering=co_clust)
        if verbose:
            print 'number of components: ', migmm.k

        return like, 1 - pproba, co_clust
    else:
        like, pproba = migmm.sample(
            gfc, null_class_proba=1 - gf1, niter=nis,
            sampling_points=spatial_coords, kfold=sub, co_clustering=co_clust)
    if verbose:
        print 'number of components: ', migmm.k

    return like, 1 - pproba


def bsa_dpmm(bf, gf0, sub, gfc, dmax, thq, ths, verbose=0):
    """ Estimation of the population level model of activation density using
    dpmm and inference

    Parameters
    ----------
    bf list of nipy.labs.spatial_models.hroi.HierarchicalROI instances
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
    ths=0, float in the rannge [0,nsubj]
        null hypothesis on region prevalence that is rejected during inference
    verbose=0, verbosity mode

    Returns
    -------
    crmap: array of shape (nnodes):
           the resulting group-level labelling of the space
    LR: a instance of sbf.LandmarkRegions that describes the ROIs found
        in inter-subject inference
        If no such thing can be defined LR is set to None
    bf: List of  nipy.labs.spatial_models.hroi.Nroi instances
        representing individual ROIs
    p: array of shape (nnodes):
       likelihood of the data under H1 over some sampling grid
    """
    from nipy.algorithms.graph.field import field_from_coo_matrix_and_data
    dom = bf[0].domain
    n_subj = len(bf)

    crmap = - np.ones(dom.size, np.int)
    LR = None
    p = np.zeros(dom.size)
    if len(sub) < 1:
        return crmap, LR, bf, p

    sub = np.concatenate(sub).astype(np.int)
    gfc = np.concatenate(gfc)
    gf0 = np.concatenate(gf0)

    g0 = 1. / dom.local_volume.sum()

    # prepare the DPMM
    dim = dom.em_dim
    g1 = g0
    prior_precision = 1. / (dmax * dmax) * np.ones((1, dim))
    dof = 10
    burnin = 100
    nis = 1000
    # nis = number of iterations to estimate p
    p, q = dpmm(gfc, 0.5, g0, g1, dof, prior_precision, 1 - gf0,
               sub, burnin, dom.coord, nis)

    if verbose:
        h1, c1 = np.histogram((1 - gf0), bins=100)
        h2, c2 = np.histogram(q, bins=100)
        try:
            import matplotlib.pylab as pl
            pl.figure()
            pl.plot(1 - gf0, q, '.')
            pl.figure()
            pl.bar(c1[:len(h1)], h1, width=0.005)
            pl.bar(c2[:len(h2)] + 0.003, h2, width=0.005, color='r')
        except ImportError:
            pass
        print 'Number of candidate regions %i, regions found %i' % (
            np.size(q), q.sum())

    Fbeta = field_from_coo_matrix_and_data(dom.topology, p)
    _, label = Fbeta.custom_watershed(0, g0)

    # append some information to the hroi in each subject
    for s in range(n_subj):
        bfs = bf[s]
        if bfs.k > 0:
            leaves_pos = [bfs.select_id(k) for k in bfs.get_leaves_id()]
            us = - np.ones(bfs.k).astype(np.int)

            # set posterior proba
            lq = np.zeros(bfs.k)
            lq[leaves_pos] = q[sub == s]
            bfs.set_roi_feature('posterior_proba', lq)

            # set prior proba
            lq = np.zeros(bfs.k)
            lq[leaves_pos] = 1 - gf0[sub == s]
            bfs.set_roi_feature('prior_proba', lq)

            pos = np.asarray(
                [np.mean(coords, 0) for coords in bfs.get_coord()])
            midx = [np.argmin(np.sum((dom.coord - pos[k]) ** 2, 1))
                    for k in range(bfs.k)]
            j = label[np.array(midx)]
            us[leaves_pos] = j[leaves_pos]
            
            # when parent regions has similarly labelled children,
            # include it also
            us = bfs.make_forest().propagate_upward(us)
            bfs.set_roi_feature('label', us)

    # derive the group-level landmarks
    # with a threshold on the number of subjects
    # that are represented in each one
    LR, nl = build_LR(bf, thq, ths, dmax, verbose=verbose)

    # make a group-level map of the landmark position
    crmap = _relabel_(label, nl)
    return crmap, LR, bf, p


def bsa_dpmm2(bf, gf0, sub, gfc, dmax, thq, ths, verbose):
    """ Estimation of the population level model of activation density using
    dpmm and inference

    Parameters
    ----------
    bf list of nipy.labs.spatial_models.hroi.HierarchicalROI instances
       representing individual ROIs
       let nr be the number of terminal regions across subjects
    gf0, array of shape (nr)
         the mixture-based prior probability
         that the terminal regions are false positives
    sub, array of shape (nr)
         the subject index associated with the terminal regions
    gfc, array of shape (nr, coord.shape[1])
         the coordinates of the of the terminal regions
    dmax float>0:
         expected cluster std in the common space in units of coord
    thq = 0.5 (float in the [0,1] interval)
        p-value of the prevalence test
    ths=0, float in the rannge [0,nsubj]
        null hypothesis on region prevalence that is rejected during inference
    verbose=0, verbosity mode

    Returns
    -------
    crmap: array of shape (nnodes):
           the resulting group-level labelling of the space
    LR: a instance of sbf.LandmarkRegions that describes the ROIs found
        in inter-subject inference
        If no such thing can be defined LR is set to None
    bf: List of  nipy.labs.spatial_models.hroi.Nroi instances
        representing individual ROIs
    Coclust: array of shape (nr,nr):
             co-labelling matrix that gives for each pair of inputs
             how likely they are in the same class according to the model
    """
    dom = bf[0].domain
    n_subj = len(bf)
    crmap = - np.ones(dom.size, np.int)
    LR = None
    p = np.zeros(dom.size)
    if len(sub) < 1:
        return crmap, LR, bf, p

    sub = np.concatenate(sub).astype(np.int)
    gfc = np.concatenate(gfc)
    gf0 = np.concatenate(gf0)

    # prepare the DPMM
    g0 = 1. / (np.sum(dom.local_volume))
    g1 = g0
    prior_precision = 1. / (dmax * dmax) * np.ones((1, dom.em_dim), np.float)
    dof = 10
    burnin = 100
    nis = 300

    q, p, CoClust = dpmm(gfc, .5, g0, g1, dof, prior_precision, 1 - gf0,
                         sub, burnin, nis=nis, co_clust=True)

    cg = wgraph_from_coo_matrix(CoClust)
    if cg.E > 0:
        cg.remove_edges(cg.weights > .5)
    u = cg.cc()
    u[p < g0] = u.max() + 1 + np.arange(np.sum(p < g0))

    if verbose:
        cg.show(gfc)

    # append some information to the hroi in each subject
    for s in range(n_subj):
        bfs = bf[s]
        if bfs is not None:
            leaves = np.asarray(
                [bfs.select_id(id) for id in bfs.get_leaves_id()])
            us = - np.ones(bfs.k).astype(np.int)
            lq = np.zeros(bfs.k)
            lq[leaves] = q[sub == s]
            bfs.set_roi_feature('posterior_proba', lq)
            lq = np.zeros(bfs.k)
            lq[leaves] = 1 - gf0[sub == s]
            bfs.set_roi_feature('prior_proba', lq)
            us[leaves] = u[sub == s]

            # when parent regions has similarly labelled children,
            # include it also
            us = bfs.make_forest().propagate_upward(us)
            bfs.set_roi_feature('label', us)

    # derive the group-level landmarks
    # with a threshold on the number of subjects
    # that are represented in each one
    LR, nl = build_LR(bf, thq, ths, dmax, verbose=verbose)

    # make a group-level map of the landmark position
    crmap = - np.ones(dom.size)
    # not implemented at the moment

    return crmap, LR, bf, CoClust

###########################################################################
# Main functions
###########################################################################


def compute_BSA_simple(dom, lbeta, dmax, thq=0.5, smin=5, ths=0, theta=3.0,
                    method='prior', verbose=0):
    """ Compute the  Bayesian Structural Activation paterns
    simplified version

    Parameters
    ----------
    dom : StructuredDomain instance,
          Description of the spatial context of the data
    lbeta: an array of shape (nbnodes, subjects):
           the multi-subject statistical maps
    dmax float>0:
         expected cluster std in the common space in units of coord
    thq = 0.5 (float):
        posterior significance threshold
        should be in the [0,1] interval
    smin = 5 (int): minimal size of the regions to validate them
    theta = 3.0 (float): first level threshold
    method: string, optional,
            the method used to assess the prior significance of the regions
    verbose=0: verbosity mode

    Returns
    -------
    crmap: array of shape (nnodes):
           the resulting group-level labelling of the space
    LR: a instance of sbf.LandmarkRegions that describes the ROIs found
        in inter-subject inference
        If no such thing can be defined LR is set to None
    bf: List of  nipy.labs.spatial_models.hroi.Nroi instances
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
        dom, lbeta, smin, theta, 'prior', verbose)

    crmap, LR, bf, p = bsa_dpmm(bf, gf0, sub, gfc, dmax, thq, ths, verbose)
    return crmap, LR, bf, p


def compute_BSA_quick(dom, lbeta, dmax, thq=0.5, smin=5, ths=0, theta=3.0,
                      verbose=0):
    """Idem compute_BSA_simple, but this one does not estimate the full density
    (on small datasets, it can be much faster)

    Parameters
    ----------
    dom : StructuredDomain instance,
          Description of the spatial context of the data
    lbeta: an array of shape (nbnodes, subjects):
           the multi-subject statistical maps
    dmax float>0:
         expected cluster std in the common space in units of coord
    thq = 0.5 (float):
        posterior significance threshold
        should be in the [0,1] interval
    smin = 5 (int): minimal size of the regions to validate them
    theta = 3.0 (float): first level threshold
    method: string, optional,
            the method used to assess the prior significance of the regions
    verbose=0: verbosity mode

    Returns
    -------
    crmap: array of shape (nnodes):
           the resulting group-level labelling of the space
    LR: a instance of sbf.LandmarkRegions that describes the ROIs found
        in inter-subject inference
        If no such thing can be defined LR is set to None
    bf: List of  nipy.labs.spatial_models.hroi.Nroi instances
        representing individual ROIs
    coclust: array of shape (nr, nr):
        co-labelling matrix that gives for each pair of cross_subject regions
        how likely they are in the same class according to the model
    """
    bf, gf0, sub, gfc = compute_individual_regions(
        dom, lbeta, smin, theta, 'prior', verbose)
    crmap, LR, bf, co_clust = bsa_dpmm2(
        bf, gf0, sub, gfc, dmax, thq, ths, verbose)
    return crmap, LR, bf, co_clust


def compute_BSA_loo(dom, lbeta, dmax, thq=0.5, smin=5, ths=0, theta=3.0,
                    verbose=0):
    """ Compute the  Bayesian Structural Activation paterns -
    with statistical validation

    Parameters
    ----------
    dom: StructuredDomain instance,
         Description of the spatial context of the data
    lbeta: an array of shape (nbnodes, subjects):
           the multi-subject statistical maps
    dmax float>0:
         expected cluster std in the common space in units of coord
    thq = 0.5 (float):
        posterior significance threshold
        should be in the [0,1] interval
    smin = 5 (int): minimal size of the regions to validate them
    theta = 3.0 (float): first level threshold
    method: string, optional,
            the method used to assess the prior significance of the regions
    verbose=0: verbosity mode

    Results
    -------
    mll, float, the average cross-validated log-likelihood across subjects
    ml0, float the log-likelihood of the model under a global null hypothesis
    """
    n_subj = lbeta.shape[1]
    nvox = dom.size
    bf, gf0, sub, gfc = compute_individual_regions(
        dom, lbeta, smin, theta, 'gauss_mixture', verbose)

    p = np.zeros(nvox)
    g0 = 1. / (np.sum(dom.local_volume))
    if len(sub) < 1:
        return np.log(g0), np.log(g0)

    sub = np.concatenate(sub).astype(np.int)
    gfc = np.concatenate(gfc)
    gf0 = np.concatenate(gf0)

    # prepare the DPMM
    g1 = g0
    dim = dom.em_dim
    prior_precision = 1. / (dmax * dmax) * np.ones((1, dim), np.float)
    dof = 10
    burnin = 100
    nis = 300
    ll0 = []
    ll2 = []

    for s in range(n_subj):
        if np.sum(sub == s) > 0:
            spatial_coords = gfc[sub == s]
            p, q = dpmm(
                gfc[sub != s], 0.5, g0, g1, dof, prior_precision,
                1 - gf0[sub != s], sub[sub != s], burnin, spatial_coords, nis)
            pp = gf0[sub == s] * g0 + p * (1 - gf0[sub == s])
            ll2.append(np.mean(np.log(pp)))
            ll0.append(np.mean(np.log(g0)))

    ml0 = np.mean(np.array(ll0))
    mll = np.mean(np.array(ll2))
    if verbose:
        print 'average cross-validated log likelihood'
        print 'null model: ', ml0, ' alternative model: ', mll

    return mll, ml0
