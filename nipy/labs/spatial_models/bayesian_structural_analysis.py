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

Author : Bertrand Thirion, 2006-2013
"""

import numpy as np
import scipy.stats as st

from .structural_bfls import build_landmarks
from nipy.algorithms.graph import wgraph_from_coo_matrix
from ...algorithms.statistics.empirical_pvalue import \
    NormalEmpiricalNull, three_classes_GMM_fit, gamma_gaussian_fit
from .hroi import HROI_as_discrete_domain_blobs

####################################################################
# Ancillary functions
####################################################################


def _relabel(label, new_values=None):
    """ Utility to relabel a pre-existing label vector 
    to the values provided in new_values.

    Parameters
    ----------
    label: array of shape(n),
           the inut labels
    new_values: array of shape(p), where p<= label.max(), optional
        if None, the output is -1 * np.ones(n)
        the new values to be given to the labels

    Returns
    -------
    new_label: array of shape (n)
    """
    if label.max() + 1 < np.size(new_values):
        raise ValueError('incompatible values for label of new_values')
    new_label = - np.ones_like(label)
    if new_values is not None:
        aux = np.arange(label.max() + 1)
        aux[0: np.size(new_values)] = new_values
        new_label[label > - 1] = aux[label[label > - 1]]
    return new_label


def _signal_to_pproba(test, learn=None, method='prior', alpha=0.01, verbose=0):
    """Convert a set of statistics to posterior probabilities of  being
    generated under H0

    Parameters
    ----------
    test: array pf shape(n_samples),
           data that is assessed
    learn: array pf shape(n_samples), optional
           data to learn a mixture model
           defaults to learn
    method: string, one of ['gauss_mixture', 'emp_null', 'gam_gauss', 'prior']
            optional
           'gauss_mixture' A Gaussian Mixture Model is used
           'emp_null' a null mode is fitted to test
           'gam_gauss' a Gamma-Gaussian mixture is used
           'prior' a hard-coded function is used
    alpha: float in the [0,1] range, optional,
           parameter that yields the prior probability that a region is active
           should be chosen close to 0

    Returns 
    -------
    posterior_null: array of shape(n_samples)
                    an estimation of the probability that the observation 
                    is generated under the null
    """
    if method == 'gauss_mixture':
        prior_strength = 100
        fixed_scale = True
        posterior_probas = three_classes_GMM_fit(
            learn, test, alpha, prior_strength, verbose, fixed_scale)
        posterior_null = posterior_probas[:, 1]
    elif method == 'emp_null':
        enn = NormalEmpiricalNull(learn)
        enn.learn()
        posterior_null = np.reshape(enn.fdr(test), np.size(test))
    elif method == 'gam_gauss':
        posterior_probas = gamma_gaussian_fit(learn, test, verbose)
        posterior_null = posterior_probas[:, 1]
    elif method == 'prior':
        y0 = st.norm.pdf(test)
        shape_, scale_ = 3., 2.
        y1 = st.gamma.pdf(test, shape_, scale=scale_)
        posterior_null = np.ravel(
            (1 - alpha) * y0 / (alpha * y1 + (1 - alpha) * y0))
    else:
        raise ValueError('Unknown method')
    return posterior_null


def _compute_individual_regions(domain, stats, threshold=3.0, smin=5,
                               method='gauss_mixture', verbose=0):
    """ Compute the individual regions that are real activation candidates

    Parameters
    ----------
    domain : StructuredDomain instance,
          generic descriptor of the space domain
    stats: an array of shape (n_voxels, n_subjects)
           the multi-subject statistical maps
    threshold: float, optional
           first level threshold
    smin: int, optional
          minimal size of the regions to validate them
    method: string, one of ['prior', 'gauss_mixture', 'gam_gauss', 'emp_null']
            optional,
            method that is used to estimate prior significance
    verbose: verbosity mode, optional

    Returns
    -------
    hrois: list of nipy.labs.spatial_models.hroi.HierrachicalROI instances
            that represent individual ROIs
            let nr be the number of terminal regions across subjects
    prior_h0, array of shape (nr)
         the mixture-based prior probability
         that the terminal regions are false positives
    subjects, array of shape (nr)
         the subject index associated with the terminal regions
    coords, array of shape (nr, coord.shape[1])
         the coordinates of the of the terminal regions

    Fixme
    -----
    Should allow for subject specific domains
    """
    hrois = []
    coords = []
    prior_h0 = []
    subjects = []
    n_subjects = stats.shape[1]
    nvox = stats.shape[0]

    for subject in range(n_subjects):
        # description in terms of blobs
        stats_ = np.reshape(stats[:, subject], (nvox, 1))
        hroi = HROI_as_discrete_domain_blobs(
            domain, stats_, threshold=threshold, smin=smin)
        
        if hroi is not None and hroi.k > 0:
            # get the leave regions (i.e. the local maxima)
            leaves = [hroi.select_id(id) for id in hroi.get_leaves_id()]

            # get the region mean statistical value
            mean_val = hroi.representative_feature('signal', 'weighted mean')
            mean_val = mean_val[leaves]

            # get the regions position
            mean_pos = np.asarray(
                [np.mean(coord, 0) for coord in hroi.get_coord()])
            hroi.set_roi_feature('position', mean_pos)
            coords.append(mean_pos[leaves])

            # compute the prior proba of being null
            learning_set = np.squeeze(stats_[stats_ != 0])
            prior_h0.append(_signal_to_pproba(mean_val, learning_set, method))
            subjects.append(subject * np.ones(mean_val.size).astype(np.int))
            
        hrois.append(hroi)
    prior_h0 = np.concatenate(prior_h0)
    subjects = np.concatenate(subjects)
    coords = np.concatenate(coords)
    return hrois, prior_h0, subjects, coords


def _dpmm(coords, alpha, null_density, dof, prior_precision, prior_h0, 
          subjects, sampling_coords=None, n_iter=1000, burnin=100, 
          co_clust=False, verbose=False):
    """Apply the dpmm analysis to compute clusters from regions coordinates
    """
    from nipy.algorithms.clustering.imm import MixedIMM

    dim = coords.shape[1]
    migmm = MixedIMM(alpha, dim)
    migmm.set_priors(coords)
    migmm.set_constant_densities(
        null_dens=null_density, prior_dens=null_density)
    migmm._prior_dof = dof
    migmm._prior_scale = np.diag(prior_precision[0] / dof)
    migmm._inv_prior_scale_ = [np.diag(dof * 1. / (prior_precision[0]))]
    migmm.sample(coords, null_class_proba=prior_h0, niter=burnin, init=False,
                 kfold=subjects)

    # sampling
    like, pproba, co_clustering = migmm.sample(
        coords, null_class_proba=prior_h0, niter=n_iter, kfold=subjects, 
        sampling_points=sampling_coords, co_clustering=True)

    if co_clust:
        return like, 1 - pproba, co_clustering
    else:
        return like, 1 - pproba


def bsa_dpmm(hrois, prior_h0, subjects, coords, sigma, prevalence_pval, 
             prevalence_threshold, dof=10, alpha=.5, n_iter=1000, burnin=100,
             verbose=0):
    """ Estimation of the population level model of activation density using
    dpmm and inference

    Parameters
    ----------
    hrois: list of nipy.labs.spatial_models.hroi.HierarchicalROI instances
       representing individual ROIs
    Let nr be the number of terminal regions across subjects
    prior_h0: array of shape (nr)
         the mixture-based prior probability
         that the terminal regions are true positives
    subjects: array of shape (nr)
         the subject index associated with the terminal regions
    coords: array of shape (nr, coord.shape[1])
         the coordinates of the of the terminal regions
    sigma: float>0,
         expected cluster scatter in the common space in units of coord
    prevalence_pval: float in the [0,1] interval, optional
                     p-value of the prevalence test
    prevalence_threshold: float in the rannge [0,nsubj]
                          null hypothesis on region prevalence
    verbose=0, verbosity mode

    Returns
    -------
    label_map: array of shape (nnodes):
               the resulting group-level labelling of the space
    landmarks: a instance of sbf.LandmarkRegions that describes the ROIs found
               in inter-subject inference
               If no such thing can be defined landmarks is set to None
    hrois: List of  nipy.labs.spatial_models.hroi.HierarchicalROI instances
           representing individual ROIs
    density: array of shape (nnodes):
             likelihood of the data under H1 over some sampling grid
    """
    from nipy.algorithms.graph.field import field_from_coo_matrix_and_data
    domain = hrois[0].domain
    n_subjects = len(hrois)

    label_map = - np.ones(domain.size, np.int)
    landmarks = None
    density = np.zeros(domain.size)
    if len(subjects) < 1:
        return label_map, landmarks, hrois, density

    null_density = 1. / domain.local_volume.sum()

    # prepare the DPMM
    dim = domain.em_dim
    prior_precision = 1. / (sigma ** 2) * np.ones((1, dim))

    # n_iter = number of iterations to estimate density
    density, post_proba = _dpmm(
        coords, alpha, null_density, dof, prior_precision, prior_h0,
        subjects, domain.coord, n_iter=n_iter, burnin=burnin)

    Fbeta = field_from_coo_matrix_and_data(domain.topology, density)
    _, label = Fbeta.custom_watershed(0, null_density)

    # append some information to the hroi in each subject
    for subject in range(n_subjects):
        bfs = hrois[subject]
        if bfs.k > 0:
            leaves_pos = [bfs.select_id(k) for k in bfs.get_leaves_id()]

            # set posterior proba
            post_proba_ = np.zeros(bfs.k)
            post_proba_[leaves_pos] = post_proba[subjects == subject]
            bfs.set_roi_feature('posterior_proba', post_proba_)

            # set prior proba
            prior_proba = np.zeros(bfs.k)
            prior_proba[leaves_pos] = 1 - prior_h0[subjects == subject]
            bfs.set_roi_feature('prior_proba', prior_proba)
            
            # assign labels to ROIs 
            pos = np.asarray(
                [np.mean(coords, 0) for coords in bfs.get_coord()])
            midx = np.array([np.argmin(np.sum((domain.coord - pos[k]) ** 2, 1))
                             for k in range(bfs.k)])
            roi_label = - np.ones(bfs.k).astype(np.int)
            j = label[midx]
            roi_label[leaves_pos] = j[leaves_pos]
            
            # when parent regions has similarly labelled children,
            # include it also
            roi_label = bfs.make_forest().propagate_upward(roi_label)
            bfs.set_roi_feature('label', roi_label)

    # derive the group-level landmarks
    # with a threshold on the number of subjects
    # that are represented in each one
    landmarks, new_values = build_landmarks(
        hrois, prevalence_pval, prevalence_threshold, sigma, verbose=verbose)

    # make a group-level map of the landmark position
    label_map = _relabel(label, new_values)
    return label_map, landmarks, hrois, density


def bsa_dpmm2(hrois, prior_h0, subjects, coords, sigma, prevalence_pval,
              prevalence_threshold, dof=10, alpha=.5, n_iter=1000, burnin=100,
              verbose=False):
    """ Estimation of the population level model of activation density using
    dpmm and inference

    Parameters
    ----------
    hrois list of nipy.labs.spatial_models.hroi.HierarchicalROI instances
       representing individual ROIs
       let nr be the number of terminal regions across subjects
    prior_h0, array of shape (nr)
         the mixture-based prior probability
         that the terminal regions are false positives
    subjects, array of shape (nr)
         the subject index associated with the terminal regions
    coords, array of shape (nr, coord.shape[1])
         the coordinates of the of the terminal regions
    sigma float>0:
         expected cluster std in the common space in units of coord
    prevalence_pval = 0.5 (float in the [0,1] interval)
        p-value of the prevalence test
    prevalence_threshold=0, float in the rannge [0,nsubj]
        null hypothesis on region prevalence that is rejected during inference
    verbose=0, verbosity mode

    Returns
    -------
    label_map: array of shape (nnodes):
           the resulting group-level labelling of the space
    landmarks: a instance of sbf.LandmarkRegions that describes the ROIs found
        in inter-subject inference
        If no such thing can be defined landmarks is set to None
    hrois: List of  nipy.labs.spatial_models.hroi.Nroi instances
        representing individual ROIs
    Coclust: array of shape (nr,nr):
             co-labelling matrix that gives for each pair of inputs
             how likely they are in the same class according to the model
    """
    domain = hrois[0].domain
    n_subjects = len(hrois)
    label_map = - np.ones(domain.size, np.int)
    landmarks = None
    density = np.zeros(domain.size)
    if len(subjects) < 1:
        return label_map, landmarks, hrois, density

    # prepare the DPMM
    null_density = 1. / (np.sum(domain.local_volume))
    prior_precision = 1. / (sigma ** 2) * np.ones((1, domain.em_dim), np.float)

    post_proba, density, co_clustering = _dpmm(
        coords, alpha, null_density, dof, prior_precision, prior_h0,
        subjects,  n_iter=n_iter, burnin=burnin, co_clust=True)

    contingency_graph = wgraph_from_coo_matrix(co_clustering)
    if contingency_graph.E > 0:
        contingency_graph.remove_edges(contingency_graph.weights > .5)
    
    components = contingency_graph.cc()
    components[density < null_density] = components.max() + 1 +\
        np.arange(np.sum(density < null_density))

    # append some information to the hroi in each subject
    for subject in range(n_subjects):
        bfs = hrois[subject]
        if bfs is not None:
            if bfs.k > 0:
                leaves = np.asarray(
                    [bfs.select_id(id) for id in bfs.get_leaves_id()])
                roi_label = - np.ones(bfs.k).astype(np.int)

                # posterior proba of activation
                post_proba_ = np.zeros(bfs.k)
                post_proba_[leaves] = post_proba[subjects == subject]
                bfs.set_roi_feature('posterior_proba', post_proba_)

                # prior_proba of activation
                prior_proba = np.zeros(bfs.k)
                prior_proba[leaves] = 1 - prior_h0[subjects == subject]
                bfs.set_roi_feature('prior_proba', prior_proba)
                roi_label[leaves] = components[subjects == subject]

                # when parent regions has similarly labelled children,
                # include it also
                roi_label = bfs.make_forest().propagate_upward(roi_label)
                bfs.set_roi_feature('label', roi_label)
            else:
                 bfs.set_roi_feature('label', np.array([]))

    # derive the group-level landmarks
    # with a threshold on the number of subjects
    # that are represented in each one
    landmarks, _ = build_landmarks(
        hrois, prevalence_pval, prevalence_threshold, sigma, verbose=verbose)

    # make a group-level map of the landmark position
    label_map = - np.ones(domain.size)
    # not implemented at the moment

    return label_map, landmarks, hrois, co_clustering

###########################################################################
# Main function
###########################################################################


def compute_landmarks(
    domain, stats, sigma, prevalence_pval=0.5, prevalence_threshold=0, 
    threshold=3.0, smin=5, method='prior', algorithm='quick', verbose=0):
    """ Compute the  Bayesian Structural Activation paterns
    simplified version

    Parameters
    ----------
    domain : StructuredDomain instance,
          Description of the spatial context of the data
    stats: an array of shape (nbnodes, subjects):
           the multi-subject statistical maps
    sigma float>0:
         expected cluster std in the common space in units of coord
    prevalence_pval = 0.5 (float):
        posterior significance threshold
        should be in the [0,1] interval
    smin = 5 (int): minimal size of the regions to validate them
    threshold = 3.0 (float): first level threshold
    method: string, optional,
            the method used to assess the prior significance of the regions
    algorithm: string, one of ['quick', 'standard'],
               method used to compute the landmarks
    verbose=0: verbosity mode

    Returns
    -------
    label_map: array of shape (nnodes):
           the resulting group-level labelling of the space
    landmarks: a instance of sbf.LandmarkRegions that describes the ROIs found
        in inter-subject inference
        If no such thing can be defined landmarks is set to None
    hrois: List of  nipy.labs.spatial_models.hroi.Nroi instances
        representing individual ROIs
    density: array of shape (nnodes):
       likelihood of the data under H1 over some sampling grid

    Notes
    -----
    In that case, the DPMM is used to derive a spatial density of
    significant local maxima in the volume. Each terminal (leaf)
    region which is a posteriori significant enough is assigned to the
    nearest mode of this distribution
    """
    hrois, prior_h0, subjects, coords = _compute_individual_regions(
        domain, stats, threshold, smin, method, verbose)
    
    if algorithm == 'standard':
        label_map, landmarks, hrois, density = bsa_dpmm(
            hrois, prior_h0, subjects, coords, sigma, prevalence_pval,
            prevalence_threshold, verbose=verbose)
        return label_map, landmarks, hrois, density
    elif algorithm == 'quick':  
        label_map, landmarks, hrois, co_clust = bsa_dpmm2(
            hrois, prior_h0, subjects, coords, sigma, prevalence_pval,
            prevalence_threshold, verbose=verbose)
        return label_map, landmarks, hrois, co_clust
    else:
        raise ValueError('Unknown method')
