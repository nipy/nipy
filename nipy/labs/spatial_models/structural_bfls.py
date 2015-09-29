# vi: set ft=python sts=4 ts=4 sw=4 et:
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
"""
The main routine of this module implement the LandmarkRegions class,
that is used to represent Regions of interest at the population level
(in a template space).

This has been used in
Thirion et al. Structural Analysis of fMRI
Data Revisited: Improving the Sensitivity and Reliability of fMRI
Group Studies.  IEEE TMI 2007

Author : Bertrand Thirion, 2006-2013
"""
from __future__ import print_function
from __future__ import absolute_import

#autoindent

import numpy as np
from scipy import stats


def _threshold_weight_map(x, fraction):
    """threshold a positive map in order to retain a certain fraction of the
    total value"""
    sorted_x = - np.sort(- x)
    if fraction < sorted_x[0] / x.sum():
        return np.zeros_like(x)

    idx = np.where(np.cumsum(sorted_x) < fraction * x.sum())[0][-1]
    x[x < sorted_x[idx]] = 0
    return x


class LandmarkRegions(object):
    """
    This class is intended to represent a set of inter-subject regions
    It should inherit from some abstract multiple ROI class,
    not implemented yet.
    """

    def __init__(self, domain, k, indiv_coord, subjects, confidence):
        """ Building the landmark_region

        Parameters
        ----------
        domain: ROI instance
                defines the spatial context of the SubDomains
        k: int,
           the number of landmark regions considered
        indiv_coord:  k-length list of arrays,
                      coordinates of the nodes in some embedding space.
        subjects: k-length list of integers
              these correspond to an ROI feature:
              the subject index of individual regions
        confidence: k-length list of arrays,
                    confidence values for the regions (0 is low, 1 is high)
        """
        self.domain = domain
        self.k = int(k)
        if len(indiv_coord) != k:
            raise ValueError('len(indiv_coord) should be equal to %d' % k)
        if len(subjects) != k:
            raise ValueError('len(subjects) should be equal to %d' % k)
        if len(confidence) != k:
            raise ValueError('len(confidence) should be equal to %d' % k)
        self.position = indiv_coord
        self.subjects = subjects
        self.confidence = confidence

    def centers(self):
        """returns the average of the coordinates for each region
        """
        pos = self.position
        centers_ = np.array([np.mean(pos[k], 0) for k in range(self.k)])
        return centers_

    def kernel_density(self, k=None, coord=None, sigma=1.):
        """ Compute the density of a component as a kde

        Parameters
        ----------
        k: int (<= self.k) or None
           component upon which the density is computed
           if None, the sum is taken over k
        coord: array of shape(n, self.dom.em_dim), optional
            a set of input coordinates
        sigma: float, optional
               kernel size

        Returns
        -------
        kde: array of shape(n)
             the density sampled at the coords
        """
        from nipy.algorithms.utils.fast_distance import euclidean_distance
        if coord is None:
            coord = self.domain.coord
        if k is None:
            kde = np.zeros(coord.shape[0])
            for k in range(self.k):
                pos = self.position[k]
                dist = euclidean_distance(pos, coord)
                kde += np.exp(- dist ** 2 / (2 * sigma ** 2)).sum(0)
        else:
            k = int(k)
            pos = self.position[k]
            dist = euclidean_distance(pos, coord)
            kde = np.exp(- dist ** 2 / (2 * sigma ** 2)).sum(0)
        return kde / (2 * np.pi * sigma ** 2) ** (pos.shape[1] / 2)

    def map_label(self, coord=None, pval=1., sigma=1.):
        """Sample the set of landmark regions
        on the proposed coordiante set cs, assuming a Gaussian shape

        Parameters
        ----------
        coord: array of shape(n,dim), optional,
               a set of input coordinates
        pval: float in [0,1]), optional
              cutoff for the CR, i.e.  highest posterior density threshold
        sigma: float, positive, optional
               spatial scale of the spatial model

        Returns
        -------
        label: array of shape (n): the posterior labelling
        """
        if coord is None:
            coord = self.domain.coord
        label = - np.ones(coord.shape[0])
        null_density = 1. / self.domain.local_volume.sum()
        if self.k > 0:
            aux = - np.zeros((coord.shape[0], self.k))
            for k in range(self.k):
                kde = self.kernel_density(k, coord, sigma)
                aux[:, k] = _threshold_weight_map(kde, pval)

            aux[aux < null_density] = 0
            maux = np.max(aux, 1)
            label[maux > 0] = np.argmax(aux, 1)[maux > 0]
        return label

    def show(self):
        """function to print basic information on self
        """
        centers = self.centers()
        subjects = self.subjects
        prevalence = self.roi_prevalence()
        print("index", "prevalence", "mean_position", "individuals")
        for i in range(self.k):
            print(i, prevalence[i], centers[i], np.unique(subjects[i]))

    def roi_prevalence(self):
        """ Return a confidence index over the different rois

        Returns
        -------
        confid: array of shape self.k
               the population_prevalence
        """
        prevalence_ = np.zeros(self.k)
        subjects = self.subjects
        for j in range(self.k):
            subjj = subjects[j]
            conf = self.confidence[j]
            for ls in np.unique(subjj):
                lmj = 1 - np.prod(1 - conf[subjj == ls])
                prevalence_[j] += lmj
        return prevalence_


def build_landmarks(domain, coords, subjects, labels, confidence=None,
                    prevalence_pval=0.95, prevalence_threshold=0, sigma=1.):
    """
    Given a list of hierarchical ROIs, and an associated labelling, this
    creates an Amer structure wuch groups ROIs with the same label.

    Parameters
    ----------
    domain: discrete_domain.DiscreteDomain instance,
            description of the spatial context of the landmarks
    coords: array of shape(n, 3)
            Sets of coordinates for the different objects
    subjects: array of shape (n), dtype = np.int
              indicators of the dataset the objects come from
    labels: array of shape (n), dtype = np.int
            index of the landmark the object is associated with
    confidence: array of shape (n),
                measure of the significance of the regions
    prevalence_pval: float, optional
    prevalence_threshold: float, optional,
                   (c) A label should be present in prevalence_threshold
                   subjects with a probability>prevalence_pval
                   in order to be valid
    sigma: float optional,
          regularizing constant that defines a prior on the region extent

    Returns
    -------
    LR : None or structural_bfls.LR instance
         describing a cross-subject set of ROIs. If inference yields a null
         result, LR is set to None
    newlabel: array of shape (n)
              a relabelling of the individual ROIs, similar to u,
              that discards labels that do not fulfill the condition (c)
    """
    if confidence is None:
        confidence = np.ones(labels.size)
    intrasubj = np.concatenate([np.arange(np.sum(subjects == s))
                                for s in np.unique(subjects)])

    coordinates = []
    subjs = []
    pps = []
    n_labels = int(labels.max() + 1)
    valid = np.zeros(n_labels).astype(np.int)

    # do some computation to find which regions are worth reporting
    for i in np.unique(labels[labels > - 1]):
        mean_c, var_c = 0., 0.
        subjects_i = subjects[labels == i]
        for subject_i in np.unique(subjects_i):
            confidence_i = 1 - np.prod(1 - confidence[(labels == i) *
                                                      (subjects == subject_i)])
            mean_c += confidence_i
            var_c += confidence_i * (1 - confidence_i)

        # If noise is too low the variance is 0: ill-defined:
        var_c = max(var_c, 1e-14)

        # if above threshold, get some information to create the landmarks
        if (stats.norm.sf(prevalence_threshold, mean_c, np.sqrt(var_c)) >
            prevalence_pval):
            coord = np.vstack([
                    coords[subjects == s][k] for (k, s) in zip(
                        intrasubj[labels == i],
                        subjects[labels == i])])
            valid[i] = 1
            coordinates.append(coord)
            subjs.append(subjects_i)
            pps.append(confidence[labels == i])

    # relabel the ROIs
    maplabel = - np.ones(n_labels).astype(np.int)
    maplabel[valid > 0] = np.cumsum(valid[valid > 0]) - 1

    # create the landmark regions structure
    LR = LandmarkRegions(domain, np.sum(valid), indiv_coord=coordinates,
                         subjects=subjs, confidence=pps)
    return LR, maplabel
