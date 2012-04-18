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

Author : Bertrand Thirion, 2006-2010
"""

#autoindent

import numpy as np
from scipy import stats


class LandmarkRegions(object):
    """
    This class is intended to represent a set of inter-subject regions
    It should inherit from some abstract multiple ROI class,
    not implemented yet.
    """

    def __init__(self, domain, k, indiv_coord, subj, id=''):
        """ Building the landmark_region

        Parameters
        ----------
        domain: ROI instance
                defines the spatial context of the SubDomains
        k: int, the number of regions considered
        indiv_coord:  k-length list of arrays, optional,
                      coordinates of the nodes in some embedding space.
        subj: k-length list of integers
              these correspond to and ROI feature:
              the subject index of individual regions

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
            raise ValueError('data should have length k')
        self.features.update({fid: data})

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
        from ...algorithms.utils.fast_distance import euclidean_distance

        coord = self.get_feature('position')
        h = np.zeros(self.k)
        for k in range(self.k):
            pk = coord[k]
            sk = pk.shape[0]
            if sk < 2:
                h[k] = 0
            else:
                edk = euclidean_distance(pk)
                h[k] = edk.sum() / (sk * (sk - 1))
        return h

    def density(self, k, coord=None, dmax=1., dof=10):
        """Posterior density of component k

        Parameters
        ----------
        k: int, less or equal to self.k
           reference component
        coord: array of shape(n, self.dom.em_dim), optional
            a set of input coordinates
        dmax: float, optional
              regularizaing constant for the variance estimation
        dof: float, optional,
             strength of the regularization

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

        if k > self.k:
            raise ValueError('wrong region index')

        pos = self.get_feature('position')[k]
        center = pos.mean(0)
        dim = self.domain.em_dim

        if coord == None:
            coord = self.domain.coord

        if coord.shape[1] != dim:
            raise ValueError("incompatible dimensions")

        n_points = pos.shape[0]
        dx = pos - center
        covariance = np.dot(dx.T, dx) / n_points
        U, S, V = svd(covariance, 0)
        S = (n_points * S + dmax ** 2 * np.ones(dim) * dof) / (n_points + dof)
        sqrts = 1. / np.sqrt(S)
        dx = coord - center
        dx = np.dot(dx, U)
        dx = np.dot(dx, np.diag(sqrts))
        delta = np.sum(dx ** 2, 1)
        lcst = - np.log(2 * np.pi) * dim / 2 + (np.log(sqrts)).sum()
        pd = np.exp(lcst - delta / 2)
        return pd, delta

    def hpd(self, k, coord=None, pval=0.95, dmax=1.0):
        """Sample the posterior probability of being in k
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
        hpd, delta = self.density(k, coord, dmax)

        import scipy.special as sp
        gamma = 2 * sp.erfinv(pval) ** 2
        #
        #--- all the following is to solve the equation
        #--- erf(x/sqrt(2))-x*exp(-x**2/2)/sqrt(pi/2) = alpha
        #--- should better be put elsewhere

        def dicho_solve_lfunc(alpha, eps=1.e-7):
            if alpha > 1:
                raise ValueError("no solution for alpha>1")
            if alpha > 1 - 1.e-15:
                return np.inf
            if alpha < 0:
                raise ValueError("no solution for alpha<0")
            if alpha < 1.e-15:
                return 0

            xmin = sp.erfinv(alpha) * np.sqrt(2)
            xmax = 2 * xmin
            while lfunc(xmax) < alpha:
                xmax *= 2
                xmin *= 2
            return (dichomain_lfunc(xmin, xmax, eps, alpha))

        def dichomain_lfunc(xmin, xmax, eps, alpha):
            x = (xmin + xmax) / 2
            if xmax < xmin + eps:
                return x
            else:
                if lfunc(x) > alpha:
                    return dichomain_lfunc(xmin, x, eps, alpha)
                else:
                    return dichomain_lfunc(x, xmax, eps, alpha)

        def lfunc(x):
            return sp.erf(x / np.sqrt(2)) - x * np.exp(-x ** 2 / 2) / \
                np.sqrt(np.pi / 2)

        gamma = dicho_solve_lfunc(pval) ** 2
        hpd[delta > gamma] = 0
        return hpd

    def map_label(self, coord=None, pval=0.95, dmax=1.):
        """Sample the set of landmark regions
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
        if coord == None:
            coord = self.domain.coord
        label = - np.ones(coord.shape[0])
        if self.k > 0:
            aux = - np.ones((coord.shape[0], self.k))
            for k in range(self.k):
                aux[:, k] = self.hpd(k, coord, pval, dmax)

            maux = np.max(aux, 1)
            label[maux > 0] = np.argmax(aux, 1)[maux > 0]
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
        if fid not in self.features:
            # the feature has not been defined
            print 'using per ROI subject counts'
            for j in range(self.k):
                pvals[j] = np.size(np.unique(subj[j]))
            pvals = pvals > ths + 0.5 * (pvals == ths)
        else:
            for j in range(self.k):
                subjj = subj[j]
                conf = self.get_feature(fid)[j]
                mp = 0.
                vp = 0.
                for ls in np.unique(subjj):
                    lmj = 1 - np.prod(1 - conf[subjj == ls])
                    lvj = lmj * (1 - lmj)
                    mp = mp + lmj
                    vp = vp + lvj
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
        if fid not in self.features:
            for j in range(self.k):
                subjj = subj[j]
                confid[j] = np.size(np.unique(subjj))
        else:
            for j in range(self.k):
                subjj = subj[j]
                conf = self.get_feature(fid)[j]
                for ls in np.unique(subjj):
                    lmj = 1 - np.prod(1 - conf[subjj == ls])
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
        if np.size(feature) != self.k:
            raise ValueError('Incompatible feature dimension')

        cs = self.domain.coord
        aux = np.zeros((cs.shape[0], self.k))
        for k in range(self.k):
            aux[:, k], _ = self.density(k, cs)

        wsum = np.dot(aux, feature)
        return wsum

    def prevalence_density(self):
        """Returns a weighted map of self.prevalence

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
    bf : list of nipy.labs.spatial_models.hroi.Nroi instances
       it is assumd that each list corresponds to one subject
       each HierarchicalROI is assumed to have the roi_features
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
    """
    dim = bf[0].domain.em_dim

    # prepare various variables to ease information manipulation
    nbsubj = np.size(bf)
    subj = np.concatenate([s * np.ones(bf[s].k, np.int)
                           for s in range(nbsubj)])
    u = np.concatenate([bf[s].get_roi_feature('label')
                        for s in range(nbsubj)if bf[s].k > 0])
    u = np.squeeze(u)
    if 'prior_proba' in bf[0].roi_features:
        conf = np.concatenate([bf[s].get_roi_feature('prior_proba')
                                for s in range(nbsubj)if bf[s].k > 0])
    else:
        conf = np.ones(u.size)
    intrasubj = np.concatenate([np.arange(bf[s].k)
                                for s in range(nbsubj)])

    coords = []
    subjs = []
    pps = []
    n_labels = int(u.max() + 1)
    valid = np.zeros(n_labels).astype(np.int)

    # do some computation to find which regions are worth reporting
    for i in np.unique(u[u > - 1]):
        mp = 0.
        vp = 0.
        subjj = subj[u == i]
        for ls in np.unique(subjj):
            lmj = 1 - np.prod(1 - conf[(u == i) * (subj == ls)])
            lvj = lmj * (1 - lmj)
            mp = mp + lmj
            vp = vp + lvj

        # If noise is too low the variance is 0: ill-defined:
        vp = max(vp, 1e-14)

        # if above threshold, get some information to create the LR
        if verbose:
            print 'lr', i, valid.sum(), ths, mp, thq

        if stats.norm.sf(ths, mp, np.sqrt(vp)) > thq:
            sj = np.size(subjj)
            coord = np.zeros((sj, dim))
            for (k, s, a) in zip(intrasubj[u == i], subj[u == i], range(sj)):
                coord[a] = bf[s].get_roi_feature('position')[k]

            valid[i] = 1
            coords.append(coord)
            subjs.append(subjj)
            pps.append(conf[u == i])

    # relabel the ROIs
    maplabel = - np.ones(n_labels).astype(np.int)
    maplabel[valid > 0] = np.cumsum(valid[valid > 0]) - 1
    for s in range(nbsubj):
        if bf[s].k > 0:
            us = bf[s].get_roi_feature('label')
            us[us > - 1] = maplabel[us[us > - 1]]
            bf[s].set_roi_feature('label', us)

    # create the landmark regions structure
    k = np.sum(valid)
    LR = LandmarkRegions(bf[0].domain, k, indiv_coord=coords, subj=subjs)
    LR.set_feature('confidence', pps)
    return LR, maplabel
