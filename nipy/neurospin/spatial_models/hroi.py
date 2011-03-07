# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This module contains the specification of 'heierarchical ROI' object,
Which is used in spatial models of the library such as structural analysis

The connection with other classes is not completely satisfactory at the moment:
there should be some intermediate classes between 'Fields' and 'hroi'

Author : Bertrand Thirion, 2009-2011
"""

import numpy as np
import nipy.neurospin.graph.graph as fg

from nipy.neurospin.graph.forest import Forest
from nipy.neurospin.spatial_models.mroi import SubDomains
from nipy.neurospin.graph.field import field_from_coo_matrix_and_data

NINF = - np.infty


def hroi_agglomeration(input_hroi, criterion='size', smin=0):
    """ Performs an agglomeration then a selction of regions
    so that a certain size or volume criterion is staisfied

    Parameters
    ----------
    input_hroi: HierachicalROI instance,
                the input hROI
    criterion: string, optional
               to be chosen among 'size' or 'volume'
    smin: float, optional
          the applied criterion

    Returns
    -------
    output_hroi:  HierachicalROI instance
    """
    if criterion not in ['size', 'volume']:
        return ValueError('unknown criterion')
    output_hroi = input_hroi.copy()
    k = 2 * output_hroi.get_k()

    # iteratively agglomertae regions  that are too small
    while k > output_hroi.get_k():
        k = output_hroi.get_k()
        if criterion == 'size':
            value = output_hroi.get_size()
        if criterion == 'volume':
            value = np.array(
                [output_hroi.domain.get_volume(output_hroi.label == i)
                 for i in range(output_hroi.k)])

        output_hroi.merge_ascending(value > smin)
        output_hroi.merge_descending()
        if output_hroi.k == 0:
            break
        value = output_hroi.get_size()
        if value.max() < smin:
            break

    # finally remove those regions for which the criterion cannot be matched
    output_hroi.select(value > smin)
    return output_hroi


def HROI_as_discrete_domain_blobs(domain, data, threshold=NINF, smin=0,
                                  rid='', criterion='size'):
    """ Instantiate an HierarchicalROI as the blob decomposition
    of data in a certain domain

    Parameters
    ----------
    domain: discrete_domain.StructuredDomain instance,
            definition of the spatial context
    data: array of shape (domain.size),
          the corresponding data field
    threshold: float optional,
               thresholding level
    smin: float, optional,
          a threhsold on region size or cardinality.
    rid: string, optional,
         a region identifier

    Returns
    -------
    nroi: HierachicalROI instance
    """
    if threshold > data.max():
        label = - np.ones(data.shape)
        parents = np.array([])
        return HierarchicalROI(domain, label, parents, rid=rid)

    # check size
    df = field_from_coo_matrix_and_data(domain.topology, data)
    idx, height, parents, label = df.threshold_bifurcations(th=threshold)
    nroi = HierarchicalROI(domain, label, parents, rid=rid)

    # Create a signal feature
    nroi.make_feature('signal', np.reshape(data, (np.size(data), 1)))

    # agglomerate regions in order to compact the structure if necessary
    nroi = hroi_agglomeration(nroi, criterion=criterion, smin=smin)
    return nroi


def HROI_from_watershed(domain, data, threshold=NINF, rid=''):
    """Instantiate an HierarchicalROI as the watershed of a certain dataset

    Parameters
    ----------
    domain: discrete_domain.StructuredDomain instance,
            definition of the spatial context
    data: array of shape (domain.size),
          the corresponding data field
    threshold: float optional,
               thresholding level

    Returns
    -------
    the HierachicalROI instance

    Fixme
    -----
    should be a subdomain (?)
    Additionally a discrete_field is created, with the key 'index'.
                 It contains the index in the field from which
                 each point of each ROI
    """
    if threshold > data.max():
        label = - np.ones(data.shape)
        parents = np.array([])
        return HierarchicalROI(domain, label, parents, rid=rid)

    df = field_from_coo_matrix_and_data(domain.topology, data)
    idx, height, parents, label = df.custom_watershed(0, threshold)
    nroi = HierarchicalROI(domain, label, parents, rid=rid)

    # this is  a custom thing, sorry
    nroi.set_roi_feature('seed', idx)
    return nroi


########################################################################
# Hierarchical ROI
########################################################################


class HierarchicalROI(SubDomains):
    """Class that handles hierarchical ROIs
    """

    def __init__(self, domain, label, parents, rid=''):
        """Building the HierarchicalROI
        """
        self.parents = np.ravel(parents).astype(np.int)
        SubDomains.__init__(self, domain, label, rid)

    def select(self, valid, rid='', no_empty_label=True):
        """
        Remove the rois for which valid==0 and update the hierarchy accordingly
        Note that auto=True automatically
        """
        SubDomains.select(self, valid, rid, True, no_empty_label)
        if np.sum(valid) == 0:
            self.parents = np.array([])
        else:
            self.parents = Forest(len(self.parents), self.parents).subforest(
                valid.astype(np.bool)).parents.astype(np.int)

    def make_graph(self):
        """ output an fff.graph structure to represent the ROI hierarchy
        """
        if self.k == 0:
            return None
        weights = np.ones(self.k)
        edges = (np.vstack((np.arange(self.k), self.parents))).T
        return fg.WeightedGraph(self.k, edges, weights)

    def make_forest(self):
        """output an fff.forest structure to represent the ROI hierarchy
        """
        if self.k == 0:
            return None
        G = Forest(self.k, self.parents)
        return G

    def merge_ascending(self, valid):
        """Remove the non-valid ROIs by including them in
        their parents when it exists

        Parameters
        ----------
        valid array of shape(self.k)

        Note
        ----
        if valid[k]==0 and self.parents[k]==k, k is not removed
        """
        if np.size(valid) != self.k:
            raise ValueError("not the correct dimension for valid")
        if self.k == 0:
            return
        order = self.make_forest().reorder_from_leaves_to_roots()
        for j in order:
            if valid[j] == 0:
                fj = self.parents[j]
                if fj != j:
                    self.parents[self.parents == j] = fj
                    self.label[self.label == j] = fj
                    fids = self.features.keys()
                    for fid in fids:
                        dfj = self.features[fid][fj]
                        dj = self.features[fid][j]
                        self.features[fid][fj] = np.vstack((dfj, dj))
                else:
                    valid[j] = 1

        self.select(valid)

    def merge_descending(self, methods=None):
        """ Remove the items with only one son by including them in their son

        Parameters
        ----------
        methods indicates the way possible features are dealt with
        (not implemented yet)

        Caveat
        ------
        if roi_features have been defined, they will be removed
        """
        if self.k == 0:
            return

        valid = np.ones(self.k).astype('bool')
        order = self.make_forest().reorder_from_leaves_to_roots()[:: - 1]
        for j in order:
            i = np.nonzero(self.parents == j)
            i = i[0]
            if np.sum(i != j) == 1:
                i = int(i[i != j])
                self.parents[i] = self.parents[j]
                self.label[self.label == j] = i
                valid[j] = 0
                fids = self.features.keys()
                for fid in fids:
                    di = self.features[fid][i]
                    dj = self.features[fid][j]
                    self.features[fid][i] = np.vstack((di, dj))

        # finally remove  the non-valid items
        self.select(valid)

    def get_parents(self):
        return self.parents

    def get_k(self):
        return self.k

    def isleaf(self):
        """
        """
        if self.k == 0:
            return np.array([])
        return Forest(self.k, self.parents).isleaf()

    def reduce_to_leaves(self, rid=''):
        """create a  new set of rois which are only the leaves of self
        """
        isleaf = Forest(self.k, self.parents).isleaf()
        label = self.label.copy()
        label[isleaf[self.label] == 0] = -1
        k = np.sum(isleaf.astype(np.int))
        if self.k == 0:
            return HierarchicalROI(self.domain, label, np.array([]), rid)

        parents = np.arange(k)
        nroi = HierarchicalROI(self.domain, label, parents, rid)

        # now copy the features
        fids = self.features.keys()
        for fid in fids:
            df = [self.features[fid][k] for k in range(self.k) if isleaf[k]]
            nroi.set_feature(fid, df)
        return nroi

    def copy(self, rid=''):
        """ Returns a copy of self. self.domain is not copied.
        """
        cp = make_hroi_from_subdomain(SubDomains.copy(self, rid),
                                      self.parents.copy())
        return cp

    def representative_feature(self, fid, method='mean'):
        """Compute an ROI-level feature given the discrete features

        Parameters
        ----------
        fid(string) the discrete feature under consideration
        method='average' the assessment method

        Returns
        -------
        the computed roi-feature is returned
        """
        if method not in['min', 'max', 'mean', 'cumulated_mean', 'median',
                         'weighted mean']:
            raise  ValueError('unknown method')
        if method == 'cumulated_mean':
            data = self.features[fid]
            d0 = data[0]
            if np.size(d0) == np.shape(d0)[0]:
                np.reshape(d0, (np.size(d0), 1))
            fdim = d0.shape[1]
            ldata = np.zeros((self.k, fdim))
            for k in range(self.k):
                dk = self.make_forest().get_descendents(k)
                card = np.sum(self.get_size()[dk])
                for ch in dk:
                    ldata[k] += np.sum(data[ch], 0)
                ldata[k] /= card
            self.set_roi_feature(fid, ldata)
        else:
            ldata = SubDomains.representative_feature(self, fid, method)

        return ldata


def make_hroi_from_subdomain(sub_domain, parents):
    """ Instantiate an HROi from a SubDomain instance and parents
    """
    hroi = HierarchicalROI(sub_domain.domain, sub_domain.label, parents,
                           sub_domain.id)
    for k in sub_domain.features.keys():
        hroi.set_feature(k, sub_domain.features[k])
    for k in sub_domain.roi_features.keys():
        hroi.set_roi_feature(k, sub_domain.roi_features[k])
    return hroi
