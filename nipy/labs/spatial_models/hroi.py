# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This module contains the specification of 'hierarchical ROI' object,
Which is used in spatial models of the library such as structural analysis

The connection with other classes is not completely satisfactory at the moment:
there should be some intermediate classes between 'Fields' and 'hroi'

Author : Bertrand Thirion, 2009-2011
         Virgile Fritsch <virgile.fritsch@inria.fr>

"""
from __future__ import absolute_import

import numpy as np

from nipy.algorithms.graph.graph import WeightedGraph
from nipy.algorithms.graph.forest import Forest
from nipy.algorithms.graph.field import field_from_coo_matrix_and_data
from .mroi import SubDomains

NINF = - np.inf


def hroi_agglomeration(input_hroi, criterion='size', smin=0):
    """Performs an agglomeration then a selection of regions
    so that a certain size or volume criterion is satisfied.

    Parameters
    ----------
    input_hroi: HierarchicalROI instance
      The input hROI
    criterion: str, optional
      To be chosen among 'size' or 'volume'
    smin: float, optional
      The applied criterion

    Returns
    -------
    output_hroi:  HierarchicalROI instance
    """
    if criterion not in ['size', 'volume']:
        return ValueError('unknown criterion')
    output_hroi = input_hroi.copy()
    k = 2 * output_hroi.k
    if criterion == 'size':
        value = output_hroi.get_size()
    if criterion == 'volume':
        value = output_hroi.get_volume()

    # iteratively agglomerate regions that are too small
    while k > output_hroi.k:
        k = output_hroi.k
        # regions agglomeration
        output_hroi.merge_ascending(output_hroi.get_id()[value <= smin])
        # suppress parents nodes having only one child
        output_hroi.merge_descending()
        # early stopping 1
        if output_hroi.k == 0:
            break
        # early stopping 2
        if criterion == 'size':
            value = output_hroi.get_size()
        if criterion == 'volume':
            value = output_hroi.get_volume()
        if value.max() < smin:
            break

    # finally remove those regions for which the criterion cannot be matched
    output_hroi.select_roi(output_hroi.get_id()[value > smin])
    return output_hroi


def HROI_as_discrete_domain_blobs(domain, data, threshold=NINF, smin=0,
                                  criterion='size'):
    """Instantiate an HierarchicalROI as the blob decomposition
    of data in a certain domain.

    Parameters
    ----------
    domain : discrete_domain.StructuredDomain instance,
      Definition of the spatial context.
    data : array of shape (domain.size)
      The corresponding data field.
    threshold : float, optional
      Thresholding level.
    criterion : string, optional
      To be chosen among 'size' or 'volume'.
    smin: float, optional
      A threshold on the criterion.

    Returns
    -------
    nroi: HierachicalROI instance with a `signal` feature.

    """
    if threshold > data.max():
        # return an empty HROI structure
        label = - np.ones(data.shape)
        parents = np.array([])
        return HierarchicalROI(domain, label, parents)

    # check size
    df = field_from_coo_matrix_and_data(domain.topology, data)
    idx, parents, label = df.threshold_bifurcations(th=threshold)
    nroi = HierarchicalROI(domain, label, parents)
    # create a signal feature
    data = np.ravel(data)
    signal = [data[nroi.select_id(id, roi=False)] for id in nroi.get_id()]
    nroi.set_feature('signal', signal)
    # agglomerate regions in order to compact the structure if necessary
    nroi = hroi_agglomeration(nroi, criterion=criterion, smin=smin)
    return nroi


def HROI_from_watershed(domain, data, threshold=NINF):
    """Instantiate an HierarchicalROI as the watershed of a certain dataset

    Parameters
    ----------
    domain: discrete_domain.StructuredDomain instance
      Definition of the spatial context.
    data: array of shape (domain.size)
      The corresponding data field.
    threshold: float, optional
      Thresholding level.

    Returns
    -------
    nroi : ``HierarchichalROI`` instance
        The HierachicalROI instance with a ``seed`` feature.
    """
    if threshold > data.max():
        # return an empty HROI structure
        label = - np.ones(data.shape)
        parents = np.array([])
        return HierarchicalROI(domain, label, parents)

    df = field_from_coo_matrix_and_data(domain.topology, data)
    idx, label = df.custom_watershed(0, threshold)
    parents = np.arange(idx.size).astype(int)
    nroi = HierarchicalROI(domain, label, parents)

    nroi.set_roi_feature('seed', idx)
    return nroi


########################################################################
# Hierarchical ROI
########################################################################
class HierarchicalROI(SubDomains):
    """Class that handles hierarchical ROIs

    Parameters
    ----------
    k : int
      Number of ROI in the SubDomains object
    label : array of shape (domain.size), dtype=np.int
      An array use to define which voxel belongs to which ROI.
      The label values greater than -1 correspond to subregions
      labelling. The labels are recomputed so as to be consecutive
      integers.
      The labels should not be accessed outside this class. One has to
      use the API mapping methods instead.
    features : dict {str: list of object, length=self.k}
      Describe the voxels features, grouped by ROI
    roi_features : dict {str: array-like, shape=(self.k, roi_feature_dim)
      Describe the ROI features. A special feature, `id`, is read-only and
      is used to give an unique identifier for region, which is persistent
      through the MROI objects manipulations. On should access the different
      ROI's features using ids.
    parents : np.ndarray, shape(self.k)
      self.parents[i] is the index of the parent of the i-th ROI.

    TODO: have the parents as a list of id rather than a list of indices.
    """

    def __init__(self, domain, label, parents, id=None):
        """Building the HierarchicalROI
        """
        SubDomains.__init__(self, domain, label, id=id)
        self.parents = np.ravel(parents).astype(np.int)

    ###
    # Getters for very basic features or roi features
    ###
    def get_volume(self, id=None, ignore_children=True):
        """Get ROI volume

        Parameters
        ----------
        id: any hashable type, optional
          Id of the ROI from which we want to get the volume.
          Can be None (default) if we want all ROIs's volumes.
        ignore_children : bool, optional
          Specify if the volume of the node should include
          (ignore_children = False) or not the one of its children
          (ignore_children = True).

        Returns
        -------
        volume : float
          if an id is provided,
             or list of float
          if no id provided (default)
        """
        if ignore_children:
            # volume of the children is not included
            volume = SubDomains.get_volume(self, id)
        else:
            # volume of the children is included
            if id is not None:
                volume = SubDomains.get_volume(self, id)
                desc = self.make_forest().get_descendants(
                    self.select_id(id), exclude_self=True)
                # get children volume
                for k in desc:
                    volume = volume + SubDomains.get_volume(
                        self, self.get_id()[k])
            else:
                volume = []
                for id in self.get_id():
                    roi_volume = SubDomains.get_volume(self, id)
                    desc = self.make_forest().get_descendants(
                        self.select_id(id), exclude_self=True)
                    # get children volume
                    for k in desc:
                        roi_volume = roi_volume + SubDomains.get_volume(
                            self, self.get_id()[k])
                    volume.append(roi_volume)
        return volume

    def get_size(self, id=None, ignore_children=True):
        """Get ROI size (counted in terms of voxels)

        Parameters
        ----------
        id: any hashable type, optional
          Id of the ROI from which we want to get the size.
          Can be None (default) if we want all ROIs's sizes.
        ignore_children: bool, optional
          Specify if the size of the node should include
          (ignore_children = False) or not the one of its children
          (ignore_children = True).

        Returns
        -------
        size: int
          if an id is provided,
             or list of int
          if no id provided (default)

        """
        if ignore_children:
            # size of the children is not included
            size = SubDomains.get_size(self, id)
        else:
            # size of the children is included
            if id is not None:
                size = SubDomains.get_size(self, id)
                desc = self.make_forest().get_descendants(
                    self.select_id(id), exclude_self=True)
                # get children size
                for k in desc:
                    size = size + SubDomains.get_size(self, self.get_id()[k])
            else:
                size = []
                for id in self.get_id():
                    roi_size = SubDomains.get_size(self, id)
                    desc = self.make_forest().get_descendants(
                        self.select_id(id), exclude_self=True)
                    # get children size
                    for k in desc:
                        roi_size = roi_size + SubDomains.get_size(
                            self, self.get_id()[k])
                    size.append(roi_size)
        return size

    def select_roi(self, id_list):
        """Returns an instance of HROI with only the subset of chosen ROIs.

        The hierarchy is set accordingly.

        Parameters
        ----------
        id_list: list of id (any hashable type)
          The id of the ROI to be kept in the structure.

        """
        valid = np.asarray([int(i in id_list) for i in self.get_id()])
        if np.size(id_list) == 0:
            # handle the case of an empty selection
            new_parents = np.array([])
            self = HierarchicalROI(
                self.domain, -np.ones(self.label.size), np.array([]))
        else:
            # get new parents
            new_parents = Forest(self.k, self.parents).subforest(
                valid.astype(np.bool)).parents.astype(np.int)
        SubDomains.select_roi(self, id_list)
        self.parents = new_parents
        self.recompute_labels()

    def make_graph(self):
        """Output an nipy graph structure to represent the ROI hierarchy.

        """
        if self.k == 0:
            return None
        weights = np.ones(self.k)
        edges = (np.vstack((np.arange(self.k), self.parents))).T
        return WeightedGraph(self.k, edges, weights)

    def make_forest(self):
        """Output an nipy forest structure to represent the ROI hierarchy.

        """
        if self.k == 0:
            return None
        G = Forest(self.k, self.parents)
        return G

    def merge_ascending(self, id_list, pull_features=None):
        """Remove the non-valid ROIs by including them in
        their parents when it exists.

        Parameters
        ----------
        id_list: list of id (any hashable type)
          The id of the ROI to be merged into their parents.
          Nodes that are their own parent are unmodified.
        pull_features: list of str
          List of the ROI features that will be pooled from the children
          when they are merged into their parents. Otherwise, the receiving
          parent would keep its own ROI feature.
        """
        if pull_features is None:
            pull_features = []
        if self.k == 0:
            return
        id_list = [k for k in self.get_id() if k in id_list]

        # relabel maps old labels to new labels 
        relabel = np.arange(self.k)

        # merge nodes, one at a time
        for c_id in id_list:
            # define alias for clearer indexing
            c_pos = self.select_id(c_id)
            p_pos = self.parents[c_pos]
            p_id = self.get_id()[p_pos]
            
            if p_pos != c_pos:
                # this will be used in many places
                mask_pos = np.ones(self.k, np.bool)
                mask_pos[c_pos] = False
                
                # set new parents
                self.parents = self.parents[mask_pos]
                self.parents[self.parents == c_pos] = p_pos
                self.parents[self.parents > c_pos] -=  1
                self.k -= 1

                # merge labels
                relabel[relabel == c_id] = p_id

                # compute new features
                for fid in list(self.features):
                    # replace feature
                    # (without the API since self is in an inconsistent state)
                    dj = self.get_feature(fid)
                    dj[p_pos] = np.hstack((dj[self.select_id(c_id)], 
                                           dj[self.select_id(p_id)]))
                    del dj[c_pos]
                    self.features[fid] = dj

                # compute new roi features
                for fid in list(self.roi_features):
                    dj = self.get_roi_feature(fid)
                    if fid in pull_features:
                        # modify only if `pull` requested
                        dj[p_pos] = dj[c_pos]
                    self.roi_features[fid] = dj[mask_pos]

        # update the labels        
        self.label[self.label > -1] = relabel[self.label[self.label > - 1]]
        self.recompute_labels()

    def merge_descending(self, pull_features=None):
        """ Remove the items with only one son by including them in their son

        Parameters
        ----------
        methods indicates the way possible features are dealt with
        (not implemented yet)

        Caveat
        ------
        if roi_features have been defined, they will be removed
        """
        if pull_features is None:
            pull_features = []

        if self.k == 0:
            return
        
        # relabel maps old labels to new labels 
        relabel = np.arange(self.k)

        # merge nodes, one at a time
        id_list = self.get_id()[:: - 1]
        
        for p_id in id_list:
            p_pos = self.select_id(p_id)
            p_children = np.nonzero(self.parents == p_pos)[0]
            
            if p_pos in p_children:
                # remove current node from its children list
                p_children = p_children[p_children != p_pos]

            if p_children.size == 1:
                # merge node if it has only one child
                c_pos = p_children[0]
                c_id = self.get_id()[c_pos]
                mask_pos = np.ones(self.k, np.bool)
                mask_pos[p_pos] = False

                # set new parents
                self.parents[c_pos] = self.parents[p_pos]
                if self.parents[c_pos] == p_pos:
                    self.parents[c_pos] = c_pos
                self.parents = self.parents[mask_pos]
                self.parents[self.parents > p_pos] -= 1
                # merge labels
                relabel[relabel == p_pos] = relabel[c_pos]
                self.k -= 1
                
                # compute new features
                for fid in list(self.features):
                    # replace feature
                    # (without the API since self is in an inconsistent state)
                    dj = self.get_feature(fid)
                    dj[c_pos] = np.hstack((dj[self.select_id(c_id)], 
                                           dj[self.select_id(p_id)]))
                    del dj[p_pos]
                    self.features[fid] = dj

                # compute new roi features
                for fid in list(self.roi_features):
                    dj = self.get_roi_feature(fid)
                    if fid in pull_features:
                        # modify only if `pull` requested
                        dj[c_pos] = dj[p_pos]
                    self.roi_features[fid] = dj[mask_pos]
                                        
        # update HROI structure
        self.label[self.label > -1] = relabel[self.label[self.label > - 1]]
        self.recompute_labels()

    def get_parents(self):
        """Return the parent of each node in the hierarchy

        The parents are represented by their position in the nodes flat list.

        TODO:
        The purpose of this class API is not to rely on this order, so
        we should have self.parents as a list of ids instead of a list of
        positions
        """
        return self.parents

    def get_leaves_id(self):
        """Return the ids of the leaves.

        """
        if self.k == 0:
            return np.array([])
        # locate the positions of the children of each node
        is_leaf_aux = [np.where(self.parents == k)[0] for k in range(self.k)]
        # select nodes that has no child (different from themselves)
        is_leaf = np.asarray(
            [(len(child) == 0) or (len(child) == 1 and child[0] == i)
             for i, child in enumerate(is_leaf_aux)])
        # finaly return ids
        return self.get_id()[is_leaf]

    def reduce_to_leaves(self):
        """Create a  new set of rois which are only the leaves of self.

        Modification of the structure is done in place. One way therefore
        want to work on a copy a of a given HROI oject.

        """
        if self.k == 0:
            # handle the empy HROI case
            return HierarchicalROI(
                self.domain, -np.ones(self.domain.size), np.array([]))
        leaves_id = self.get_leaves_id()
        self.select_roi(leaves_id)

    def copy(self):
        """ Returns a copy of self.

        self.domain is not copied.

        """
        cp = HierarchicalROI(
            self.domain, self.label.copy(), self.parents.copy(), self.get_id())
        # copy features
        for fid in self.features:
            cp.set_feature(fid, self.get_feature(fid))
        # copy ROI features
        for fid in self.roi_features:
            cp.set_roi_feature(fid, self.get_roi_feature(fid))
        return cp

    def representative_feature(self, fid, method='mean', id=None,
                               ignore_children=True, assess_quality=True):
        """Compute a ROI representative of a given feature.

        Parameters
        ----------
        fid: str,
          Feature id
        method: str,
          Method used to compute a representative.
          Chosen among 'mean' (default), 'max', 'median', 'min',
          'weighted mean'.
        id: any hashable type
          Id of the ROI from which we want to extract a representative feature.
          Can be None (default) if we want to get all ROIs's representatives.
        ignore_children: bool,
          Specify if the volume of the node should include
          (ignore_children = False) or not the one of its children
          (ignore_children = True).
        assess_quality: bool
          If True, a new roi feature is created, which represent the quality
          of the feature representative (the number of non-nan value for the
          feature over the ROI size).
          Default is False.

        """
        rf = []
        eps = 1.e-15
        feature_quality = np.zeros(self.k)
        for i, k in enumerate(self.get_id()):
            f = self.get_feature(fid, k)
            p_pos = self.select_id(k)
            if not ignore_children:
                # also include the children features
                desc = np.nonzero(self.parents == p_pos)[0]
                if p_pos in desc:
                    desc = desc[desc != p_pos]
                for c in desc:
                    f = np.concatenate(
                        (f, self.get_feature(fid, self.get_id()[c])))
            # NaN-resistant representative
            if f.ndim == 2:
                nan = np.isnan(f.sum(1))
            else:
                nan = np.isnan(f)
            # feature quality
            feature_quality[i] = (~nan).sum() / float(nan.size)
            # compute representative
            if method == "mean":
                rf.append(np.mean(f[~nan], 0))
            if method == "weighted mean":
                lvk = self.get_local_volume(k)
                if not ignore_children:
                    # append weights for children's voxels
                    for c in desc:
                        lvk = np.concatenate(
                            (lvk,
                             self.get_local_volume(fid, self.select_id(c))))
                tmp = np.dot(lvk[~nan], f[~nan].reshape((-1, 1))) / \
                    np.maximum(eps, np.sum(lvk[~nan]))
                rf.append(tmp)
            if method == "min":
                rf.append(np.min(f[~nan]))
            if method == "max":
                rf.append(np.max(f[~nan]))
            if method == "median":
                rf.append(np.median(f[~nan], 0))
        if id is not None:
            summary_feature = rf[self.select_id(id)]
        else:
            summary_feature = rf

        if assess_quality:
            self.set_roi_feature('%s_quality' % fid, feature_quality)
        return np.array(summary_feature)


def make_hroi_from_subdomain(sub_domain, parents):
    """Instantiate an HROi from a SubDomain instance and parents

    """
    hroi = HierarchicalROI(sub_domain.domain, sub_domain.label, parents)
    # set features
    for fid in sub_domain.features:
        hroi.set_feature(fid, sub_domain.get_feature(fid))
    # set ROI features
    for fid in sub_domain.roi_features:
        hroi.set_roi_feature(fid, sub_domain.get_roi_feature(fid))
    return hroi
