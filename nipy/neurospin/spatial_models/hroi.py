# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This module contains the specification of 'heierarchical ROI' object,
Which is used in spatial models of the library such as structural analysis

The connection with other classes is not completely satisfactory at the moment:
there should be some intermediate classes between 'Fields' and 'hroi'

Author : Bertrand Thirion, 2009
"""

import numpy as np
import nipy.neurospin.graph.graph as fg

from nipy.neurospin.graph.forest import Forest
from nipy.neurospin.spatial_models.mroi import SubDomains
from nipy.neurospin.graph.field import field_from_coo_matrix_and_data

def HROI_as_discrete_domain_blobs(dom, data, threshold=-np.infty, smin=0,
                                  id=''):
    """
    """
    if threshold > data.max():
        label = -np.ones(data.shape)
        parents = np.array([])
        return HierarchicalROI(dom, label, parents, id=id)
    
    # check size
    df = field_from_coo_matrix_and_data(dom.topology, data)
    idx, height, parents, label = df.threshold_bifurcations(th=threshold)    
    k = np.size(idx)

    nroi = HierarchicalROI(dom, label, parents, id=id)

    # Create a signal feature
    nroi.make_feature('signal', np.reshape(data, (np.size(data), 1)))
    
    # perform smin reduction
    k = 2* nroi.get_k()
    while k>nroi.get_k():
        k = nroi.get_k()
        size = nroi.get_size()
        nroi.merge_ascending(size>smin)
        nroi.merge_descending()
        if nroi.k==0:
            break
        size = nroi.get_size()
        if size.max()<smin:
            break #return None
        
        nroi.select(size>smin)        
    return nroi

def HROI_from_watershed(domain, data, threshold=-np.infty, id=''):
    """
    Instantiate an HierarchicalROI as the watershed of a certain dataset
    
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
        label = -np.ones(data.shape)
        parents = np.array([])
        return HierarchicalROI(domain, label, parents, id=id)

    df = field_from_coo_matrix_and_data(domain.topology, data)
    idx, height, parents, label = df.custom_watershed(0, threshold)
    nroi = HierarchicalROI(domain, label, parents, id=id)

    # this is  a custom thing, sorry
    nroi.set_roi_feature('seed', idx)
    return nroi



########################################################################
# Hierarchical ROI
########################################################################

class HierarchicalROI(SubDomains):

    def __init__(self, domain, label, parents, id='' ):
        """
        Building the HierarchicalROI
        """
        self.parents = np.ravel(parents).astype(np.int)
        SubDomains.__init__(self, domain, label, id)

    def select(self, valid, id='', no_empty_label=True):
        """
        Remove the rois for which valid==0 and update the hierarchy accordingly
        Note that auto=True automatically
        """
        SubDomains.select(self, valid, id, True, no_empty_label )
        if np.sum(valid)==0:
            self.parents = np.array([])
        else:
            self.parents = Forest(len(self.parents), self.parents).subforest(
                valid.astype(np.bool)).parents.astype(np.int)
            
        
    def make_graph(self):
        """
        output an fff.graph structure to represent the ROI hierarchy
        """
        if self.k==0:
            return None
        weights = np.ones(self.k)
        edges = (np.vstack((np.arange(self.k), self.parents))).T
        return fg.WeightedGraph(self.k, edges, weights)

    def make_forest(self):
        """
        output an fff.forest structure to represent the ROI hierarchy
        """
        if self.k==0:
            return None
        G = Forest(self.k, self.parents)
        return G

    def merge_ascending(self, valid):
        """
        self.merge_ascending(valid)

        Remove the non-valid ROIs by including them in
        their parents when it exists

        Parameters
        ----------
        valid array of shape(self.k)

        Note
        ----
        if valid[k]==0 and self.parents[k]==k, k is not removed
        """
        if np.size(valid)!= self.k:
            raise ValueError,"not the correct dimension for valid"
        if self.k==0:
            return
        order = self.make_forest().reorder_from_leaves_to_roots()
        for j in order:
            if valid[j]==0:
                fj =  self.parents[j]
                if fj!=j:
                    self.parents[self.parents==j]=fj
                    self.label[self.label==j] = fj
                    fids = self.features.keys()
                    for fid in fids:
                        dfj = self.features[fid][fj]
                        dj = self.features[fid][j]
                        self.features[fid][fj] = np.vstack((dfj, dj))
                else:
                    valid[j]=1
        
        self.select(valid)

    def merge_descending(self, methods=None):
        """
        self.merge_descending()
        Remove the items with only one son
        by including them in their son
        
        Parameters
        ----------
        methods indicates the way possible features are dealt with
        (not implemented yet)

        Caveat
        ------
        if roi_features have been defined, they will be removed
        """
        if self.k==0:
            return

        valid = np.ones(self.k).astype('bool')
        order = self.make_forest().reorder_from_leaves_to_roots()[::-1]
        for j in order:
            i = np.nonzero(self.parents==j)
            i = i[0]
            if np.sum(i!=j)==1:
                i = int(i[i!=j])
                self.parents[i] = self.parents[j]
                self.label[self.label==j] = i
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
        if self.k==0:
            return np.array([])
        return Forest(self.k, self.parents).isleaf()
        
    def reduce_to_leaves(self, id=''):
        """
        create a  new set of rois which are only the leaves of self
        """
        isleaf = Forest(self.k, self.parents).isleaf()
        label = self.label.copy()
        label[isleaf[self.label]==0] = -1
        k = np.sum(isleaf.astype(np.int))
        if self.k==0:
            return HierarchicalROI(self.domain, label, np.array([]), id)
        
        parents = np.arange(k)
        nroi = HierarchicalROI(self.domain, label, parents, id)
            
        # now copy the features
        fids = self.features.keys()
        for fid in fids:
            df = [self.features[fid][k] for k in range(self.k) if isleaf[k]]
            nroi.set_feature(fid, df)
        return nroi

    def copy(self, id=''):
        """
        returns a copy of self. self.domain is not copied.
        """
        cp = make_hroi_from_subdomain(SubDomains.copy(self, id),
                                      self.parents.copy())
        return cp

    def representative_feature(self, fid, method='mean'):
        """
        Compute an ROI-level feature given the discrete features
        
        Parameters
        ----------
        fid(string) the discrete feature under consideration
        method='average' the assessment method
        
        Returns
        -------
        the computed roi-feature is returned
        """
        if method not in['min','max','mean','cumulated_mean', 'median',
                         'weighted mean']:
            raise  ValueError, 'unknown method'
        if method=='cumulated_mean':
            data = self.features[fid]
            d0 = data[0]
            if np.size(d0) == np.shape(d0)[0]:
                np.reshape(d0, (np.size(d0),1))
            fdim = d0.shape[1]
            ldata = np.zeros((self.k, fdim))
            for k in range(self.k):
                dk = self.make_forest().get_descendents(k)
                card = np.sum(self.get_size()[dk])
                for ch in dk:
                    ldata[k] += np.sum(data[ch],0)  
                ldata[k]/=card
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
