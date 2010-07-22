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
from nipy.neurospin.spatial_models.roi_ import MultipleROI, SubDomains
from nipy.neurospin.graph.field import field_from_coo_matrix_and_data

def NROI_as_discrete_domain_blobs(dom, data, threshold=-np.infty, smin=0,
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
    nroi.make_feature('signal', data)
    
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

def NROI_from_watershed(domain, data, threshold=-np.infty, id=''):
    """
    Instantiate an HierrachicalROI as the watershed of a certain dataset
    
    Parameters  
    ----------
    domain: discrete_domain.DiscreteDomain instance,
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
        Building the NROI
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

    def representative_feature(self, fid, method='average'):
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

#########################################################################
# NestedROI class
#########################################################################
    
class NestedROI(MultipleROI):

    def __init__(self, dim, parents, coord, local_volume, topology=None,
                 referential='', id='' ):
        """
        Building the NROI
        """
        self.parents = np.ravel(parents).astype(np.int)
        k = parents.size
        MultipleROI.__init__(self, dim, k, coord, local_volume, topology,
                             referential, id)

    def select(self, valid, id=''):
        """
        Remove the rois for which valid==0 and update the hierarchy accordingly
        Note that auto=True automatically
        """
        MultipleROI.select(self, valid, auto=True)
        if np.sum(valid)==0:
            self.parents=[]
            self.k = 0
        else:
            self.parents = Forest(len(self.parents), self.parents).subforest(
                valid.astype(np.bool)).parents.astype(np.int)
            self.k = np.size(self.parents)
            
        
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
        order = self.make_forest().reorder_from_leaves_to_roots()
        for j in order:#range(self.k):
            if valid[j]==0:
                fj =  self.parents[j]
                if fj!=j:
                    self.parents[self.parents==j]=fj
                    dfj = self.coord[fj]
                    dj =  self.coord[j]
                    self.coord[fj] = np.vstack((dfj, dj))
                    dfj = self.local_volume[fj]
                    dj =  self.local_volume[j]
                    self.local_volume[fj] = np.hstack((dfj, dj))

                    fids = self.features.keys()
                    for fid in fids:
                        dfj = self.discrete_features[fid][fj]
                        dj = self.discrete_features[fid][j]
                        self.discrete_features[fid][fj] = np.vstack((dfj, dj))
                else:
                    valid[j]=1

        self.select(valid)

    def merge_descending(self,methods=None):
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
        valid = np.ones(self.k).astype('bool')
        # fixme : redoder things
        
        order = self.make_forest().reorder_from_leaves_to_roots()[::-1]
        for j in order:#range(self.k):
            i = np.nonzero(self.parents==j)
            i = i[0]
            if np.sum(i!=j)==1:
                i = int(i[i!=j])
                di = self.coord[i]
                dj =  self.coord[j]
                self.coord[i] = np.vstack((di, dj))
                di = self.local_volume[i]
                dj =  self.local_volume[j]
                self.local_volume[i] = np.hstack((di, dj))
                self.parents[i] = self.parents[j]
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
        k = np.sum(isleaf.astype(np.int))
        if self.k==0: return NestedROI(self.dim, referential=self.referential,
                                       id=id)
       
        parents = np.arange(k)
        coord = [self.coord[k] for k in np.nonzero(isleaf)[0]]
        vol = [self.local_volume[k] for k in np.nonzero(isleaf)[0]]
        nroi = NestedROI(self.dim, parents, coord, vol,
                         referential=self.referential, id=id)
            
        # now copy the features
        fids = self.features.keys()
        for fid in fids:
            df = [self.features[fid][k] for k in range(self.k) if isleaf[k]]
            nroi.set_feature(fid, df)
        return nroi

    def copy(self, id=''):
        """
        returns a copy of self
        """
        cp = MultipleROI.copy(self, id)
        cp.parents = self.parents.copy()
        return cp

######################################################################
# Old NROI class --- deprecated
######################################################################
from roi import MultipleROI as MROI

class NROI(MROI, Forest):
    """
    Class for ntested ROIs.
    This inherits from both the Forest and MultipleROI
    self.k (int): number of nodes/structures included into it
    parents = None: array of shape(self.k) describing the
            hierarchical relationship
    affine=np.eye(4), array of shape(4,4),
        coordinate-defining affine transformation
    shape=None, tuple of length 3 defining the size of the grid 
        implicit to the discrete ROI definition
    """

    def __init__(self, parents=None, affine=np.eye(4), shape=None,
                 xyz=None, id='nroi'):
        """
        Building the NROI
        
        Parameters
        ----------
        parents=None: array of shape(k) providing
                      the hierachical structure
                      if parent==None, None is returned
        affine=np.eye(4), array of shape(4,4),
                           coordinate-defining affine transformation
        shape=None, tuple of length 3 defining the size of the grid 
                    implicit to the discrete ROI definition
        xyz=None, list of arrays of shape (3, nvox)
                 that yield the grid position of each grid guy.
        id='nroi', string, region identifier
        """
        
        if parents==None:
            return None
        k = np.size(parents)
        Forest.__init__(self,k,parents)
        MROI.__init__(self, id, k, affine, shape, xyz)

    def clean(self, valid):
        """
        remove the rois for which valid==0
        and update the hierarchy accordingly
        In case sum(valid)==0, 0 is returned
        """
        if np.sum(valid)==0:
            return None
        # fixme: is None a correct way of dealing with k=0 ?

        # first clean as a forest
        sf = self.subforest(valid)
        Forest.__init__(self, sf.V, sf.parents)

        # then clean as a multiple ROI
        MROI.clean(self, valid)
        return self.V
        
        
    def make_graph(self):
        """
        output an fff.graph structure to represent the ROI hierarchy
        """
        weights = np.ones(self.k)
        edges = np.transpose(np.vstack((np.arange(self.k), self.parents)))
        G = fg.WeightedGraph(self.k,edges,weights)
        return G

    def make_forest(self):
        """
        output an fff.forest structure to represent the ROI hierarchy
        """
        G = Forest(self.k,self.parents)
        return G

    def merge_ascending(self,valid,methods=None):
        """
        self.merge_ascending(valid)

        Remove the non-valid items by including them in
        their parents when it exists
        methods indicates the way possible features are dealt with.
        (not implemented yet)

        Parameters
        ----------
        valid array of shape(self.k)

        Caveat
        ------
        if roi_features have been defined, they will be removed
        """
        if np.size(valid)!= self.k:
            raise ValueError,"not the correct dimension for valid"
        for j in range(self.k):
            if valid[j]==0:
                fj =  self.parents[j]
                if fj!=j:
                    self.parents[self.parents==j]=fj
                    dfj = self.xyz[fj]
                    dj =  self.xyz[j]
                    self.xyz[fj] = np.vstack((dfj,dj))
                    fids = self.discrete_features.keys()
                    for fid in fids:
                        dfj = self.discrete_features[fid][fj]
                        dj = self.discrete_features[fid][j]
                        self.discrete_features[fid][fj] = np.vstack((dfj,dj))
                else:
                    valid[j]=1

        fids = self.roi_features.keys()
        for fid in fids: self.remove_roi_feature(fid)

        self.clean(valid)

    def merge_descending(self,methods=None):
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
        valid = np.ones(self.k).astype('bool')
        for j in range(self.k):
            i = np.nonzero(self.parents==j)
            i = i[0]
            if np.sum(i!=j)==1:
                i = int(i[i!=j])
                di = self.xyz[i]
                dj =  self.xyz[j]
                self.xyz[i] = np.vstack((di,dj))
                self.parents[i] = self.parents[j]
                valid[j] = 0
                fids = self.discrete_features.keys()
                for fid in fids:
                        di = self.discrete_features[fid][i]
                        dj = self.discrete_features[fid][j]
                        self.discrete_features[fid][i] = np.vstack((di,dj))

        # finally remove  the non-valid items
        fids = self.roi_features.keys()
        for fid in fids: self.remove_roi_feature(fid)
        self.clean(valid)
    
    def get_parents(self):
        return self.parents

    def get_k(self):
       return self.k

    def reduce_to_leaves(self):
        """
        create a  new set of rois which are only the leaves of self
        if there is none (this should not happen),
        None is returned
        """
        isleaf = self.isleaf()
        k = np.sum(isleaf.astype(np.int))
        if self.k==0: return None
        parents = np.arange(k)
        xyz = [self.xyz[k].copy() for k in np.nonzero(isleaf)[0]]
        nroi = NROI(parents, self.affine, self.shape, xyz)

        # now copy the roi_features
        fids = self.roi_features.keys()
        for fid in fids:
            nroi.set_roi_feature(fid,self.roi_features[fid][isleaf])
            
        # now copy the discrete_features
        fids = self.discrete_features.keys()
        for fid in fids:
            df = [self.discrete_features[fid][k] for k in range(self.k)
                  if isleaf[k]]
            nroi.set_discrete_feature(fid,df)
        return nroi

    def copy(self):
        """
        returns a copy of self
        """
        xyz = [self.xyz[k].copy() for k in range(self.k)]
        nroi = NROI(self.parents.copy(), self.affine, self.shape ,xyz)

        # now copy the roi_features
        fids = self.roi_features.keys()
        for fid in fids:
            nroi.set_roi_feature(fid,self.roi_features[fid].copy())

        # now copy the discrete_features
        fids = self.discrete_features.keys()
        for fid in fids:
            df = [self.discrete_features[fid][k].copy() for k in range(self.k)]
            nroi.set_discrete_feature(fid,df)
        return nroi
    
    def discrete_to_roi_features(self, fid, method='average'):
        """
        Compute an ROI-level feature given the discrete features
        
        Parameters
        ----------
        fid(string) the discrete feature under consideration
        method='average' the assessment method
        
        Results
        ------
        the computed roi-feature is returned
        """
        if method not in['min','max','average','cumulated_average']:
            raise  ValueError, 'unknown method'
        if method=='cumulated_average':
            data = self.discrete_features[fid]
            d0 = data[0]
            if np.size(d0) == np.shape(d0)[0]:
                np.reshape(d0,(np.size(d0),1))
            fdim = d0.shape[1]
            ldata = np.zeros((self.k,fdim))
            for k in range(self.k):
                dk = self.get_descendents(k)
                card = np.sum(self.get_size()[dk])
                for ch in dk:
                    ldata[k] += np.sum(data[ch],0)  
                ldata[k]/=card
            self.set_roi_feature(fid,ldata)
        else:
            ldata = MROI.discrete_to_roi_features(self, fid, method)

        return ldata


def NROI_from_watershed_dep(Field, affine, shape, xyz, refdim=0, threshold=-np.infty):
    """
    Instantiate an NROI object from a given Field and a referential
    
    Parameters  
    ----------
    Field nipy.neurospin.graph.field.Field instance
          in which the nested structure is extracted
          It is meant to be a the topological representation
          of a masked image or a mesh
    affine=np.eye(4), array of shape(4,4)
         coordinate-defining affine transformation
    shape=None, tuple of length 3 defining the size of the grid
        implicit to the discrete ROI definition  
    xyz: array of shape (Field.V,3) that represents grid coordinates
         of the object
    refdim=0: dimension fo the Field to consider (when multi-dimensional)
    threshold is a threshold so that only values above th are considered
       by default, threshold = -infty (numpy)

   Results
   -------
   the nroi instance
      
    Note
    ----
    when no region is produced (no Nroi can be defined),
         the return value is None
    Additionally a discrete_field is created, with the key 'index'.
                 It contains the index in the field from which 
                 each point of each ROI
    """
    if Field.field[:,refdim].max()>threshold:
        idx, height, parents, label = Field.custom_watershed(refdim,threshold)
    else:
        idx = []
        parents = []
        label = -np.ones(Field.V)
        
    k = np.size(idx)
    if k==0: return None
    discrete = [xyz[label==i] for i in range(k)]
    nroi = NestedROI(parents, affine, shape, discrete)
    
    #Create the index of each point within the Field
    midx = [np.expand_dims(np.nonzero(label==i)[0],1) for i in range(k)]
    nroi.set_discrete_feature('index', midx)

    # this is  a custom thing, sorry
    nroi.set_roi_feature('seed', idx)
    return nroi

def NROI_from_field(Field, affine, shape, xyz, refdim=0, threshold=-np.infty,
                    smin = 0):
    """
    Instantiate an NROI object from a given Field and a referntial
    (affine, shape)
    
    Parameters
    ----------
    Field : nipy.neurospin.graph.field.Field instance
          in which the nested structure is extracted
          It is meant to be a the topological representation
          of a masked image or a mesh
    affine=np.eye(4), array of shape(4,4)
         coordinate-defining affine transformation
    shape=None, tuple of length 3 defining the size of the grid
        implicit to the discrete ROI definition      
    xyz: array of shape (Field.V, 3) that represents grid coordinates
         of the object
    threshold is a threshold so that only values above th are considered
       by default, threshold = -infty (numpy)
    smin is the minimum size (in number of nodes) of the blobs to 
         keep.

    Results
    -------
    the nroi instance
    
    Note
    ----
    when no region is produced (no Nroi can be defined),
         the return value is None
    Additionally, a discrete_field is created, with the key 'index'.
                 It contains the index in the field from which 
                 each point of each ROI
    """
    if Field.field[:,refdim].max()>threshold:
        idx, height, parents, label = Field.threshold_bifurcations(refdim,
                                                                   threshold)
    else:
        idx = []
        parents = []
        label = -np.ones(Field.V)

    k = np.size(idx)
    if k==0: return None
    discrete = [xyz[label==i] for i in range(k)]
    nroi = NROI(parents, affine, shape, discrete)

    # Create the index of each point within the Field
    midx = [np.expand_dims(np.nonzero(label==i)[0], 1) for i in range(k)]
    nroi.set_discrete_feature('index', midx)

    #define the voxels
    # as an mroi, it should have a method to be instantiated
    #from a field/masked array ?
    k = 2* nroi.get_k()

    if k==0:
        return None

    while k>nroi.get_k():
        k = nroi.get_k()
        size = nroi.get_size()
        nroi.merge_ascending(size>smin,None)
        nroi.merge_descending(None)
        size = nroi.get_size()
        if size.max()<smin: return None
        
        nroi.clean(size>smin)
        nroi.check()
        
    return nroi
