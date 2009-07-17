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
from nipy.neurospin.utils.roi import MultipleROI



def NROI_from_field(Field,header,xyz,refdim=0,th=-np.infty,smin = 0):
    """
    Instantiate an NROI object from a given Field and a header
    
    Parameters
    ----------
    Field : nipy.neurospin.graph.field.Field instance
          in which the nested structure is extracted
          It is meant to be a the topological representation
          of a masked image or a mesh
    header : a referential-describing information
           (in the  future this should become a standard nipy transformation)
    xyz: array of shape (Field.V,3) that represents grid coordinates
         of the object
    th is a threshold so that only values above th are considered
       by default, th = -infty (numpy)
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
    if Field.field[:,refdim].max()>th:
        idx,height,parents,label = Field.threshold_bifurcations(refdim,th)
    else:
        idx = []
        parents = []
        label = -np.ones(Field.V)
        
    k = np.size(idx)
    if k==0: return None
    discrete = [xyz[label==i] for i in range(k)]
    nroi = NROI(parents,header,discrete)

    #feature = [Field.get_field()[label==i] for i in range(k)]
    #nroi.set_discrete_feature('activation', feature)
    
    # Create the index of each point within the Field
    midx = [np.expand_dims(np.nonzero(label==i)[0],1) for i in range(k)]
    nroi.set_discrete_feature('index', midx)
    
    #define the voxels
    # as an mroi, it should have a method to be instantiated
    #from a field/masked array ?
    k = 2* nroi.get_k()
    if k>0:
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

def NROI_from_watershed(Field,header,xyz,refdim=0,th=-np.infty):
    """
    Instantiate an NROI object from a given Field and a header
    
    Parameters  
    ----------
    Field nipy.neurospin.graph.field.Field instance
          in which the nested structure is extracted
          It is meant to be a the topological representation
          of a masked image or a mesh
     header : a referential-describing information
           (in the  future this should become a standard nipy transformation)
    xyz: array of shape (Field.V,3) that represents grid coordinates
         of the object
    refdim=0: dimension fo the Field to consider (when multi-dimensional)
    th is a threshold so that only values above th are considered
       by default, th = -infty (numpy)

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
    if Field.field[:,refdim].max()>th:
        idx,height,parents,label = Field.custom_watershed(refdim,th)
    else:
        idx = []
        parents = []
        label = -np.ones(Field.V)
        
    k = np.size(idx)
    if k==0: return None
    discrete = [xyz[label==i] for i in range(k)]
    nroi = NROI(parents,header,discrete)
    #feature = [Field.get_field()[label==i] for i in range(k)]
    #nroi.set_discrete_feature('activation', feature)
    
    #Create the index of each point within the Field
    midx = [np.expand_dims(np.nonzero(label==i)[0],1) for i in range(k)]
    nroi.set_discrete_feature('index', midx)

    # this is  a custom thing, sorry
    nroi.set_roi_feature('seed', idx)
    return nroi


class NROI(MultipleROI,Forest):
    """
    Class for ntested ROIs.
    This inherits from both the Forest and MultipleROI
    self.k (int): number of nodes/structures included into it
    parents = None: array of shape(self.k) describing the
            hierarchical relationship
    header (temporary): space-defining image header
           to embed the structure in an image space
    """

    def __init__(self,parents=None,header=None,xyz=None,id=None):
        """
        Building the NROI
        
        Parameters
        ----------
        parents=None: array of shape(k) providing
                      the hierachical structure
                      if parent==None, None is returned
        header=None: space defining information
                     (to be replaced by a more adequate structure)
        xyz=None list of position arrays
                 that yield the grid position of each grid guy.
        id=None: region identifier
        """
        if parents==None:
            return None
        k = np.size(parents)
        Forest.__init__(self,k,parents)
        MultipleROI.__init__(self,id, k,header,xyz)

    def clean(self, valid):
        """
        remove the rois for which valid==0
        and update the hierarchy accordingly
        In case sum(valid)==0, 0 is returned
        """
        if np.sum(valid)==0:
            return None
        # fixme :this is not coherent !
        # first clean as a forest
        sf = self.subforest(valid)
        Forest.__init__(self,sf.V,sf.parents)

        # then clean as a multiple ROI
        MultipleROI.clean(self, valid)
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
        h2 = reduce_to_leaves(self)
        create a  new set of rois which are only the leaves of self
        if there is none (this should not happen),
        None is returned
        """
        isleaf = self.isleaf()
        k = np.sum(isleaf.astype(np.int))
        if self.k==0: return None
        parents = np.arange(k)
        xyz = [self.xyz[k].copy() for k in np.nonzero(isleaf)[0]]
        nroi = NROI(parents,self.header,xyz)

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
        nroi = NROI(self.parents.copy(),self.header,xyz)

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
    
    def discrete_to_roi_features(self,fid,method='average'):
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
            df = self.discrete_features[fid]
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
            ldata = MultipleROI.discrete_to_roi_features(self,fid,method)

        return ldata

