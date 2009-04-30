import numpy as np
import nipy.neurospin.graph.graph as fg
import nipy.neurospin.graph.field as ff

from nipy.neurospin.graph.forest import Forest
from nipy.neurospin.utils.roi import MultipleROI





def NROI_from_field(Field,header,xyz,refdim=0,th=-np.infty,smin = 0):
    """
    Instantiate an NROI structure from a given Field and a header
    
    INPUT
    - th is a threshold so that only values above th are considered
    by default, th = -infty (numpy)
    - smin is the minimum size (in number of nodes) of the blobs to 
    keep.
    
    NOTE
    - when no region is produced (no Nroi can be defined),
    the return value is None
    - additionally a discrete_field is created, with the key 'activation',
    which refers to the input data used to create the NROI
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
    feature = [Field.get_field()[label==i] for i in range(k)]
    nroi.set_discrete_feature('activation', feature)
    
    # this should disappear in the future 
    midx = [np.expand_dims(np.nonzero(label==i)[0],1) for i in range(k)]
    nroi.set_discrete_feature('masked_index', midx)
    
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
    Instantiate an NROI structure from a given Field and a header
    
    INPUT
    - Field is the field to be watershed
    - header is a referential-defining argument
    - xyz is a set of coordinate arrays that corresponds to array positions
    - refdim=0: dimension fo the Field to consider (when multi-dimensional)
    - th is a threshold so that only values above th are considered
    by default, th = -infty (numpy)
    
    NOTE
    - when no region is produced (no Nroi can be defined),
    the return value is None
    - additionally a discrete_field is created, with the key 'activation',
    which refers to the input data used to create the NROI
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
    feature = [Field.get_field()[label==i] for i in range(k)]
    nroi.set_discrete_feature('activation', feature)
    
    # this should disappear in the future 
    midx = [np.expand_dims(np.nonzero(label==i)[0],1) for i in range(k)]
    nroi.set_discrete_feature('masked_index', midx)

    # this is  a custom thing, sorry
    nroi.set_roi_feature('seed', idx)
    return nroi


class NROI(MultipleROI,Forest):
    """
    Class for ntested ROIs.
    This inherits from both the Forest and MultipleROI
    - self.k (int): number of nodes/structures included into it
    - parents = None: array of shape(self.k) describing the
    hierarchical relationship
    - header (temporary): space-defining image header
    to embed the structure in an image space
    """

    def __init__(self,parents=None,header=None,discrete=None,id=None):
        """
        Building the NROI
        - parents=None: array of shape(k) providing
        the hierachical structure
        if parent==None, None is returned
        - header=None: space defining information
        (to be replaced by a more adequate structure)
        - discrete=None list of position arrays
        that yield the grid position of each grid guy.
        -id=None: region identifier
        """
        if parents==None:
            return None
        k = np.size(parents)
        Forest.__init__(self,k,parents)
        MultipleROI.__init__(self,id, k,header,discrete)

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

        INPUT:
        - valid array of shape(self.k)

        CAVEAT:
        if roi_features have been defined, they will be removed
        """
        if np.size(valid)!= self.k:
            raise ValueError,"not the correct dimension for valid"
        for j in range(self.k):
            if valid[j]==0:
                fj =  self.parents[j]
                if fj!=j:
                    self.parents[self.parents==j]=fj
                    dfj = self.discrete[fj]
                    dj =  self.discrete[j]
                    self.discrete[fj] = np.vstack((dfj,dj))
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
        methods indicates the way possible features are dealt with
        (not implemented yet)
        CAVEAT:
        if roi_features have been defined, they will be removed
        """
        valid = np.ones(self.k).astype('bool')
        for j in range(self.k):
            i = np.nonzero(self.parents==j)
            i = i[0]
            if np.sum(i!=j)==1:
                i = int(i[i!=j])
                di = self.discrete[i]
                dj =  self.discrete[j]
                self.discrete[i] = np.vstack((di,dj))
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
        discrete = [self.discrete[k].copy() for k in np.nonzero(isleaf)[0]]
        nroi = NROI(parents,self.header,discrete)

        # now copy the roi_features
        fids = self.roi_features.keys()
        for fid in fids:
            nroi.set_roi_feature(fid,self.roi_features(fid)[isleaf])
            
        # now copy the discrete_features
        fids = self.discrete_features.keys()
        for fid in fids:
            df = [self.discrete_features[fid][k] for k in range(self.k) if isleaf[k]]
            nroi.set_discrete_feature(fid,df)
        return nroi

    def copy(self):
        """ returns a copy of self
        """
        discrete = [self.discrete[k].copy() for k in range(self.k)]
        nroi = NROI(self.parents.copy(),self.header,discrete)

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
        INPUT:
        - fid(string) the discrete feature under consideration
        - method='average' the assessment method
        OUPUT:
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

# -----------------------------------------------------------------------
# ----------- deprecated stuff ------------------------------------------
# -----------------------------------------------------------------------

def  _generate_blobs_(Field,refdim=0,th=-np.infty,smin = 0):
        """
        NROI = threshold_bifurcations(refdim = 0,th=-infty,smin=0)

        INPUT
        - th is a threshold so that only values above th are considered
        by default, th = -infty (numpy)
        - smin is the minimum size (in number of nodes) of the blobs to 
        keep.

         """
        if Field.field.max()>th:
            idx,height,parents,label = Field.threshold_bifurcations(refdim,th)
        else:
            idx = []
            parents = []
            label = -np.ones(Field.V)
        
        k = np.size(idx)
        nroi = ROI_Hierarchy(k,idx, parents,label)      
        k = 2* nroi.get_k()
        if k>0:
            while k>nroi.get_k():
                k = nroi.get_k()
                size = nroi.compute_size()
                nroi.merge_ascending(size>smin,None)
                nroi.merge_descending(None)
                size = nroi.compute_size()
                nroi.clean(size>smin)
                nroi.check()
        return nroi


class ROI_Hierarchy:
    """
    Class for the modelling of hierarchical ROIs
    main attributes:
    k : number of regions in the graph
    parents : parents in the tree sense of an ROI
    seed :  reference item(voxel) for a given ROI
    Label : associated labelling of a certain dataset
    ROI_features : list of arrays that are ROI-related features
    ROI_feature_ids : identifiers of the ROI-related features
    """

    def __init__(self,k=1, seed=None,parents=None,label=None):
        self.k = k
        if seed==None:
            self.seed = np.arange(self.k) 
        else:
            self.seed = seed

        if parents==None:
            self.parents = np.arange(self.k)
        else:
            self.parents = parents

        if label==None:
            self.label = np.arange(k)
        else:
            self.label = label
        self.ROI_features = []
        self.ROI_feature_ids = []
        self.check()

    def check(self):
        """
        check that all the informations are compatible...
        """
        OK = np.size(self.seed)==self.k
        OK = OK & (np.size(self.parents)==self.k)
        OK = OK & (self.label.max()+1 == self.k)

        if OK==0:
            print self.k,np.size(self.parents),np.size(self.seed),self.label.max()+1
            raise ValueError, "ROIhierarchy is not coherent"

    def set_parents(self,parents):
        if np.size(parents)==self.k:
            self.parents = parents
            self.check()
        else:
            print "incorrect size for parents"

    def set_seed(self,seed):
        if np.size(seed==self.k):
            self.seed = seed
            self.check()
        else:
            print "incorrect size for seed"

    def set_label(self,label):
        self.label = label
        self.check()

    def get_parents(self):
        return self.parents

    def get_seed(self):
        return self.seed

    def get_label(self):
        return self.label

    def get_leaf_label(self):
        """
        return self.labels, but with -1 instead of the label
        when the label does not correpond to a leaf
        """
        label = self.label.copy()
        nisleaf = self.isleaf()==0
        label[nisleaf[label]]=-1
        return label

    def reduce_to_leaves(self):
        """
        h2 = reduce_to_leaves(self)
        create a  new set of rois which are only the leaves of self
        """
        isleaf = self.isleaf()
        k = np.sum(isleaf.astype(np.int))
        if self.k>0:
            seed = self.seed[isleaf]
            parents = np.arange(k)
            label =  self.label.copy()
            lmap = -np.ones(self.k,'i')
            lmap[isleaf] = np.arange(k)
            label[label>-1] = lmap[label[label>-1]]
            h = ROI_Hierarchy(k, seed,parents,label)
        else:
            h = self
        return h

    def get_k(self):
        return self.k


    def copy(self):
        RH = ROI_Hierarchy(self.k,self.seed.copy(),self.parents.copy(),
                           self.label.copy())
        for j in range(len(self.ROI_features)):
            RH.set_roi_feature(self.ROI_features[j], self.ROI_feature_ids[j])
        return RH


    def set_roi_feature(self, feature, feature_id):
        """
        add a new feature to the class
        """
        if (feature.shape[0])==self.k:
            self.ROI_features.append(feature)
            self.ROI_feature_ids.append(feature_id)

    def get_roi_feature(self, feature_id):
        """
        give the feature associated with a given id, if it ecists
        """
        i = np.array([fid==feature_id for fid in self.ROI_feature_ids])
        i = np.nonzero(i)
        i = np.reshape(i,np.size(i))
        if np.size(i)==0:
            print "The feature does not exist"
            return []
        if np.size(i)==1:
            return self.ROI_features[int(i)]
        if np.size(i)>1:
            print "amibiguous feature id"
            return []

    def remove_roi_feature(self, feature_id):
        """
        removes the feature associated with a given id, if it exists
        """
        i = np.array([fid!=feature_id for fid in self.ROI_feature_ids])
        i = np.nonzero(i)
        i = np.reshape(i,np.size(i))
        Rf = [self.ROI_features[j] for j in i]
        Rfid =[self.ROI_feature_ids[j] for j in i]
        self.ROI_features = Rf
        self.ROI_feature_ids= Rfid
        

    def make_feature(self,data,id,method='mean'):
        """
        self.make_feature(data,id,method='mean')
        given a dataset, compute a ROI-based feature through averaging,
        min or max or cumulative mean
        INPUT:
        data = array of shape np.size((self.label))
        id = (string) id of the feature
        method = 'min','mean','max' or 'cumulative_mean'
        """
        if data.shape[0]==np.size(self.label):
            if method == 'mean':
                feature = np.array([np.mean(data[self.label==j],0)
                                   for j in range (self.k)])
            if method == 'min':
                feature = np.array([np.min(data[self.label==j],0)
                                   for j in range (self.k)])
            if method == 'max':
                feature = np.array([np.max(data[self.label==j],0)
                                   for j in range (self.k)])
            if method == 'cumulative_mean':
                feature = np.array([np.sum(data[self.label==j],0)
                                   for j in range (self.k)])
                weight = np.array([np.sum(self.label==j)
                                  for j in range (self.k)])
                for j in range(self.k):
                    fj = feature[j]
                    wj = weight[j]
                    i = j
                    while self.parents[i]!=i:
                        feature[self.parents[i]]+=fj
                        weight[self.parents[i]]+=wj
                        i=self.parents[i]
                feature = np.array([feature[j]/weight[j]
                                   for j in range (self.k)])
                
            self.ROI_features.append(feature)
            self.ROI_feature_ids.append(id)
        else:
            print "incorrect size for data"

    def make_graph(self):
        """
        output an fff.graph stracture to represent the ROI hierarchy
        """
        weights = np.ones(self.k)
        edges = np.transpose(np.vstack((np.arange(self.k), self.parents)))
        G = fg.WeightedGraph(self.k,edges,weights)
        return G

    def make_forest(self):
        """
        output an fff.forest structure to represent the ROI hierarchy
        """
        weights = np.ones(self.k)
        G = Forest(self.k,self.parents)
        return G

    def compute_size(self):
        size = np.array([ np.size(np.nonzero(self.label==i)) for i in range(self.k)])
        return size
    
    def clean(self,valid):
        """
        remove the non-valid compnents and reorder the ROI list
        """
        if self.k>0:
            if np.size(valid)==self.k:
            ## remove unvalid null components
                for j in range(self.k):
                    if valid[self.parents[j]]==0:
                        self.parents[j]=j
                        
                
                iconvert = np.nonzero(valid)
                iconvert = np.reshape(iconvert, np.size(iconvert))
                convert = -np.ones(self.k).astype(np.int)
                aux = np.cumsum(valid.astype(np.int))-1
                convert[valid] = aux[valid]
                #print valid,iconvert,aux,convert
                self.k = np.size(iconvert)
            
                self.seed = self.seed[iconvert]
                q = np.nonzero(self.label>-1)
                q = np.reshape(q,np.size(q))
                self.label[q] = convert[self.label[q]]
                self.parents = convert[self.parents[iconvert]]

                self.seed = self.seed[:self.k]
                self.parents = self.parents[:self.k]
                self.V = self.k

                self.ROI_features = [f[iconvert] for f in self.ROI_features]
                self.ROI_features = [f[:self.k] for f in self.ROI_features]
                self.check()
            else:
                raise ValueError,"incoherent size"
        
    def merge_ascending(self,valid,methods=None):
        """
        self.merge_ascending(valid)
        Remove the non-valid items by including them in
        their parents when it exists
        methods indicates the way possible features are dealt with
        """
        for j in range(self.k):
            if valid[j]==0:
                fj =  self.parents[j]
                if fj!=j:
                    self.parents[self.parents==j]=fj
                    self.label[self.label==j]=fj
                else:
                    valid[j]=1
        
        self.clean(valid)

    def merge_descending(self,methods=None):
        """
        self.merge_descending()
        Remove the items with only one son
        by including them in their son
        methods indicates the way possible features are dealt with
        """
        valid = np.ones(self.k).astype('bool')
        for j in range(self.k):
            i = np.nonzero(self.parents==j)
            i = i[0]
            if np.sum(i!=j)==1:
                i = int(i[i!=j])
                
                self.label[self.label==j]=i
                self.parents[i] = self.parents[j]
                valid[j] = 0
                # todo : update the features!!!

        # finally remove  the non-valid items
        self.clean(valid)

    def isfield(self,id):
        """
        tests whether a given id is among the current list of ROI_features
        """
        b = 0
        for s in self.ROI_feature_ids:
            if s==id:
                b=1
        return b
    
    def isleaf(self):
        """
        Returns a boolean vector of size self.k
        ==1 iff the item is the parents of nobody
        """
        b = np.ones(self.k).astype('bool')
        for i in range(self.k):
            if self.parents[i]!=i:
                b[self.parents[i]] = False
        return b


    def maxdepth(self):
        """
        depth = self.maxdepth()
        return a labelling of the nodes so that leaves
        are labelled by 0
        and depth[i] = max_{j \in ch[i]} depth[j] + 1
        recursively
        # depth_from_leaves
        """
        depth = -np.ones(self.k,'i')
        depth[self.isleaf()]=0

        for j in range(self.k):
            dc = depth.copy()
            for i in range(self.k):
                if self.parents[i]!=i:
                    depth[self.parents[i]] = np.maximum(depth[i]+1,depth[self.parents[i]])
            if dc.max()==depth.max():
                break
        return depth

    def tree_depth(self):
        """
        td = self.tree_depth()
        return the maximal depth of any node in the tree
        """
        depth = self.maxdepth()
        #print "tree depth ", self.k, depth.max()+1
        return depth.max()+1
            

    def propagate_upward_and(self,prop):
        """
        prop = self.propagate_to_root(prop)
        propagates some binary property in the tree
        that is defined in the leaves
        so that prop[parents] = AND(prop[children])
        # propagate_upward_and
        """
        if np.size(prop)!=self.k:
            raise ValueError,"incoherent size for prop"

        prop[self.isleaf()==False]=True

        for j in range(self.tree_depth()):
            for i in range(self.k):
                if prop[i] == False:
                    prop[self.parents[i]] = False

        return prop

    def propagate_upward(self,label):
        """
        label = self.propagate_upward(label)
        Assuming that label is a certain positive integer field
        (i.e. labels)
        that is defined at the leaves of the tree
        and can be compared,
        this propagates these labels to the parents whenever
        the children nodes have coherent propserties
        otherwise the parent value is unchanged
        """
        if np.size(label)!=self.k:
            raise ValueError,"incoherent size for label"

        f = self.make_forest()
        ch = f.get_children()
        depth = self.maxdepth()
        for j in range(1,depth.max()+1):
            for i in range(self.k):
                if depth[i]==j:
                    if np.size(np.unique(label[ch[i]]))==1:
                        label[i] = np.unique(label[ch[i]])
                    
        return label
        

    def subtree(self,k):
        """
        l = self.subtree(k)
        returns an array of the nodes included in the subtree rooted in k
        #rooted_subtree
        """
        if k>self.k:
            raise ValueError,"incoherent value for k"
        if k<0:
            raise ValueError,"incoherent value for k"

        valid = np.zeros(self.k)
        valid[k] = 1
        sk = 0
        while valid.sum()>sk:
            sk = valid.sum().copy()
            for i in range(self.k):
                if valid[self.parents[i]]==1:
                    valid[i]=1

        i = np.nonzero(valid)
        i = np.reshape(i,np.size(i))
        return i

    def feature_argmax(self,dmap):
        """
        idx = self.argmax(dmap)
        INPUT:
        - dmap: array of shape(np.size(self.label))
        OUTPUT:
        - idx: array of indices with shape (self.k)
        for each label i in [0..k-1], argmax_{label==i}(map) is computed
        and the result is returned in idx
        """
        dmap = np.squeeze(dmap.copy())
        if np.size(dmap)!=np.size(self.label):
            raise ValueError,"Incorrect size for dmap"

        idx = [np.argmax(dmap*(self.label==i)) for i in range(self.k)]
        return np.array(idx)

