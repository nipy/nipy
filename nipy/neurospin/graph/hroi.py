import numpy as np
import graph as fg

from graph import Forest
from nipy.neurospin.utils.roi import MultipleROI

class NROI(MultipleROI,Forest):
    """
    Class for ntested ROIs.
    This inherits from both the Forest and MultipleROI 
    """

    def __init__(self,parents=None,header=None,id=None):
        """
        """
        k = np.size(parents)
        Forest.__init__(self,k,parents)
        MultipleROI.__init__(self,id, k,header)

    def clean(self, valid):
        """
        """
        MultipleROI.clean(self, valid)
        Forest.subgraph(self,valid)

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
        G = fg.Forest(self.k,self.parents)
        return G

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
        
        #for j in range(self.k):
        #    if valid[j]==0:
        #        fj =  self.parents[j]
        #        if fj!=j:
        #            self.parents[self.parents==j]=fj
        #            self.label[self.label==j]=fj
        #        else:
        #            valid[j]=1
        #
        #self.clean(valid)        

    
class HROI():
    """
    Tentative alternative definition of multiple ROI class
    """

    def __init__(self,parents=None,header=None,id=None):
        """
        """
        k = np.size(parents)
        f = Forest(self,k,parents)
        self.Forest = f
        mroi = MultipleROI(self,id, k,header)
        self.MROI = mroi


def test_nroi(verbose=0):
    """
    """
    import nifti
    nim =  nifti.NiftiImage("/tmp/blob.nii")
    header = nim.header
    k = np.size(np.unique(nim.data))-2
    nroi = NROI(parents=np.arange(k),header=header)
    nroi.from_labelled_image("/tmp/blob.nii",add=False)
    nroi.make_image("/tmp/mroi.nii")
    nroi.clean(np.arange(k)>k/5)
    nroi.set_feature_from_image('activ',"/tmp/spmT_0024.img")
    if verbose: nroi.plot_feature('activ')
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
        k = np.sum(isleaf.astype('i'))
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
            RH.set_ROI_feature(self.ROI_features[j], self.ROI_feature_ids[j])
        return RH


    def set_ROI_feature(self, feature, feature_id):
        """
        add a new feature to the class
        """
        if (feature.shape[0])==self.k:
            self.ROI_features.append(feature)
            self.ROI_feature_ids.append(feature_id)

    def get_ROI_feature(self, feature_id):
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

    def remove_feature(self, feature_id):
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
        G = fg.Forest(self.k,self.parents)
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
                convert = -np.ones(self.k).astype('i')
                aux = np.cumsum(valid.astype('i'))-1
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
            

    def propagate_AND_to_root(self,prop):
        """
        prop = self.propagate_to_root(prop)
        propagates some binary property in the tree
        that is defined in the leaves
        so that prop[parents] = AND(prop[children])
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

    def argmax(self,dmap):
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


