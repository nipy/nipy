from _field import *
from _field import __doc__
import numpy as np
import graph as fg

class Field(fg.WeightedGraph):
    """
    This is the basic field structure,
    which contains the weighted graph structure plus a matrix of data (the 'field')
    -field is an array of size(n,p) where n is the number of vertices of the graph and p is the field dimension
    """
    def __init__(self, V, edges=None, weights=None, field=None):
        """
        INPUT:
        V: the number of vertices of the graph
        edges=None: the edge array of the graph
        weights=None: the asociated weights array
        field=None: the data itself
        This is the build function
        """
        V = int(V)
        if V<1:
            raise ValueError, 'cannot create graph with no vertex'
        self.V = int(V)
        self.E = 0
        self.edges = []
        self.weights = []
        if (edges==None)&(weights==None):
            pass
        else:
            if edges.shape[0]==np.size(weights):
                E = edges.shape[0]
                # quick and dirty, sorry...
                self.V = V
                self.E = E
                self.edges = edges
                self.weights = weights
            else:
                raise ValueError, 'Incompatible size of the edges and weights matrices'
        self.field = []
        if field==None:
            pass
        else:
            if np.size(field)==self.V:
                field = np.reshape(field,(self.V,1))
            if field.shape[0]!=self.V:
                raise ValueError, 'field does not have a correct size'
            else:
                self.field = field

    def print_field(self):
        print self.field

    def get_field(self):
        return self.field
        

    def set_field(self,field):
        if np.size(field)==self.V:
            field = np.reshape(field,(self.V,1))
        if field.shape[0]!=self.V:
            raise ValueError, 'field does not have a correct size'
        else:
            self.field = field

    def closing(self,nbiter=1):
        """
        self.closing(nbiter=1)
        Morphological closing of the field
        IMPUT
        nbiter=1 : the number of iterations required
        """
        nbiter = int(nbiter)
        if self.E>0:
            if nbiter>0:
                for i in range (self.field.shape[1]):
                    self.field[:,i] = closing(self.edges[:,0],self.edges[:,1],self.field[:,i],nbiter)

    def opening(self,nbiter=1):
        """
        self.opening(nbiter=1)
        Morphological openeing of the field
        IMPUT
        nbiter=1 : the number of iterations required
        """
        nbiter = int(nbiter)
        if self.E>0:
            if nbiter>0:
                for i in range (self.field.shape[1]):
                    self.field[:,i] = opening(self.edges[:,0],self.edges[:,1],self.field[:,i],nbiter)

                    
    def dilation(self,nbiter=1):
        """
        self.dilation(nbiter=1)
        Morphological openeing of the field
        IMPUT
        nbiter=1 : the number of iterations required
        """
        nbiter = int(nbiter)
        if self.E>0:
            if nbiter>0:
                for i in range (self.field.shape[1]):
                    self.field[:,i] = dilation(self.edges[:,0],self.edges[:,1],self.field[:,i],nbiter)

    def erosion(self,nbiter=1):
        """
        self.erosion(nbiter=1)
        Morphological openeing of the field
        IMPUT
        nbiter=1 : the number of iterations required
        """
        nbiter = int(nbiter)
        if self.E>0:
            if nbiter>0:
                for i in range (self.field.shape[1]):
                    self.field[:,i] = erosion(self.edges[:,0],self.edges[:,1],self.field[:,i],nbiter)

    def get_local_maxima(self,refdim=0,th=-np.infty):
        """
        idx,depth = get_local_maxima(th=-infty)
        Look for the local maxima of  field[:,refdim]
        INPUT :
        - refdim is the field dimension over which the maxima are looked after
        - th is a threshold so that only values above th are considered
        by default, th = -infty (numpy)
        OUTPUT:
        - idx: the indices of the vertices that are local maxima
        - depth: the depth of the local maxima :
        depth[idx[i]] = q means that idx[i] is a q-order maximum
        """
        refdim = int(refdim)
        if (np.size(self.field)==0):
            raise ValueError, 'No field has been defined so far'
        if self.field.shape[1]-1<refdim:
            raise ValueError, 'refdim>field.shape[1]'
        idx = np.arange(np.sum(self.field>th))
        depth = self.V*np.ones(np.sum(self.field>th),'i')
        if self.E>0:
            try:
                idx,depth = get_local_maxima(self.edges[:,0],self.edges[:,1],self.field[:,refdim],th)
            except:
                idx = []
                depth = []
                
        return idx,depth

    def local_maxima(self,refdim=0):
        """
        Look for all the local maxima of a field
        INPUT :
        - refdim is the field dimension over which the maxima are looked after
        OUTPUT:
        - depth: a labelling of the vertices such that
         depth[v] = 0 if v is not a local maximum
         depth[v] = 1 if v is a first order maximum
         ...
         depth[v] = q if v is a q-order maximum

        """
        refdim = int(refdim)
        if (np.size(self.field)==0):
            raise ValueError, 'No field has been defined so far'
        if self.field.shape[1]-1<refdim:
            raise ValueError, 'refdim>field.shape[1]'
        depth = self.V*np.ones(self.V,'i')
        if self.E>0:
            depth = local_maxima(self.edges[:,0],self.edges[:,1],self.field[:,refdim])
        return depth

    def diffusion(self,nbiter=1):
       """
       diffusion of a field of data in the weighted graph structure
       INPUT :
       - nbiter=1: the number of iterations required
       (the larger the smoother)
       NB:
       - This is done for all the dimensions of the field
       """
       nbiter = int(nbiter)
       if (self.E>0)&(nbiter>0)&(np.size(self.field)>0):
            self.field = diffusion(self.edges[:,0],self.edges[:,1],self.weights,self.field,nbiter)


    def custom_watershed(self,refdim=0,th=-np.infty):
        """
        idx,depth, major,label = self.custom_watershed(refim = 0,th=-infty)
        perfoms a watershed analysis of the field.
        Note that bassins are found aound each maximum
        (and not minimum as conventionally)
        INPUT :
        - th is a threshold so that only values above th are considered
        by default, th = -infty (numpy)
        OUTPUT:
        - idx: the indices of the vertices that are local maxima
        - depth: the depth of the local maxima
        depth[idx[i]] = q means that idx[i] is a q-order maximum
        Note that this is also the diameter of the basins
        associated with local maxima
        - major: the label of the maximum which dominates each local maximum
        i.e. it describes the hierarchy of the local maxima
        - label : a labelling of thevertices according to their bassin
        idx, depth and major have length q, where q is the number of bassins
        label as length n: the number of vertices
        """
        if (np.size(self.field)==0):
            raise ValueError, 'No field has been defined so far'
        if self.field.shape[1]-1<refdim:
            raise ValueError, 'refdim>field.shape[1]'
        f = self.field[:, refdim]
        idx = np.nonzero(f>th)
        idx = np.reshape(idx, np.size(idx))
        depth = self.V*np.ones(np.sum(f>th), np.int)
        major = np.arange(np.sum(f>th))
        label = np.zeros(self.V, np.int)
        label[idx] = major
        if self.E>0:
            idx, depth, major, label = custom_watershed(self.edges[:, 0],
                        self.edges[:,1], f, th)
        return idx,depth,major,label

    def threshold_bifurcations(self,refdim=0,th=-np.infty):
        """
        idx,height, parents,label = threshold_bifurcations(refdim = 0,th=-infty)
        perfoms theanalysis of the level sets of the field:
        Bifurcations are defined as changes in the topology in the level sets
        when the level (threshold) is varied
        This can been thought of as a kind of Morse description
        INPUT :
        - th is a threshold so that only values above th are considered
        by default, th = -infty (numpy)
        OUTPUT:
        - idx: the indices of the vertices that are local maxima
        - height: the depth of the local maxima
        depth[idx[i]] = q means that idx[i] is a q-order maximum
        Note that this is also the diameter of the basins
        associated with local maxima
        - parents: the label of the maximum which dominates each local maximum
        i.e. it describes the hierarchy of the local maxima
        - label : a labelling of thevertices according to their bassin
        idx, depth and major have length q, where q is the number of bassins
        label as length n: the number of vertices
        """
        if (np.size(self.field)==0):
            raise ValueError, 'No field has been defined so far'
        if self.field.shape[1]-1<refdim:
            raise ValueError, 'refdim>field.shape[1]'
        idx = np.nonzero(self.field[:,refdim]>th)
        height = self.V*np.ones(np.sum(self.field>th))
        parents = np.arange(np.sum(self.field>th))
        label = np.zeros(self.V, np.int)
        label[idx] = parents
        if self.E>0:
            idx,height, parents,label = threshold_bifurcations(self.edges[:,0],self.edges[:,1],self.field[:,refdim],th)
        return idx,height,parents,label

    def generate_blobs(self,refdim=0,th=-np.infty,smin = 0):
        """
        NROI = threshold_bifurcations(refdim = 0,th=-infty,smin=0)

        INPUT
        - th is a threshold so that only values above th are considered
        by default, th = -infty (numpy)
        - smin is the minimum size (in number of nodes) of the blobs to 
        keep.  
        """
        import hroi
        if self.field.max()>th:
            idx,height,parents,label = self.threshold_bifurcations(refdim,th)
        else:
            idx = []
            parents = []
            label = -np.ones(self.V)
        
        k = np.size(idx)
        nroi = hroi.ROI_Hierarchy(k,idx, parents,label)      
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

    def constrained_voronoi(self,seed):
        """
        label = self.constrained_voronoi(seed)
        performs a voronoi parcellation of the field starting from the input seed
        INPUT:
        seed: int array of shape(p), the input seeds
        OUTPUT:
        label: The resulting labelling of the data
        """
        if (np.size(self.field)==0):
            raise ValueError, 'No field has been defined so far'
        seed = seed.astype('i')
        label = field_voronoi(self.edges[:,0],self.edges[:,1],self.field,seed)
        return label

    def copy(self):
        """
        copy function
        """
        return Field(self.V,self.edges,self.weights,self.field)

    
