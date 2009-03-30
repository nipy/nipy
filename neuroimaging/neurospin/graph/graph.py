from _graph import *
from _graph import __doc__
import numpy as np

"""
This module implements the main graph classes of fff2
Graph: basic topological graph, i.e. vertices and edges. Not well developed
WeightedGraph (Graph): Idem plus values asociated with vertices
BipartiteGraph (WeightedGraph): Idem but the graph is Bipartite
Forest(WeightedGraph): This is a special case of a Weighted Graph (i.e.  a set of trees)

Author:Bertrand Thirion
"""

class Graph:
    """
    This is the basic topological (non-weighted) directed Graph class
    fields :
    - V(int) = the number of vertices
    - E(int) = the number of edges
    - edges = array of int with shape (E,2) : the edges of the graph
    """
    
    def __init__(self, V, E=0):
        self.V = int(V)
        if self.V < 1:
            raise ValueError, 'Empty graphs cannot be created'

        self.E = int(E)
        if self.E<0:
            self.E = 0

        self.vertices =  [a for a in range(self.V)]
        self.edges = np.zeros((self.E,2),np.int)

    def set_edges(self,edges):
        """
        sets self.edges=edges if
        1. edges has a correct size
        2. edges take values in [1..V]
        """
        if np.shape(edges)!=np.shape(self.edges):
            raise ValueError, 'Incompatible size of the edge matrix'
        
        if np.size(edges)>0:
            if edges.max()+1>self.V:
                raise ValueError, 'Incorrect edge specification'
        self.edges = edges

    def get_vertices(self):
        return self.vertices
    
    def get_edges(self):
        try:
            temp = self.edges
        except:
            temp = []
        return temp

    def get_V(self):
        return self.V
    
    def get_E(self):
        return self.E

    def adjacency(self):
        A = np.zeros((self.V,self.V))
        for e in range(self.E):
            i = self.edges[e][0]
            j = self.edges[e][1]
            A[i,j] = 1
        return(A)

    def complete(self):
        self.E = self.V*self.V
        x = np.array[np.where(np.ones((self.V,self.V)))]
        self.edges =  np.transpose(x)

    def cc(self):
        """
        Returns an array of labels corresponding to the different
        connex components of the graph.
        output:
        -label: array of shape(self.V) that labels the vertices
        """
        if self.E>0:
            label = graph_cc(self.edges[:,0],self.edges[:,1],np.zeros(self.E),self.V)
        else:
            label = np.arange(self.V)        
        return label

    def degrees(self):
        """
        returns the degree of the graph vertices
        ouput:
        - rdegree: array of shape self.V (the right degree)
        - ldegree: array of shape self.V (the left degree)
        """
        if self.E>0:
            right,left = graph_degrees(self.edges[:,0],self.edges[:,1],self.V)
        else:
            right = np.zeros(self.V,np.int)
            left = np.zeros(self.V,np.int)
        return right,left

    def main_cc(self):
        if self.E>0:
            idx = graph_main_cc(self.edges[:,0],self.edges[:,1],np.zeros(self.E),self.V)
        else:
            idx = 0     
        return idx

    def show(self,figid=-1):
        """
        show the graph as a planar graph
        INPUT:
        -figid = -1 the figure id in pylab
        by default a new figure is created
        OUTPUT:
        - figid
        """
        import matplotlib.pylab as mp
        if figid>-1:
            figid = mp.figure(int(figid))
        else:
            mp.figure()
        t = (2*np.pi*np.arange(self.V))/self.V
        mp.plot(np.cos(t),np.sin(t),'.')
        for e in range(self.E):
            A = (self.edges[e,0]*2*np.pi)/self.V
            B = (self.edges[e,1]*2*np.pi)/self.V
            mp.plot([np.cos(A),np.cos(B)],[np.sin(A),np.sin(B)],'k')
        mp.axis('off')
        return figid

        


class WeightedGraph(Graph):
    """
    This is the basic weighted, directed graph class implemented in fff 
    fields :
    - V(int) = the number of vertices
    - E(int) = the number of edges
    - edges = array of int with shape (E,2) : the edges of the graph
    - weihghts = array of int with shape (E) : the weights/length of the graph edges 
    """
    def __init__(self, V, edges=None, weights=None):
        """
        INPUT:
        V: the number of edges of the graph
        edges=None: the edge array of the graph
        weights=None: the asociated weights array
        This is the build function
        """
        V = int(V)
        if V<1:
            raise ValueError, 'cannot create graph with no vertex'
        self.V = int(V)
        self.E = 0
        if (edges==None)&(weights==None):
            edges = []
            weights = []
        else:
            if edges.shape[0]==np.size(weights):
                E = edges.shape[0]
                Graph.__init__(self, V, E)
                Graph.set_edges(self,edges)
                self.weights = weights
            else:
                raise ValueError, 'Incompatible size of the edges and weights matrices'
        
    def adjacency(self):
        """
        INPUT: None
        OUTPUT:
        -A : an ((self.V*self.V),np.double) array
        that represents the adjacency matrix of the graph
        NOTE:
        Future version should allow sparse matrix coding
        """
        A = np.zeros((self.V,self.V),np.double)
        for e in range(self.E):
            i = self.edges[e][0]
            j = self.edges[e][1]
            A[i,j] = self.weights[e]
        return(A)

    def from_adjacency(self,A):
        """
        sets the edges of self according to the adjacency matrix M
        INPUT
        M: array of size(n,n) so that n = self.V
        """
        if A.shape[0] != self.V:
            raise ValueError,"bad size for A"
        if A.shape[1] != self.V:
            raise ValueError,"bad size for A"
        
        i,j = np.where(A)
        self.edges = np.transpose(np.vstack((i,j)))
        self.weights = (A[i,j])
        self.E = np.size(i)

    def set_weights(self,weights):
        """
        INPUT:
        -weights : an array of size self.V which yields edges weights
        """
        if np.size(weights)!=self.E:
            raise ValueError, 'The weight size is not the edges size'
        else:
            self.weights = np.reshape(weights,(self.E))

    def get_weights(self):
        return self.weights

    def from_3d_grid(self,xyz,k=18):
        """
        E = self.from_3d_grid(xyz,k=18)
        set the graph to be the topological neighbours graph of the dataset xyz, in the k-connectivity scheme 
        INPUT:
        - xyz: an array of int with shape (n,3), where n=self.V
        - k = 18: the number of neighbours considered. (6,18 or 26)
        OUTPUT:
        - E: the number of edges of the graph
        """
        if xyz.shape[0]!=self.V:
            raise ValueError, 'xyz should have shape n*3, with n =self.V'
                
        if xyz.shape[1]!=3:
            raise ValueError, 'xyz should have shape n*3'
        
        graph = graph_3d_grid(xyz, k)
        if graph is not None:
            i,j,d = graph
        else:
            raise TypeError, 'Creating graph from grid failed. '\
                'Maybe the grid is too big'
        self.E = np.size(i)
        self.edges = np.zeros((self.E,2),np.int)
        self.edges[:,0] = i
        self.edges[:,1] = j
        self.weights = np.array(d)
        return self.E
        
    def complete(self):
        """
        self.complete()
        makes self a complete graph (i.e. each pair of vertices is an edge)
        """
        i,j,d = graph_complete(self.V)
        self.E = self.V*self.V
        self.edges = np.zeros((self.E,2),np.int)
        self.edges[:,0] = i
        self.edges[:,1] = j
        self.weights = np.array(d)
        
        
    def eps(self,X,eps=1.):
        """
        E = self.eps(X,eps=1.)
        set the graph to be the eps-nearest-neighbours graph of the data
        INPUT:
        - The array X is assumed to be of size (n) or (n,p) 
        where n = number of vertices of the graph 
        and p = dimension of the features 
        - eps=1. :  the neighborhood width
        OUTPUT:
        - the number of edges of the resulting graph, self.E
        NB:
        - It is assumed that the features are embedded in a
        (locally) Euclidian space 
        - trivial edges (aa) are included
        - for the sake of speed it is advisable to give a PCA-preprocessed matrix X.
        """
        if np.size(X)==X.shape[0]:
            X = np.reshape(X,(np.size(X),1))
        if X.shape[0]!=self.V:
            raise ValueError, 'X.shape[0]!=self.V'
        try:
            eps = float(eps)
        except:
            "eps cannot be cast to a float"
        if np.isnan(eps):
            raise ValueError, 'eps is nan'
        if np.isinf(eps):
            raise ValueError, 'eps is inf'
        i,j,d = graph_eps(X,eps)
        self.E = np.size(i)
        self.edges = np.zeros((self.E,2),np.int)
        self.edges[:,0] = i
        self.edges[:,1] = j
        self.weights = np.array(d)
        
        
    def knn(self,X,k=1):
        """
        E = knn(X,k)
        set the graph to be the k-nearest-neighbours graph of the data
        INPUT:
        - The array X is assumed to be of size (n) or (n,p) 
        where n = number of vertices of the graph 
        and p = dimension of the features 
        - k=1 :  is the number of neighbours considered
        OUTPUT:
        - the number of edges of the resulting graph, self.E
        NB:
        - It is assumed that the features are embedded in a
        (locally) Euclidian space 
        - the knn system is symmeterized: if (ab) is one of the edges
        then (ba) is also included
        - trivial edges (aa) are not included
        - for the sake of speed it is advisable to give a PCA-preprocessed matrix X.
        """
        if np.size(X)==X.shape[0]:
            X = np.reshape(X,(np.size(X),1))
        if X.shape[0]!=self.V:
            raise ValueError, 'X.shape[0] != self.V'
        try:
            k=int(k)
        except :
            "k cannot be cast to an int"
        if np.isnan(k):
            raise ValueError, 'k is nan'
        if np.isinf(k):
            raise ValueError, 'k is inf'
        i,j,d = graph_knn(X,k)
        self.E = np.size(i)
        self.edges = np.zeros((self.E,2),np.int)
        self.edges[:,0] = i
        self.edges[:,1] = j
        self.weights = np.array(d)
        return self.E

    def mst(self,X):
        """
        l = self.mst(X)
        maskes self the MST of the array X
        INPUT:
        - X: an array of shape (n,p) 
        where n=self.V
        and  p is the feature dimension
        OUTPUT:
        - the length of the mst
        NB:
        - It is assumed that the features are embedded in a (locally) Euclidian space
        - The edge system is symmeterized: if (ab) is one of the edges
        then (ba) is another edge
        - As a consequence, the graph comprises (2*self.V-2) edges
        - the algorithm uses Boruvska's method
        """
        if np.size(X)==X.shape[0]:
            X = np.reshape(X,(np.size(X),1))
        if X.shape[0]!=self.V:
            raise ValueError, 'X.shape[0] != self.V'
        i,j,d = graph_mst(X)
        self.E = np.size(i)
        self.edges = np.zeros((self.E,2),np.int)
        self.edges[:,0] = i
        self.edges[:,1] = j
        self.weights = np.array(d)
        return self.weights.sum()/2


    def cut_redundancies(self):
        """
        self.cut_redudancies()
        Remove possibly redundant edges: if an edge (ab) is present twice
        in the edge matrix, only the first instance in kept.
        The weights are processed accordingly
        OUPUT:
        the number of edges, self.E
        """
        if self.E>0:
            i,j,d = graph_cut_redundancies(self.edges[:,0],self.edges[:,1],self.weights,self.V)
            self.E = np.size(i)
            self.edges = np.zeros((self.E,2),np.int)
            self.edges[:,0] = i
            self.edges[:,1] = j
            self.weights = np.array(d)
        return self.E
        
    def dijkstra(self,seed=0):
        """
        dg = self.dijkstra(seed=0)
        returns all the [graph] geodesic distances starting from seed
        it is mandatory that the graph weights are non-negative
        INPUT:
        - seed is the edge from which the distances are computed
        OUTPUT:
        - the graph distance dg from the seed to any edge
        """
        try:
            if self.weights.min()<0:
                raise ValueError, 'some weights are non-positive'
        except:
             raise ValueError,'undefined weights'
        if self.E>0:
            if np.size(seed)>1:
                dg = graph_dijkstra_multiseed(self.edges[:,0],self.edges[:,1],self.weights,seed,self.V)
            else:
                dg = graph_dijkstra(self.edges[:,0],self.edges[:,1],self.weights,seed,self.V)
        else:
            dg = np.infty*np.ones(self.V,np.size(seed))
            for i in range(np.size(seed)):
                dg[seed[i],i] = 0 
        return dg

    def floyd(self,seed=None):
        """
        dg = self.floyd(seed=None):
        returns all the geodesic distances starting from seeds
        it is mandatory that the graph weights are non-negative
        INPUT:
        - seed is an arry of  edges from which the distances are computed
        if seed==None, then every edge is a seed point
        OUTPUT:
        - the graph distance dg from each seed to any edge
        Note that it has size (nbseed,nbedges)
        NB:
        - In fact the algo proceeds byr epeating dijkstra's algo for each
        seed. floyd's algo is not used (O(self.V)^3 complexity...)
        - by convention, infinte distances are coded with sum(self.wedges)+1
        """
        try:
            if self.weights.min()<0:
                raise ValueError, 'some weights are non-positive'
        except:
            raise ValueError,'undefined weights'
        if seed == None:
            seed = np.arange(self.V)
        
        if self.E>0:
            dg = graph_floyd(self.edges[:,0],self.edges[:,1],self.weights,seed,self.V)
        else:
            dg = np.infty*np.ones(self.V,np.size(seed))
            for i in range(np.size(seed)):
                dg[seed[i],i] = 0 
        return dg

    def normalize(self,c=0):
        """
        self.normalize(c=0)
        Normalize the graph according to the index c
        Normalization means that the sum of the edges values
        that go into or out each vertex must sum to 1
        INPUT:
        - c=0 is an index that designates the array
        according to which D is normalized
        It is an optional argument, by default c = 0
        c == 0 => for each vertex a, sum{edge[e,0]=a} D[e]=1
        c == 1 => for each vertex b, sum{edge[e,1]=b} D[e]=1
        c == 2 => symmetric ('l2') normalization
        NB:
        Note that when sum(edge[e,.]=a) D[e]=0, nothing is performed
        """
        c = int(c)
        if c>2:
            raise ValueError, 'c>2'
        if c<0:
            raise ValueError, 'c<0'

        if self.E==0:
            if c<2:
                return np.zeros(self.V)
            else:
                return np.zeros(self.V),np.zeros(self.V)
        
        if c<2:
            i,j,d,s = graph_normalize(self.edges[:,0],self.edges[:,1],self.weights,c,self.V)
        else:
            i,j,d,s,t = graph_normalize(self.edges[:,0],self.edges[:,1],self.weights,c,self.V)
            
        self.E = np.size(i)
        self.edges = np.zeros((self.E,2),np.int)
        self.edges[:,0] = i
        self.edges[:,1] = j
        self.weights = d

        if c<2:
            return s
        else:
            return s,t

    def reorder(self,c=0):
        """
        self.reorder(c=0)
        Reorder the graph according to the index c
        INPUT:
        - c=0 is an index that designates the array
        according to which the vectors are jointly reordered
        c == 0 => reordering makes edges[:,0] increasing, and edges[:,1] increasing for  edges[:,0] fixed
        c == 1 => reordering makes edges[:,1] increasing, and edges[:,0] increasing for  edges[:,1] fixed
        c == 2 => reordering makes weights increasing
        """
        c = int(c)
        if c>2:
            raise ValueError, 'c>2'
        if c<0:
            raise ValueError, 'c<0'
        if self.E>0:
            i,j,d = graph_reorder(self.edges[:,0],self.edges[:,1],self.weights,c,self.V)
            self.E = np.size(i)
            self.edges = np.zeros((self.E,2),np.int)
            self.edges[:,0] = i
            self.edges[:,1] = j
            self.weights = d

    def set_euclidian(self,X):
        """
        self.set_euclidian(X)
        Compute the weights of the graph in an Euclidian space
        INPUT:
        X is the coordinate matrix of the vertices
        it is assumed to be of size (self.V,p)
        """
        if np.size(X)==X.shape[0]:
            X = np.reshape(X,(np.size(X),1))
        if X.shape[0]!=self.V:
            raise ValueError, 'X.shape[0] != self.V'
        if self.E>0:
            d = graph_set_euclidian(self.edges[:,0],self.edges[:,1],X)
        self.weights = d

    def set_gaussian(self,X,sigma = 0):
        """
        self.set_gaussian(X,sigma = 0)
        Compute the weights  of the graph as a gaussian function of their length
        INPUT:
        - X is the coordinate matrix of the vertices
        it is assumed to be of size (self.V,p)
        - sigma = 0 : the parameter of the gaussian function
        NB = 
        when sigma = 0, an empirical value is used :
        sigma = sqrt(mean(||X[self.edges[:,0],:]-X[self.edges[:,1],:]||^2))
        """
        sigma = float(sigma)
        if sigma<0:
            raise ValueError, 'sigma<0'
        if np.size(X)==X.shape[0]:
            X = np.reshape(X,(np.size(X),1))
        if X.shape[0]!=self.V:
            raise ValueError, 'X.shape[0] != self.V'
        if self.E>0:
            d = graph_set_gaussian(self.edges[:,0],self.edges[:,1],X,sigma)
        self.weights = d

    def symmeterize(self):
        """
        self.symmeterize()
        symmeterize the graphself , ie produces the graph
        whose adjacency matrix would be the symmetric part of
        its current adjacency matrix 
        """
        if self.E>0:
            i,j,d = graph_symmeterize(self.edges[:,0],self.edges[:,1],self.weights,self.V)
            self.E = np.size(i)
            self.edges = np.zeros((self.E,2),np.int)
            self.edges[:,0] = i
            self.edges[:,1] = j
            self.weights = d
        return self.E

    def anti_symmeterize(self):
        """
        self.anti_symmeterize()
        anti-symmeterize the self , ie produces the graph
        whose adjacency matrix would be the antisymmetric part of
        its current adjacency matrix
        """
        #A = self.adjacency()
        #A = A-np.transpose(A)
        #self.from_adjacency(A)
        if self.E>0:
            i,j,d = graph_antisymmeterize(self.edges[:,0],self.edges[:,1],self.weights,self.V)
            self.E = np.size(i)
            self.edges = np.zeros((self.E,2),np.int)
            self.edges[:,0] = i
            self.edges[:,1] = j
            self.weights = d
        return self.E

    def to_neighb(self):
        """
        ci, ne, we = self.graph_to_neighb()
        converts the graph to a neighboring system
        The neighboring system is nothing but a (sparser) representation of the edge matrix
        OUTPUT:
        The arrays ci, ne ,we such that self.edges, self.weights 
        is coded such that:
        for j in [ci[a] ci[a+1][, there exists en edge e so that
        (edge[e,0]=a,edge[e,1]=ne[j],self.weights[e] = we[j]) 
        """
        if self.E>0:
            ci,ne,we = graph_to_neighb(self.edges[:,0],self.edges[:,1],self.weights,self.V)
        else:
            ci = []
            ne = []
            we = []
        return ci,ne,we

    def Voronoi_Labelling(self,seed):
        """
        label = self.Voronoi_Labelling(seed)
        performs a voronoi labelling of the graph
        INPUT:
        - seed is an array of vertices from which the cells are built
        OUTPUT:
        - the labelling of the vertices
        """
        if np.size(seed)==0:
            raise ValueError, 'empty seed'
        if seed.max()>self.V-1:
            raise ValueError, 'seed.max()>self.V-1'
        labels = -np.ones(self.V,np.int)
        labels[seed] = np.arange(np.size(seed))
        if self.E>0:
            labels = graph_voronoi(self.edges[:,0],self.edges[:,1],self.weights,seed,self.V)
        
        return labels
 
    def cliques(self):
        """
        cliques = self.cliques()
        OUTPUT:
        a labelling of the vertices according to the clique they belong to
        the cliques are defined using replicator dynamics equations
        """
        cliques = np.arange(self.V)
        if self.E>0:
            cliques = graph_rd(self.edges[:,0],self.edges[:,1],self.weights,self.V)
        return cliques

    def remove_trivial_edges(self):
        """
        self.remove_trivial_edges()
        Removes trivial edges, i.e. edges that are (vv)-like
        self.weights and self.E are corrected accordingly
        OUTPUT:
        The number of edges
        """
        i = np.nonzero(self.edges[:,0]!=self.edges[:,1])[0]
        self.edges = self.edges[i,:]
        self.weights = self.weights[i]
        self.E = np.size(i)
        return self.E

    def subgraph(self,valid):
        """
        creates a subgraph with the vertices for which valid>0
        and with the correponding set of edges
        INPUT:
        - valid of shape (self.V): nonzero for vertices to be retained
        CAVEAT:
        - the vertices are renumbered as [1..p] where p = sum(valid>0)
        - when sum(valid==0) then None is returned 
        """
        if np.size(valid)!= self.V:
            raise ValueError, "incompatible size for self anf valid"

        if np.sum(valid>0)==0:
            return None
        
        if self.E>0:
            win_edges = (valid[self.edges]).min(1)>0
            edges = self.edges[win_edges,:]
            weights = self.weights[win_edges]
            renumb = np.hstack((0,np.cumsum(valid>0)))
            edges = renumb[edges]
            G = WeightedGraph(np.sum(valid>0),edges,weights)
        else:
            G = WeightedGraph(np.sum(valid)>0)
    
        return G

    def Kruskal(self):
        """
        K = self.Kruskal()
        creates the MST of self using Kruskal's algo.
        efficient is self is sparse
        NOTE:
        if self contains several connected components,
        self.Kruskal() will also retain a graph with k connected components
        """
        k = self.cc().max()+1
        E = 2*self.V-2
        V = self.V
        Kedges = np.zeros((E,2)).astype(np.int)
        Kweights = np.zeros(E)
        
        iw = np.argsort(self.weights)
        label  = np.arange(V) #(2*V-1)
        j = 0
        for i in range(V-k):
            a = self.edges[iw[j],0]
            b = self.edges[iw[j],1]
            d = self.weights[iw[j]]
            while label[a]==label[b]:
                    j = j+1
                    a = self.edges[iw[j],0]
                    b = self.edges[iw[j],1]
                    d = self.weights[iw[j]]
                    
            if label[a]!=label[b]:
                la = label[a]
                lb = label[b]
                label[label==lb] = la 
                Kedges[2*i,0] = a
                Kedges[2*i,1] = b
                Kedges[2*i+1,0] = b
                Kedges[2*i+1,1] = a
                Kweights[2*i] = d
                Kweights[2*i+1] = d 

        K = WeightedGraph(V,Kedges,Kweights)
        return K

    def Kruskal_dev(self):
        """
        K = self.Kruskal()
        creates the MST of self using Kriskal's algo.
        efficient is self is sparse
        NOTE:
        if self contains several connected components,
        self.Kruskal() will also retain a graph with k connected components
        """
        k = self.cc().max()+1
        E = 2*self.V-2
        V = self.V
        Kedges = np.zeros((E,2)).astype(np.int)
        Kweights = np.zeros(E)
        
        iw = np.argsort(self.weights)
        label  = np.arange(2*V-1)
        j = 0
        for i in range(V-k):
            a = self.edges[iw[j],0]
            b = self.edges[iw[j],1]
            d = self.weights[iw[j]]
            la = label[a]
            lb = label[b]
            while la != label[la]: la = label[la]
            while lb != label[lb]: lb = label[lb]

            while la==lb:
                j = j+1
                a = self.edges[iw[j],0]
                b = self.edges[iw[j],1]
                d = self.weights[iw[j]]
                la = label[a]
                lb = label[b]
                while la != label[la]: la = label[la]
                while lb != label[lb]: lb = label[lb]
                    
            if la!=lb:
                label[la] = V+i
                label[lb] = V+i
                Kedges[2*i,0] = a
                Kedges[2*i,1] = b
                Kedges[2*i+1,0] = b
                Kedges[2*i+1,1] = a
                Kweights[2*i] = d
                Kweights[2*i+1] = d 

        K = WeightedGraph(V,Kedges,Kweights)
        return K

    def Voronoi_diagram(self,seeds,samples):
        """
        self.Voronoi_diagram(seeds,samples)
        Defines the graph as the Voronoi diagram (VD)
        that links the seeds.
        The VD is defined using the sample points.
        INPUT:
        - seeds: array of shape (self.V,dim)
        - samples: array of shape (nsamples,dim)
        NOTE:
        - by default, the weights are a Gaussian function of the distance
        - The implementation is somewhat awkward
        """
        # checks
        if seeds.shape[0]!=self.V:
            raise ValueError,"The numberof seeds is not as expected"
        if np.size(seeds) == self.V:
            seeds = np.reshape(seeds,(np.size(seeds),1))
        if np.size(samples) == samples.shape[0]:
            samples = np.reshape(samples,(np.size(samples),1))
        if seeds.shape[1]!=samples.shape[1]:
            raise ValueError,"The seeds and samples do not belong to the same space"

        #1. define the graph knn(samples,seeds,2)
        i,j,d = graph_cross_knn(samples,seeds,2)
        
        #2. put all the pairs i the target graph
        Ns = np.shape(samples)[0]
        self.E = Ns
        self.edges = np.transpose(np.array([j[2*np.arange(Ns)],j[2*np.arange(Ns)+1]]))
        self.weights = np.ones(self.E)
                
        #3. eliminate the redundancies and set the weights
        self.cut_redundancies()
        self.symmeterize()
        self.set_gaussian(seeds)
        
                

    def show(self,X=None,figid=-1):
        """
        a = self.show(X=None)
        plots the current graph in 2D
        INPUT:
        - X=None, a set of coordinates that can be used
        to embed the vertices in 2D.
        to be useful X, should have shape((self.V,p))
        if p>2, a svd reduces X for display
        if p=3 the figure si 3d
        By default, the graph is presented on a circle
        - figid=-1: a figure id for pylab plotting
        by default, a new figure is created
        OUTPUT:
        a = figure handle
        NB:
        This should be used only for small graphs...
        """
        if np.size(self.weights)==0:
            fig = Graph.show()
            return fig
        
        WM = self.weights.max()
        import matplotlib.pylab as mp
        if figid >-1:
            fig = mp.figure(figid)
        else:
            fig = mp.figure()
        ml = 5.
        if (X==None):
            for e in range(self.E):
                A = (self.edges[e,0]*2*np.pi)/self.V
                B = (self.edges[e,1]*2*np.pi)/self.V
                C = max(1,int(self.weights[e]*ml/WM))
                mp.plot([np.cos(A),np.cos(B)],[np.sin(A),np.sin(B)],'k',linewidth=C)
            t = (2*np.pi*np.arange(self.V))/self.V
            mp.plot(np.cos(t),np.sin(t),'o',linewidth=ml)
                    
            mp.axis([-1.1,1.1,-1.1,1.1])
            return fig
            
        if (X.shape[0]!=self.V):
            raise ValueError,'X.shape(0)!=self.V'
        if np.size(X)==self.V:
            X = np.reshape(X,(self.V,1))                     

        if X.shape[1]==1:
            # plot the graph on a circle
            x = np.pi*(X-X.min())/(X.max()-X.min())   
            for e in range(self.E):
                A = x[self.edges[e,0]]
                B = x[self.edges[e,1]]
                C = max(1,int(self.weights[e]*ml/WM))
                mp.plot([np.cos(A),np.cos(B)],[np.sin(A),np.sin(B)],'k',linewidth=C)
                        
            mp.plot(np.cos(x),np.sin(x),'o',linewidth=ml)
            mp.axis([-1.1,1.1,-0.1,1.1])

        if X.shape[1]>3:
            Y = X.copy()
            import numpy.linalg as L
            M1,M2,M3 = L.svd(Y,0)
            Y = np.dot(M1,np.diag(M2))
            Y = Y[:,:2]
        if X.shape[1]<4:
            Y = X
                    
        if Y.shape[1]==2:
            for e in range(self.E):
                A = self.edges[e,0]
                B = self.edges[e,1]
                C = max(1,int(self.weights[e]*ml/WM))
                mp.plot([Y[A,0],Y[B,0]],[Y[A,1],Y[B,1]],'k',linewidth=C)

            mp.plot(Y[:,0],Y[:,1],'o',linewidth=ml)
            xmin = Y[:,0].min()
            ymin = Y[:,1].min()
            xmax = Y[:,0].max()
            ymax = Y[:,1].max()
            xmin = 1.1*xmin-0.1*xmax
            xmax = 1.1*xmax-0.1*xmin
            ymin = 1.1*ymin-0.1*ymax
            ymax = 1.1*ymax-0.1*ymin
            mp.axis([xmin,xmax,ymin,ymax])

        if Y.shape[1]==3:
            import matplotlib.axes3d as p3
            ax = p3.Axes3D(fig)
            #ax.scatter3D(Y[:,0],Y[:,1],Y[:,2],'o',linewidth=ml)
            #ax.plot3D([Y[A,0],Y[B,0]],[Y[A,1],Y[B,1]],[Y[A,2],Y[B,2]],'k',linewidth=C)
            ax.scatter3D(Y[:,0],Y[:,1],Y[:,2])
                    
            for e in range(self.E):
                A = self.edges[e,0]
                B = self.edges[e,1]
                C = max(1,int(self.weights[e]*ml/WM))
                ax.plot3D([Y[A,0],Y[B,0]],[Y[A,1],Y[B,1]],[Y[A,2],Y[B,2]],'k',linewidth=C)
                    
        # mp.axis('off')
        mp.show()
        return fig
    
    def converse_edge(self):
        """
        Returns the index of the edge (j,i) for each edge (i,j)
        NOTE/TODO :slow and memory-consuming version
        a C implementation is necessary
        """
        ci,ne,we = self.to_neighb()
        li = self.left_incidence()
        ri = self.right_incidence()
        tag = -np.ones(self.E,np.int)
        for v in range(self.V):
            # e = (vw)
            for e in li[v]:
                w = self.edges[e,1]
                # c=(wv)
                liw = np.array(li[w])
                c = liw[self.edges[li[w],1]==v]
                tag[e]=c
            
        #for e in range(self.E):
        #    i = self.edges[e,0]
        #    j = self.edges[e,1]
        #    np.nonzero(ne[ci[i]:ci[i+1]]
        #
        #Mdx = self.adjacency().copy()
        #Mdx[self.edges[:,0],self.edges[:,1]] = np.arange(self.E)
        #tag = Mdx[self.edges[:,1],self.edges[:,0]]
        return tag

    def remove_edges(self,valid):
        """
        self.remove_edges(valid)
        Removes all the edges for which valid==0
        INPUT:
        - valid, an array of shape (self.E)
        """
        if np.size(valid)!=self.E:
            raise ValueError, "the input vector does not have the correct size"
        valid = np.reshape(valid,np.size(valid))
        self.E = int(valid.sum())
        self.edges = self.edges[valid!=0,:]
        self.weights = self.weights[valid!=0]

    def list_of_neighbors(self):
        """
        ln = self.list_of_neighbors()
        returns the set of neighbors as a python list
        """
        ci,ne,we = self.to_neighb()
        ln = [[ne[ci[i]:ci[i+1]]] for i in range(self.V)]
        return ln
        
    def copy(self):
        """
        G = self.copy()
        returns a copy of self
        """
        G = WeightedGraph(self.V,self.edges.copy(),self.weights.copy())
        return G

    def skeleton(self):
        """
        G = self.skeleton()
        returns a MST that based on self.weights
        Caveat: self must be connected
        """
        # check that self is connected
        u = self.cc()
        if u.max()>0:
            raise ValueError, "cannot create the skeleton for unconnected graphs"
        i,j,d = graph_skeleton(self.edges[:,0],self.edges[:,1],self.weights,self.V)
        E = np.size(i)
        edges = np.zeros((E,2),np.int)
        edges[:,0] = i
        edges[:,1] = j
        weights = np.array(d)
        G = WeightedGraph(self.V,edges,weights)
        return G

    def left_incidence(self):
        """
        return the left incidence matrix of self
        as a list of lists:
        i.e. the list[[e.0.0,..,e.0.i(0)],..,[e.V.0,E.V.i(V)]]
        where e.i.j is the set of edge indexes so that
        e.i.j[0] = i
        """
        linc = []
        for i in range(self.V):
            linc.append([])
        for e in range(self.E):
            i = self.edges[e,0]
            a = linc[i]
            a.append(e)
        return linc

    def right_incidence(self):
        """
        return the right incidence matrix of self
        as a list of lists:
        i.e. the list[[e.0.0,..,e.0.i(0)],..,[e.V.0,E.V.i(V)]]
        where e.i.j is the set of edge indexes so that
        e.i.j[1] = i
        """
        rinc = []
        for i in range(self.V):
            rinc.append([])
        for e in range(self.E):
            i = self.edges[e,1]
            a = rinc[i]
            a.append(e)
        return rinc

    def is_connected(self):
        """
        States whether self is connected or not
        """
        if self.V<1:
            raise ValueError, "empty graph"
        if self.V<2:
            return True
        if self.E==0:
            return False
        b = graph_is_connected(self.edges[:,0],self.edges[:,1],self.weights,self.V)
        if b==-1:
            raise ValueError, "problem in the c function"
        
        return int(b)

    def WeightedDegree(self,c):
        """
        returns the sum of weighted degree of graph self
        INPUT:
        - c (int)
        if c==0 considering left side
        if c==1 considering right side of the edges
        OUTPUT:
        - wd : array of shape (self.V): the resulting weighted degree
        NOTE:inefficient code
        """
        if c==0:
            mlist = self.left_incidence()
        else:
            mlist = self.right_incidence()
        w = self.get_weights()
        wd = [np.sum(w[n]) for n in mlist]
        wd = np.array(wd)
        return wd



    
class BipartiteGraph(WeightedGraph):
    """
    This is a bipartite graph structure, i.e.
    a graph there are two types of nodes, such that
    edges can exist only between nodes of type 1 and type 2
    (not within)
    fields:
    - V (int,>0) the number of type 1 vertices
    - W (int,>0) the number of type 2 vertices
    - E : (int) the number of edges
    - edges: array of shape (self.E,2) reprensenting pairwise neighbors
    - weights, array of shape (self.E), +1/-1 for scending/descending links 
    """

    def __init__(self, V,W, edges=None, weights=None):
        """
        INPUT:
        V: the number of edges of the graph
        edges=None: the edge array of the graph
        weights=None: the asociated weights array
        This is the build function
        """
        V = int(V)
        W = int(W)
        if (V<1) or (W<1):
            raise ValueError, 'cannot create graph with no vertex'
        self.V = V
        self.W = W
        
        self.E = 0
        if (edges==None)&(weights==None):
            self.edges = np.array([],np.int)
            self.weights = np.array([])
        else:
            if edges.shape[0]==np.size(weights):
                E = edges.shape[0]
                self.E = E
                self.edges = -np.ones((E,2),np.int)
                self.set_edges(edges)
                #print np.shape(weights),self.E
                WeightedGraph.set_weights(self,weights)
            else:
                raise ValueError, 'Incompatible size of the edges and weights matrices'

    def set_edges(self,edges):
        """
        sets self.edges=edges if
        1. edges has a correct size
        2. edges take values in [0..V-1]*[0..W-1]
        """
        if np.shape(edges)!=np.shape(self.edges):
            raise ValueError, 'Incompatible size of the edge matrix'
        
        if np.size(edges)>0:
            if edges.max(0)[0]+1>self.V:
                raise ValueError, 'Incorrect edge specification'
            if edges.max(0)[1]+1>self.W:
                raise ValueError, 'Incorrect edge specification'
        self.edges = edges

    def check_feature_matrices(self,X,Y):
        """
        self.check_feature_matrices(self,X,Y)
        checks wether the dismension of X and Y is coherent with self
        and possibly reshape it
        INPUT:
        - The arrays X,Y is assumed to be of size (self.V) or (self.V,p)
        and (self.W) or (self.W,p) respectively
        where p = dimension of the features
        """
        if np.size(X)==X.shape[0]:
            X = np.reshape(X,(np.size(X),1))
        if np.size(Y)==Y.shape[0]:
            Y = np.reshape(Y,(np.size(Y),1))  
        if X.shape[1]!=Y.shape[1]:
            raise ValueError, 'X.shape[1] should = Y.shape[1]'
        if X.shape[0]!=self.V:
            raise ValueError, 'X.shape[0]!=self.V'
        if Y.shape[0]!=self.W:
            raise ValueError, 'Y.shape[0]!=self.W'

    def copy(self):
        """
        G = self.copy()
        returns a copy of self
        """
        G = BipartiteGraph(self.V,self.W,self.edges.copy(),self.weights.copy())
        return G
    

    def cross_eps(self,X,Y,eps=1.):
        """
        E = self.cross_eps(X,Y,eps=1.)
        set the graph to be the eps-neighbours graph of from X to Y
        INPUT:
        - The arrays X,Y is assumed to be of size (self.V) or (self.V,p)
        and (self.W) or (self.W,p) respectively
        where p = dimension of the features
        - eps=1 : the neighbourhood size considered
        OUTPUT:
        - the number of edges of the resulting graph, self.E
        NOTE:
        - It is assumed that the features are embedded in a
        (locally) Euclidian space 
        - for the sake of speed it is advisable to give a PCA-preprocessed matrices X and Y.
        """
        self.check_feature_matrices(X,Y)
        try:
            eps = float(eps)
        except:
            "eps cannot be cast to a float"
        if np.isnan(eps):
            raise ValueError, 'eps is nan'
        if np.isinf(eps):
            raise ValueError, 'eps is inf'
        i,j,d = graph_cross_eps(X,Y,eps)
        self.E = np.size(i)
        self.edges = np.zeros((self.E,2),np.int)
        self.edges[:,0] = i
        self.edges[:,1] = j
        self.weights = np.array(d)
        return self.E

    def cross_eps_robust(self,X,Y,eps=1.):
        """
        E = self.cross_eps(X,Y,eps=1.)
        set the graph to be the eps-neighbours graph of from X to Y
        this procedure is robust in the sense that for each row of X
        at least one matching row Y is found, even though the distance
        is greater than eps.
        INPUT:
        - The arrays X,Y is assumed to be of size (self.V) or (self.V,p)
        and (self.W) or (self.W,p) respectively
        where p = dimension of the features
        - eps=1 : the neighbourhood size considered
        OUTPUT:
        - the number of edges of the resulting graph, self.E
        NB:
        - It is assumed that the features are embedded in a
        (locally) Euclidian space 
        - for the sake of speed it is advisable to give a PCA-preprocessed matrices X and Y.
        """
        self.check_feature_matrices(X,Y)
        try:
            eps = float(eps)
        except:
            "eps cannot be cast to a float"
        if np.isnan(eps):
            raise ValueError, 'eps is nan'
        if np.isinf(eps):
            raise ValueError, 'eps is inf'
        i,j,d = graph_cross_eps_robust(X,Y,eps)
        self.E = np.size(i)
        self.edges = np.zeros((self.E,2),np.int)
        self.edges[:,0] = i
        self.edges[:,1] = j
        self.weights = np.array(d)
        return self.E
    
    def cross_knn(self,X,Y,k=1):
        """
        E = self.cross_knn(X,Y,k)
        set the graph to be the k-nearest-neighbours graph of from X to Y
        INPUT:
            - The arrays X,Y is assumed to be of size (self.V) or (self.V,p)
        and (self.W) or (self.W,p) respectively
        where p = dimension of the features
        - k=1 :  is the number of neighbours considered
        OUTPUT:
        - the number of edges of the resulting graph, self.E
        NB:
        - It is assumed that the features are embedded in a
        (locally) Euclidian space 
        - for the sake of speed it is advisable to give a PCA-preprocessed matrices X and Y.
        """
        self.check_feature_matrices(X,Y)
        try:
            k=int(k)
        except :
            "k cannot be cast to an int"
        if np.isnan(k):
            raise ValueError, 'k is nan'
        if np.isinf(k):
            raise ValueError, 'k is inf'
        i,j,d = graph_cross_knn(X,Y,k)
        self.E = np.size(i)
        self.edges = np.zeros((self.E,2),np.int)
        self.edges[:,0] = i
        self.edges[:,1] = j
        self.weights = np.array(d)
        return self.E

class Forest(WeightedGraph):
    """
    This is a Forest structure, i.e. a set of trees
    - the nodes can be segmented into trees
    - within each tree a node has one parent and children
    (hierarchical structure)
    - some of the nodes can be viewed as leaves, other as roots
    - the edges within a tree are associated with a weight:
    +1 from child to parent
    -1 from parent to child

    fields:
    - V : (int,>0) the number of vertices
    - E : (int) the number of edges
    - parents: array of shape (self.V) the parent array
    - edges: array of shape (self.E,2) reprensenting pairwise neighbors
    - weights, array of shape (self.E), +1/-1 for scending/descending links 
    - children: list of arrays that represents the childs of any node
    """
    def __init__(self, V, parents=None):
        """
        INPUT:
        V: the number of edges of the graph
        parents = None: array of shape (V) the parents of the graph
        by default, the parents are set to range(V), i.e. each  
        node is its own parent, and each node is a tree
        """
        V = int(V)
        if V<1:
            raise ValueError, 'cannot create graphs with no vertex'
        self.V = int(V)

        # define the parents
        if parents==None:
            self.parents = np.arange(self.V)
        else:
            if np.size(parents)!=V:
                raise ValueError, 'Incorrect size for parents'
            if parents.max()>self.V:
                raise ValueError, 'Incorrect value for parents' 
            
            self.parents = np.reshape(parents,self.V)

        self.define_graph_attributes()

        if self.check()==0:
            raise ValueError, 'The proposed structure is not a forest'
        self.children = []

    def define_graph_attributes(self):
        """
        define the edge and weights array
        """          
        self.edges = np.array([])
        self.weights = np.array([])
        i = np.nonzero(self.parents!=np.arange(self.V))[0]
        if np.size(i)>0:
            E1 = np.hstack((i,self.parents[i]))
            E2 = np.hstack((self.parents[i],i))
            self.edges = np.transpose(np.vstack((E1,E2)))
            self.weights = np.hstack((np.ones(np.size(i)),-np.ones(np.size(i))))
            
        self.E = np.size(self.weights)


    def compute_children(self):
        """
        self.compute_children()
        define the children list
        OUPUT:
        - children: a list of self.V lists
        that yields the children of each node 
        """
        self.children = [ np.array([]) for v in range(self.V)]
        if self.E>0:
            i = self.weights<0
            a = self.edges[i,0]
            b = self.edges[i,1]
            ci,ne,we = graph_to_neighb(a,b,np.ones(np.size(a)),self.V)
            
            self.children = [ np.array(ne[ci[v]:ci[v+1]]) for v in range(self.V)]
        
    def get_children(self,v=-1):
        """
        returns the list list of children arrays in all the forest
        if v==-1 or the children of v otherwise
        """
        v = int(v)
        if v>-1:
            if v>self.V-1:
                raise ValueError, 'the given node index is too high'
        if self.children==[]:
            self.compute_children()
        if v==-1:
            return self.children
        else:
            return self.children[v]

    def get_descendents(self,v):
        """
        returns the nodes that are children of v
        """
        v = int(v)
        if v<0:
            raise ValueError, 'the given node index is too low'
        if v>self.V-1:
            raise ValueError, 'the given node index is too high'
        if self.children==[]:
            self.compute_children()
        if len(self.children[v])==0:
            return [v]
        else:
            desc = [v]
            for w in self.children[v]:
                temp = self.get_descendents(w)
                for q in temp:
                    desc.append(q)
            
        return desc


    def check(self):
        """
        Check that the proposed  is indeed a graph, i.e. contains no loop
        NOTE : slow implementation... to be rewritten in C
        OUTPUT :
        a boolean b==0 iff there are loops, 1  otherwise
        """
        b = 1
        if self.V==1:
            return b
        for v in range(self.V):
            w = v
            q = 0
            while(self.parents[w]!=w):
                w = self.parents[w]
                if w==v:
                    b = 0
                    break
                q+=1
                if q>self.V:
                    b = 0
                    break
            if b==0: break
                    
        return b

    def isleaf(self):
        """
        returns a bool array of shape(self.V)
        so that isleaf==1 iff the node is a leaf in the forest
        (has no kids)
        """
        leaves = np.ones(self.V).astype('bool')
        if self.E>0:
            leaves[self.edges[self.weights>0,1]]=0
        return leaves

    def isroot(self):
        """
        returns a bool array of shape(self.V)
        so that isleaf==1 iff the node is a root in the forest
        i.e. : is its own parent
        """
        roots = np.array(self.parents == np.arange(self.V))
        return roots

    def subforest(self,valid):
        """
        creates a subforest with the vertices for which valid>0
        and with the correponding set of edges
        the children of deleted vertices become their own parent
        a new forest is created
        """
        if np.size(valid)!= self.V:
            raise ValueError, "incompatible size for self anf valid"

        parents = self.parents.copy()
        j = np.nonzero(valid[self.parents]==0)[0]
        parents[j] = j
        parents = parents[valid]
        renumb = np.hstack((0,np.cumsum(valid)))
        parents = renumb[parents]

        F = Forest(np.sum(valid),parents)
        #F.define_graph_attributes()
        return F

    def merge_simple_branches(self):
        """
        merge the branches of the forest that are the only child
        of the parent branch into their child 
        """
        valid = np.ones(self.V).astype('bool')
        children = self.get_children()
        for k in range(self.V):
            if np.size(children[k])==1:
                valid[k]=0
        return self.subforest(valid)
        

    def all_distances(self,seed = None):
        """
        returns all the distanceof the graph  as a tree
        dg = self.all_distances(seed=None)
        by convention infinte distances are given the distance np.infty
        """
        if self.E>0:
            w = self.weights.copy()
            self.weights = np.absolute(self.weights)
            dg = self.floyd(seed)
            dg[dg==(np.sum(self.weights)+1)] = np.infty
            self.weights = w
            return dg
        else:
            return np.array([])

    def depth_from_leaves(self):
        """
        compute a labelling of the nodes
        which is 0 for the leaves, 1 for their parents etc
        and maximal for the roots
        """
        depth = 1-self.isleaf().astype(np.int)
        depth[depth>0]=-1
        q = 0
        while np.sum(depth>-1)<self.V:
            for i in np.flatnonzero(depth==q):
                if depth[self.parents[i]]<0:
                    depth[self.parents[i]] = q+1
            q+=1

        return depth
        

    def reorder_from_leaves_to_roots(self):
        """
        reorder the tree so that the leaves come first
        then their parents and so on, and the roots are last
        the permutation necessary to apply to all vertex-based information  
        """
        depth = self.depth_from_leaves()
        order = np.argsort(depth)
        iorder = np.arange(self.V)
        for i in range(self.V): iorder[order[i]] = i 
        parents = iorder[self.parents[order]]
        self.parents = parents
        self.define_graph_attributes()
        return order
        

    def leaves_of_a_subtree(self, ids,custom=False):
        """
        tests whether the given nodes within ids represent
        all the leaves of a certain subtree of self
        if custom==true the behavior of the function is somewhat mroe bizare:
        - the different connected components are considered
        as being in a same greater tree
        - when a node has more than two subbbranches,
        any subset of these children is considered as a subtree
        """
        leaves = self.isleaf().astype('bool')
        for i in ids :
            if leaves[i]==0:
                raise ValueError, "some of the ids are not leaves"
        
        #1. find the highest node that is a common ancestor to all leaves
        com_ancestor = ids[0]
        for i in ids:
            ca = i
            dca = self.get_descendents(ca)
            while com_ancestor not in dca:
                ca = self.parents[ca]
                dca = self.get_descendents(ca)
                if (ca==self.parents[ca])&(com_ancestor not in dca):
                    ca=-1
                    break
            com_ancestor=ca
        #print com_ancestor
            
        #2. check whether all the children of this ancestor are within ids
        if com_ancestor>-1:
            st = self.get_descendents(com_ancestor)
            valid = [i in ids for i in st if leaves[i]]
            bool = (np.sum(valid)==np.size(valid))
            if custom==True:
                kids = self.get_children(com_ancestor)
                if np.size(kids)>2:
                    bool=True
                    for v in kids:
                        st = np.array(self.get_descendents(v))
                        st = st[leaves[st]]
                        if np.size(st)>1:
                            valid = [i in ids for i in st]
                            bool *= ((np.sum(valid)==np.size(valid))+np.sum(valid==0))
                            ## to be checked !!!
        else:
            if custom==False:
                st = np.squeeze(np.nonzero(leaves))
                valid = [i in ids for i in st]
                bool = (np.sum(valid)==np.size(valid))
            else:
                cc = self.cc()
                bool = True
                for i in ids:
                    st = np.squeeze(np.nonzero((cc==cc[i])*leaves))
                    #print type(st),ids
                    if np.size(st)>1:
                        valid = [i in ids for i in st]
                        bool *= (np.sum(valid)==np.size(valid))
                    else:
                        bool*= (st in ids)
        return bool

def concatenate_graphs(G1,G2):
    """
    G = concatenate(G1,G2)
    Sets G as  the concatenation of the graphs G1 and G2
    It is thus assumed that the vertices of G1 and G2 are isjoint sets
    INPUT:
    - G1,K2: the two graphs to be concatenated
    OUPUT
    - G
    NOTE: this implies that the vertices of G corresponding to G2
    are labeled [G1.V .. G1.V+G2.V]
    """
    V = G1.V+G2.V
    edges = np.vstack((G1.edges,G1.V+G2.edges))
    weights = np.hstack((G1.weights,G2.weights))
    G = WeightedGraph(V,edges,weights)
    return G
