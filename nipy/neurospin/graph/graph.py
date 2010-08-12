# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from _graph import __doc__, graph_cc, graph_degrees, graph_main_cc, \
            graph_3d_grid, graph_complete, graph_knn, graph_mst, \
            graph_dijkstra, graph_dijkstra_multiseed, graph_floyd, \
            graph_eps, graph_reorder, graph_cut_redundancies, \
            graph_normalize, graph_set_euclidian, graph_set_gaussian, \
            graph_voronoi, graph_symmeterize, graph_antisymmeterize, \
            graph_to_neighb, graph_cross_knn, graph_cross_eps, \
            graph_cross_eps_robust, graph_rd, graph_skeleton, \
            graph_is_connected, graph_bpmatch
import numpy as np

"""
This module implements the main graph classes of fff2
Graph: basic topological graph, i.e. vertices and edges. Not well developed
WeightedGraph (Graph): Idem plus values asociated with vertices
BipartiteGraph (WeightedGraph): Idem but the graph is Bipartite


Author: Bertrand Thirion, 2006--2009
Fixme: add graph creation routines that are more practical 
       than current procedures
       
"""

class Graph:
    """
    This is the basic topological (non-weighted) directed Graph class
    fields:

    Member variables
    ----------------
    - V (int > 0): the number of vertices
    - E (int >= 0): the number of edges

    Properties
    ----------
    - vertices (list, type=int, shape=(V,))  vertices id
    - edges (list, type=int, shape=(E,2)): edges as vertices id tuples
    
    """

    ### Constructor
    def __init__(self, V, E=0, edges=None):
        """
        Constructor

        Parameters
        ----------
        - V (int): the number of vertices
        - E (int): the number of edges
        
        """
        # deal with vertices
        self.__set_V(V)
        self.vertices = np.arange(self.V)
        
        # deal with edges
        if not isinstance(edges, None.__class__):
            self.__set_E(np.shape(edges)[0])
            self.set_edges(edges)
        else:
            self.__set_E(E)
            self.set_edges(np.zeros((self.E,2), dtype=int))
    
    
    ### Accessors
    def get_vertices(self):
        """
        To get the graph's vertices (as id)

        """
        return self.vertices
    
    def get_edges(self):
        """
        To get the graph's edges
        
        """
        try:
            temp = self.edges
        except:
            temp = []
        return temp

    def get_V(self):
        """
        To get the number of vertices in the graph
        
        """
        return self.V
    
    def get_E(self):
        """
        To get the number of edges in the graph
        
        """
        return self.E

    ### Mutators
    def __set_V(self, V):
        """
        Sets the graph's number of vertices.
        This methods is defined as private since we don't want
        the number of vertices to be modified outside the graph object methods.
        
        """
        self.V = int(V)
        if self.V < 1:
            raise ValueError, 'Empty graphs cannot be created'
    
    def __set_E(self, E):
        """
        Sets the graph's number of edges.
        This methods is defined as private since we don't want
        the number of edges to be modified outside the graph object methods.
        
        """
        self.E = int(E)
        if self.E < 0:
            self.E = 0
    
    def set_edges(self, edges):
        """
        Sets the graph's edges

        Preconditions
        -------------
        - edges has a correct size
        - edges take values in [1..V]
        
        """
        if (not isinstance(edges, None.__class__) and (edges.size != 0)):
            if ((np.shape(edges)[0] != self.E) or \
                (np.shape(edges)[1] != 2)):
                raise ValueError, 'Incompatible size of the edge matrix'
            if edges.max()+1 > self.V:
                raise ValueError, 'Incorrect edge specification'
            self.edges = edges
        else:
            self.edges = []
        

    ### Methods
    def adjacency(self):
        """
        Builds the adjacency matrix of the graph

        Todo
        ----
        Maybe can we store the adjacency matrix once computed.
        We should yet update the edges mutator to take any modification
        into account.

        """
        A = np.zeros((self.V,self.V))
        A[self.edges[:,0],self.edges[:,1]] = 1
        
        return A

    def complete(self):
        """
        Transforms the graph into a complete one.
        
        Warning
        -------
        Note that this can't be undone since it modifies the graph's edges.

        """
        V = self.get_V()
        self.E = V ** 2
        x = np.array[np.where(np.ones((V,V)))]
        self.set_edges(np.transpose(x))

    def cc(self):
        """
        Returns an array of labels corresponding to the different
        connex components of the graph.
        
        Returns
        -------
        label: array of shape(self.V), labelling of the vertices
        
        """
        if self.E > 0:
            label = graph_cc(self.edges[:,0], self.edges[:,1], 
                             np.zeros(self.E), self.V)
        else:
            label = np.arange(self.V)        
        return label

    def degrees(self):
        """
        Returns the degree of the graph vertices.
        
        Returns
        -------
        rdegree: (array, type=int, shape=(self.V,)), the right degrees
        ldegree: (array, type=int, shape=(self.V,)), the left degrees
        
        """
        if self.E > 0:
            right,left = graph_degrees(self.edges[:,0], self.edges[:,1], self.V)
        else:
            right = np.zeros(self.V, dtype=int)
            left = np.zeros(self.V, dtype=int)
        
        return right, left

    def main_cc(self):
        """
        Returns the indexes of the vertices within the main cc
        
        Returns
        -------
        idx: array of shape (sizeof main cc)
        
        """
        if self.E > 0:
            idx = graph_main_cc(self.edges[:,0], self.edges[:,1], 
                                np.zeros(self.E), self.V)
        else:
            idx = 0     
        return idx

    def show(self, ax=None):
        """
        Shows the graph as a planar one.
        
        Parameters
        ----------
        ax, axis handle
        
        Returns
        -------
        ax, axis handle
        """
        import matplotlib.pylab as plt
        
        if ax == None:
            plt.figure()
            ax = plt.subplot(1, 1, 1)

        t = (2*np.pi*np.arange(self.V)) / self.V
        plt.plot(np.cos(t), np.sin(t), '.')
        #for e in range(self.E):
        #    A = (self.edges[e,0]*2*np.pi) / self.V
        #    B = (self.edges[e,1]*2*np.pi) / self.V
        #    ax.plot([np.cos(A),np.cos(B)], [np.sin(A),np.sin(B)], 'k')
        planar_edges = np.ravel((self.edges*2*np.pi) / self.V)
        ax.plot(np.cos(planar_edges), np.sin(planar_edges), 'k')
        ax.axis('off')
        
        return ax

        
def wgraph_from_coo_matrix(x):
    """
    Instantiates a weighted graph from a (sparse) coo_matrix

    Parameters
    ----------
    x: scipy.sparse.coo_matrix instance,
       the input matrix

    Returns
    -------
    wg: WeightedGraph instance
    """
    if x.shape[0]!=x.shape[1]:
        raise ValueError, "the input coo_matrix is not square"
    i, j = x.nonzero()
    edges = np.vstack((i,j)).T
    weights = x.data
    wg = WeightedGraph(x.shape[0], edges, weights)
    return wg

def wgraph_from_adjacency(x):
    """
    Instantiates a weighted graph from a square 2D array

    Parameters
    ----------
    x: 2D array instance,
       the input matrix
    
    Returns
    -------
    wg: WeightedGraph instance
    
    """
    if x.shape[0]!=x.shape[1]:
        raise ValueError, "the input coo_matrix is not square"
    i, j = x.nonzero()
    edges = np.vstack((i, j)).T
    weights = x[i, j]
    wg = WeightedGraph(x.shape[0], edges, weights)
    return wg

    
class WeightedGraph(Graph):
    """
    This is the basic weighted, directed graph class implemented in fff 
    fields:

    Member variables
    ----------------
    - V (int): the number of vertices
    - E (int): the number of edges

    Properties
    ----------
    - vertices (list, type=int, shape=(V,)): vertices id
    - edges (list, type=int, shape=(E,2)): edges as vertices id tuples
    - weights (list, type=int, shape=(E,)): weights/lenghts of the graph's edges
               
    """
    ### Constructor
    def __init__(self, V, edges=None, weights=None):
        """
        Constructor

        Parameters
        ----------
        - V (int > 0): the number of vertices
        - edges (array, type=int, shape=(E,2)): edges of the graph
        - weights (array, type=int, shape=(E,)): weights/lenghts of the edges
    
        """
        Graph.__init__(self, V, edges=edges)

        if isinstance(weights, None.__class__):
            new_weights = []
        else:
            new_weights = weights
        self.set_weights(new_weights)
        
    def adjacency(self):
        """
        Create the adjacency matrix of self

        Returns
        -------
        A : an ((self.V*self.V),np.double) array
            adjacency matrix of the graph
        
        Caveat
        ------
        may break if self.V is large
        Future version should allow sparse matrix coding
        
        """
        A = np.zeros((self.V,self.V), dtype=np.double)
        A[self.edges[:,0],self.edges[:,1]] = self.weights
        
        return A

    def from_adjacency(self, A):
        """
        sets the edges of self according to the adjacency matrix M
        
        Parameters
        ----------
        M: array of shape(sef.V, self.V)
        
        """
        if A.shape[0] != self.V:
            raise ValueError, "bad size for A"
        if A.shape[1] != self.V:
            raise ValueError, "bad size for A"
        
        i,j = np.where(A)
        self.edges = np.transpose(np.vstack((i,j)))
        self.weights = np.ravel(A[i, j])
        self.E = np.size(i)
        

    def set_weights(self, weights):
        """
        
        Parameters
        ----------
        weights : an array of shape(self.V), edges weights
        
        """
        if np.size(weights) != self.E:
            raise ValueError, 'The weight size is not the edges size'
        else:
            self.weights = np.reshape(weights, (self.E))

    def get_weights(self):
        return self.weights

    def from_3d_grid(self, xyz, k=18):
        """
        Sets the graph to be the topological neighbours graph 
        of the three-dimensional coordinates set xyz, 
        in the k-connectivity scheme 
        
        Parameters
        ----------
        xyz: array of shape (self.V, 3) and type np.int,
        k = 18: the number of neighbours considered. (6, 18 or 26)
        
        Returns
        -------
        E(int): the number of edges of self
        
        """
        if xyz.shape[0] != self.V:
            raise ValueError, 'xyz should have shape n*3, with n =self.V'
                
        if xyz.shape[1] != 3:
            raise ValueError, 'xyz should have shape n*3'
        
        graph = graph_3d_grid(xyz, k)
        if graph is not None:
            i, j, d = graph
        else:
            raise TypeError, 'Creating graph from grid failed. '\
                'Maybe the grid is too big'
        self.E = np.size(i)
        self.edges = np.zeros((self.E,2), np.int)
        self.edges[:,0] = i
        self.edges[:,1] = j
        self.weights = np.array(d)
        return self.E
        
    def complete(self):
        """
        self.complete()
        makes self a complete graph (i.e. each pair of vertices is an edge)
        """
        i, j, d = graph_complete(self.V)
        self.E = self.V * self.V
        self.edges = np.zeros((self.E,2), np.int)
        self.edges[:,0] = i
        self.edges[:,1] = j
        self.weights = np.array(d)
        
        
    def eps(self, X, eps=1.):
        """
        Sets the graph to be the eps-nearest-neighbours graph of the data
        
        Parameters
        ----------
        X: array of shape (self.V) or (self.V,p)
          where p = dimension of the features
          data used for eps-neighbours computation
        eps=1. (float),  the neighborhood width

        Returns
        -------
        self.E the number of edges of the resulting graph
        
        Note
        ----
        It is assumed that the features are embedded in a
           (locally) Euclidian space 
        trivial edges (aa) are included
        for the sake of speed it is advisable to give 
            a PCA-preprocessed matrix X
        """
        if np.size(X) == X.shape[0]:
            X = np.reshape(X, (np.size(X),1))
        if X.shape[0] != self.V:
            raise ValueError, 'X.shape[0] != self.V'
        try:
            eps = float(eps)
        except:
            "eps cannot be cast to a float"
        if np.isnan(eps):
            raise ValueError, 'eps is nan'
        if np.isinf(eps):
            raise ValueError, 'eps is inf'
        i, j, d = graph_eps(X, eps)
        self.E = np.size(i)
        self.edges = np.zeros((self.E,2), np.int)
        self.edges[:,0] = i
        self.edges[:,1] = j
        self.weights = np.array(d)
        
        
    def knn(self, X, k=1):
        """
        E = knn(X, k)
        set the graph to be the k-nearest-neighbours graph of the data

        Parameters
        ----------
        X array of shape (self.V) or (self.V,p)
          where p = dimension of the features
          data used for eps-neighbours computation
        k=1 :  is the number of neighbours considered
        
        Returns
        -------
        - self.E (int): the number of edges of the resulting graph
        
        Note
        ----
        It is assumed that the features are embedded in a
           (locally) Euclidian space 
        the knn system is symmeterized: if (ab) is one of the edges
            then (ba) is also included
        trivial edges (aa) are not included
        for the sake of speed it is advisable to give 
            a PCA-preprocessed matrix X.
        """
        if np.size(X) == X.shape[0]:
            X = np.reshape(X, (np.size(X),1))
        if X.shape[0] != self.V:
            raise ValueError, 'X.shape[0] != self.V'
        try:
            k = int(k)
        except :
            "k cannot be cast to an int"
        if np.isnan(k):
            raise ValueError, 'k is nan'
        if np.isinf(k):
            raise ValueError, 'k is inf'
        i, j, d = graph_knn(X,k)
        self.E = np.size(i)
        self.edges = np.zeros((self.E,2), np.int)
        self.edges[:,0] = i
        self.edges[:,1] = j
        self.weights = np.array(d)
        return self.E

    def mst(self, X):
        """
        makes self the MST of the array X

        Parameters
        ----------
        X: an array of shape (self.V,dim) 
           p is the feature dimension of X
        
        Returns
        -------
        tl (float) the total length of the mst
        
        Note
        ----
        It is assumed that the features are embedded in a 
           (locally) Euclidian space
        The edge system is symmeterized: if (ab) is one of the edges
            then (ba) is another edge
        As a consequence, the graph comprises (2*self.V-2) edges
        the algorithm uses Boruvska's method
        """
        if np.size(X) == X.shape[0]:
            X = np.reshape(X, (np.size(X),1))
        if X.shape[0] != self.V:
            raise ValueError, 'X.shape[0] != self.V'
        i, j, d = graph_mst(X)
        self.E = np.size(i)
        self.edges = np.zeros((self.E,2), np.int)
        self.edges[:,0] = i
        self.edges[:,1] = j
        self.weights = np.array(d)
        return self.weights.sum() / 2


    def cut_redundancies(self):
        """
        self.cut_redudancies()
        Remove possibly redundant edges: if an edge (ab) is present twice
        in the edge matrix, only the first instance in kept.
        The weights are processed accordingly
        
        Returns
        -------
        - E(int): the number of edges, self.E
        """
        if self.E > 0:
            i, j, d = graph_cut_redundancies(self.edges[:,0], self.edges[:,1], 
                                             self.weights, self.V)
            self.E = np.size(i)
            self.edges = np.zeros((self.E,2), np.int)
            self.edges[:,0] = i
            self.edges[:,1] = j
            self.weights = np.array(d)
        return self.E
        
    def dijkstra(self, seed=0):
        """
        returns all the [graph] geodesic distances starting from seed
        it is mandatory that the graph weights are non-negative
        
        Parameters
        ----------
        seed (int, >-1,<self.V) or array of shape(p) 
             edge(s) from which the distances are computed
                
        Returns
        -------
        dg: array of shape (self.V) ,
            the graph distance dg from ant vertex to the nearest seed 
                
        Note
        ----
        it is mandatory that the graph weights are non-negative
        """
        try:
            if self.weights.min() < 0:
                raise ValueError, 'some weights are non-positive'
        except:
             raise ValueError, 'undefined weights'
        if self.E > 0:
            if np.size(seed) > 1:
                dg = graph_dijkstra_multiseed(self.edges[:,0], self.edges[:,1],
                                              self.weights, seed, self.V)
            else:
                dg = graph_dijkstra(self.edges[:,0], self.edges[:,1],
                                    self.weights, seed, self.V)
        else:
            dg = np.infty * np.ones(self.V, np.size(seed))
            for i in range(np.size(seed)):
                dg[seed[i],i] = 0 
        return dg

    def floyd(self, seed=None):
        """
        Compute all the geodesic distances starting from seeds
        it is mandatory that the graph weights are non-negative
        
        Parameters
        ----------
        seed= None: array of shape (nbseed), type np.int 
             vertex indexes from which the distances are computed
             if seed==None, then every edge is a seed point
        
        Returns
        -------
        dg array of shape (nbseed,self.V) 
                the graph distance dg from each seed to any vertex
        
        Note
        ----
        It is mandatory that the graph weights are non-negative
        The algorithm  proceeds byr epeating dijkstra's algo for each
            seed. floyd's algo is not used (O(self.V)^3 complexity...)
        By convention, infinte distances are coded with sum(self.wedges)+1
        """
        if seed == None:
            seed = np.arange(self.V)

        if self.E == 0:
            dg = np.infty * np.ones((self.V,np.size(seed)))
            for i in range(np.size(seed)):
                dg[seed[i],i] = 0 
            return dg
        
        try:
            if self.weights.min() < 0:
                raise ValueError, 'some weights are non-positive'
        except:
            raise ValueError, 'undefined weights'
       
        dg = graph_floyd(self.edges[:,0], self.edges[:,1],
                         self.weights, seed, self.V)
        return dg

    def normalize(self, c=0):
        """
        Normalize the graph according to the index c
        Normalization means that the sum of the edges values
        that go into or out each vertex must sum to 1
        
        Parameters
        ----------
        c=0 in {0,1,2}, optional: index that designates the way
            according to which D is normalized
            c == 0 => for each vertex a, sum{edge[e,0]=a} D[e]=1
            c == 1 => for each vertex b, sum{edge[e,1]=b} D[e]=1
            c == 2 => symmetric ('l2') normalization
        
        Note
        ----
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
            i,j,d,s = graph_normalize(self.edges[:,0], self.edges[:,1],
                    self.weights,c,self.V)
        else:
            i,j,d,s,t = graph_normalize(self.edges[:,0], self.edges[:,1], 
                      self.weights,c,self.V)
            
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
        Reorder the graph according to the index c
        
        Parameters
        ----------
        c=0 in {0,1,2}, index that designates the array
            according to which the vectors are jointly reordered
            c == 0 => reordering makes edges[:,0] increasing, 
                 and edges[:,1] increasing for  edges[:,0] fixed
            c == 1 => reordering makes edges[:,1] increasing, 
                 and edges[:,0] increasing for  edges[:,1] fixed
            c == 2 => reordering makes weights increasing
        """
        c = int(c)
        if c>2:
            raise ValueError, 'c>2'
        if c<0:
            raise ValueError, 'c<0'
        if self.E>0:
            i,j,d = graph_reorder(self.edges[:,0],self.edges[:,1],
                  self.weights,c,self.V)
            self.E = np.size(i)
            self.edges = np.zeros((self.E,2),np.int)
            self.edges[:,0] = i
            self.edges[:,1] = j
            self.weights = d

    def set_euclidian(self, X):
        """
        Compute the weights of the graph as the distances between the 
        corresponding rows of X, which represents an embdedding of self
        
        Parameters
        ----------
        X array of shape (self.V, edim),
          the coordinate matrix of the embedding
        """
        if np.size(X)==X.shape[0]:
            X = np.reshape(X,(np.size(X),1))
        if X.shape[0]!=self.V:
            raise ValueError, 'X.shape[0] != self.V'
        if self.E>0:
            d = graph_set_euclidian(self.edges[:,0],self.edges[:,1],X)
        self.weights = d

    def set_gaussian(self, X, sigma=0):
        """
        Compute the weights  of the graph as a gaussian function 
        of the dinstance  between the 
        corresponding rows of X, which represents an embdedding of self
        
        Parameters
        ----------
        X array of shape (self.V,dim) 
          the coordinate matrix of the embedding
        sigma=0, float : the parameter of the gaussian function
        
        Note
        ----
        when sigma = 0, the following value is used :
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
        if self.E>0:
            i,j,d = graph_antisymmeterize(self.edges[:,0],self.edges[:,1],
                    self.weights,self.V)
            self.E = np.size(i)
            self.edges = np.zeros((self.E,2),np.int)
            self.edges[:,0] = i
            self.edges[:,1] = j
            self.weights = d
        return self.E

    def to_neighb(self):
        """
        converts the graph to a neighboring system
        The neighboring system is nothing but a (sparser) 
        representation of the edge matrix
        
        Returns
        -------
        ci, ne, we: arrays of shape (self.V+1), (self.E), (self.E)  
            such that self.edges, self.weights 
            is coded such that:
            for j in [ci[a] ci[a+1][, there exists en edge e so that
            (edge[e,0]=a,edge[e,1]=ne[j],self.weights[e] = we[j]) 
        """
        if self.E>0:
            ci,ne,we = graph_to_neighb(self.edges[:,0],self.edges[:,1],
                     self.weights,self.V)
        else:
            ci = []
            ne = []
            we = []
        return ci,ne,we

    def Voronoi_Labelling(self,seed):
        """
        label = self.Voronoi_Labelling(seed)
        performs a voronoi labelling of the graph
        
        Parameters
        ----------
        seed array of shape (nseeds), type (np.int), 
             vertices from which the cells are built
        
        Returns
        -------
        - labels : array of shape (self.V) the labelling of the vertices
        
        fixme: how is dealt the case of diconnected graph ?
        """
        if np.size(seed)==0:
            raise ValueError, 'empty seed'
        if seed.max()>self.V-1:
            raise ValueError, 'seed.max()>self.V-1'
        labels = -np.ones(self.V,np.int)
        labels[seed] = np.arange(np.size(seed))
        if self.E>0:
            labels = graph_voronoi(self.edges[:,0], self.edges[:,1],
                   self.weights, seed,self.V)
        
        return labels
 
    def cliques(self):
        """
        Extraction of the graphe cliques
        these are defined using replicator dynamics equations
        
        Returns
        -------
        - cliques: array of shape (self.V), type (np.int)
          labelling of the vertices according to the clique they belong to
        """
        cliques = np.arange(self.V)
        if self.E>0:
            cliques = graph_rd(self.edges[:,0], self.edges[:,1],
                    self.weights, self.V)
        return cliques

    def remove_trivial_edges(self):
        """
        Removes trivial edges, i.e. edges that are (vv)-like
        self.weights and self.E are corrected accordingly
         
        Returns
        -------
        - self.E (int): The number of edges
        """
        i = np.nonzero(self.edges[:,0]!=self.edges[:,1])[0]
        self.edges = self.edges[i,:]
        self.weights = self.weights[i]
        self.E = np.size(i)
        return self.E

    def subgraph(self,valid):
        """
        Creates a subgraph with the vertices for which valid>0
        and with the correponding set of edges

        Parameters   
        ----------
        valid array of shape (self.V): nonzero for vertices to be retained
        
        Returns
        -------
        G WeightedGraph instance, the desired subgraph of self
                
        Note
        ----
        The vertices are renumbered as [1..p] where p = sum(valid>0)
        when sum(valid==0) then None is returned 
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
        Creates the Minimum Spanning Tree  self using Kruskal's algo.
        efficient is self is sparse
        
        Returns
        -------
        K: WeightedGraph instance
           the resulting MST 
        
        Note
        ----
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
        Creates the Minimum Spanning Tree  self using Kruskal's algo.
        efficient is self is sparse
        
        Returns
        -------
        K: WeightedGraph instance
           the resulting MST 
        
        Note
        ----
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
        Defines the graph as the Voronoi diagram (VD)
        that links the seeds.
        The VD is defined using the sample points.

        Parameters
        ----------
        seeds: array of shape (self.V,dim)
        samples: array of shape (nsamples,dim)
        
        Note
        ----
        by default, the weights are a Gaussian function of the distance
        The implementation is not optimal
        """
        # checks
        if seeds.shape[0]!=self.V:
            raise ValueError,"The numberof seeds is not as expected"
        if np.size(seeds) == self.V:
            seeds = np.reshape(seeds,(np.size(seeds),1))
        if np.size(samples) == samples.shape[0]:
            samples = np.reshape(samples,(np.size(samples),1))
        if seeds.shape[1]!=samples.shape[1]:
            raise ValueError,"The seeds and samples do not belong \
                                  to the same space"

        #1. define the graph knn(samples,seeds,2)
        i,j,d = graph_cross_knn(samples,seeds,2)
        
        #2. put all the pairs i the target graph
        Ns = np.shape(samples)[0]
        self.E = Ns
        self.edges = np.array([j[2*np.arange(Ns)],j[2*np.arange(Ns)+1]]).T
        self.weights = np.ones(self.E)
                
        #3. eliminate the redundancies and set the weights
        self.cut_redundancies()
        self.symmeterize()
        self.set_gaussian(seeds)
        
    def show(self, X=None, ax=None):
        """
        a = self.show(X=None)
        plots the current graph in 2D
        
        Parameters
        ----------
        X=None, array of shape (self.V,2) 
                a set of coordinates that can be used
                to embed the vertices in 2D.
                if X.shape[1]>2, a svd reduces X for display
                By default, the graph is presented on a circle
        ax: ax handle, optional
        
        Returns
        -------
        ax: axis handle
        
        Note
        ----
        This should be used only for small graphs.
        """
        if np.size(self.weights)==0:
            return Graph.show()
        
        wm = self.weights.max()
        import matplotlib.pylab as mp
        if ax==None:
            mp.figure()
            ax = mp.subplot(1,1,1)
            
        ml = 5.
        if (X==None):
            for e in range(self.E):
                A = (self.edges[e,0]*2*np.pi)/self.V
                B = (self.edges[e,1]*2*np.pi)/self.V
                C = max(1,int(self.weights[e]*ml/wm))
                mp.plot([np.cos(A),np.cos(B)],[np.sin(A),np.sin(B)],'k',
                        linewidth=C)
            t = (2*np.pi*np.arange(self.V))/self.V
            mp.plot(np.cos(t),np.sin(t),'o',linewidth=ml)
                    
            mp.axis([-1.1,1.1,-1.1,1.1])
            return ax
            
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
                C = max(1,int(self.weights[e]*ml/wm))
                mp.plot([np.cos(A),np.cos(B)],[np.sin(A),np.sin(B)], 
                         'k',linewidth=C)
                        
            mp.plot(np.cos(x),np.sin(x),'o',linewidth=ml)
            mp.axis([-1.1,1.1,-0.1,1.1])

        if X.shape[1]>2:
            Y = X.copy()
            import numpy.linalg as L
            M1,M2,M3 = L.svd(Y,0)
            Y = np.dot(M1,np.diag(M2))
            Y = Y[:,:1]
        if X.shape[1]<3:
            Y = X
                    
        if Y.shape[1]==2:
            for e in range(self.E):
                A = self.edges[e,0]
                B = self.edges[e,1]
                C = max(1,int(self.weights[e]*ml/wm))
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

        return ax
    
    def converse_edge(self):
        """
        Returns the index of the edge (j,i) for each edge (i,j)
        Note: a C implementation might be necessary
        """
        ci,ne,we = self.to_neighb()
        li = self.left_incidence()
        self.right_incidence()
        tag = -np.ones(self.E,np.int)
        for v in range(self.V):
            # e = (vw)
            for e in li[v]:
                w = self.edges[e,1]
                # c=(wv)
                liw = np.array(li[w])
                c = liw[self.edges[li[w],1]==v]
                tag[e]=c
        return tag

    def remove_edges(self, valid):
        """
        Removes all the edges for which valid==0

        Parameters
        ----------
        valid, an array of shape (self.E)
        """
        if np.size(valid)!=self.E:
            raise ValueError, "the input vector does not have the correct size"
        valid = np.reshape(valid,np.size(valid))
        self.E = int(valid.sum())
        self.edges = self.edges[valid!=0,:]
        self.weights = self.weights[valid!=0]

    def list_of_neighbors(self):
        """
        returns the set of neighbors of self as a list of arrays
        """
        ci,ne,we = self.to_neighb()
        ln = [[ne[ci[i]:ci[i+1]]] for i in range(self.V)]
        return ln
        
    def copy(self):
        """
        returns a copy of self
        """
        G = WeightedGraph(self.V,self.edges.copy(),self.weights.copy())
        return G

    def skeleton(self):
        """
        returns a MST that based on self.weights
        Note: self must be connected
        """
        # check that self is connected
        u = self.cc()
        if u.max()>0:
            raise ValueError, "cannot create the skeleton for \
                              unconnected graphs"
        i,j,d = graph_skeleton(self.edges[:,0],self.edges[:,1],
                                self.weights,self.V)
        E = np.size(i)
        edges = np.zeros((E,2),np.int)
        edges[:,0] = i
        edges[:,1] = j
        weights = np.array(d)
        G = WeightedGraph(self.V,edges,weights)
        return G

    def left_incidence(self):
        """
        Returns
        -------
        the left incidence matrix of self
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
        Returns
        -------
        the right incidence matrix of self
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
        b = graph_is_connected(self.edges[:,0],self.edges[:,1],
                        self.weights,self.V)
        if b==-1:
            raise ValueError, "problem in the c function"
        
        return int(b)

    def WeightedDegree(self,c):
        """
        returns the sum of weighted degree of graph self
        
        Parameters
        ----------
        c (int): side selection
          if c==0 considering left side
          if c==1 considering right side of the edges
       
       Returns
       -------
        wd : array of shape (self.V),
           the resulting weighted degree
        
        Note: slow implementation
        """
        if c==0:
            mlist = self.left_incidence()
        else:
            mlist = self.right_incidence()
        w = self.get_weights()
        wd = [np.sum(w[n]) for n in mlist]
        wd = np.array(wd)
        return wd

    def to_coo_matrix(self):
        """
        Returns
        -------
        sp: scipy.sparse matrix instance,
            that encodes the adjacency matrix of self
        """
        import scipy.sparse as sps
        i = self.edges[:,0]
        j = self.edges[:,1]
        sm = sps.coo_matrix( (self.weights, (i, j)), shape=(self.V, self.V))
        return sm
    
class BipartiteGraph(WeightedGraph):
    """
    This is a bipartite graph structure, i.e.
    a graph there are two types of nodes, such that
    edges can exist only between nodes of type 1 and type 2
    (not within)
    fields of this class:
    V (int,>0) the number of type 1 vertices
    W (int,>0) the number of type 2 vertices
    E : (int) the number of edges
    edges: array of shape (self.E,2) reprensenting pairwise neighbors
    weights, array of shape (self.E), +1/-1 for scending/descending links 
    """

    def __init__(self, V,W, edges=None, weights=None):
        """
        
        Parameters
        ----------
        V (int), the number of vertices of subset 1
        W (int), the number of vertices of subset 2
        edges=None: array of shape (self.E,2) 
                    the edge array of the graph
        weights=None: array of shape (self.E) 
                      the asociated weights array
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
                raise ValueError, 'Incompatible size of the edges and \
                                  weights matrices'

    def set_edges(self,edges):
        """
        sets self.edges=edges if
             1. edges has a correct size
             2. edges take values in [0..V-1]*[0..W-1]
        
        Parameters
        ----------
        edges: array of shape(self.E,2): set of candidate edges
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
        checks wether the dismension of X and Y is coherent with self
        and possibly reshape it

        Parameters
        ----------
        X,Y arrays of shape (self.V) or (self.V,p)
          and (self.W) or (self.W,p) respectively
          where p = common  dimension of the features
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
        returns a copy of self
        """
        G = BipartiteGraph(self.V,self.W,self.edges.copy(),
                        self.weights.copy())
        return G
    

    def cross_eps(self,X,Y,eps=1.):
        """
        set the graph to be the eps-neighbours graph of from X to Y

        Parameters
        ----------
        X,Y arrays of shape (self.V) or (self.V,p)
            and (self.W) or (self.W,p) respectively
            where p = common dimension of the features
        eps=1, float : the neighbourhood size considered
        
        Returns
        -------
        self.E (int) the number of edges of the resulting graph
        
        Note
        ----
        It is assumed that the features are embedded 
           in a (locally) Euclidian space 
        for the sake of speed it is advisable to give PCA-preprocessed 
            matrices X and Y.
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
        Set the graph to be the eps-neighbours graph of from X to Y
        this procedure is robust in the sense that for each row of X
        at least one matching row Y is found, even though the distance
        is greater than eps.
        
        Parameters
        ----------
        X,Y: arrays of shape (self.V) or (self.V,p)
             and (self.W) or (self.W,p) respectively
             where p = dimension of the features
        eps=1, float, the neighbourhood size considered
        
        Returns
        -------
        self.E (int) the number of edges of the resulting graph

        Note
        ----
        It is assumed that the features are embedded in a
           (locally) Euclidian space 
        for the sake of speed it is advisable to give
            PCA-preprocessed matrices X and Y.
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
        set the graph to be the k-nearest-neighbours graph of from X to Y

        Parameters
        ----------
        X,Y arrays of shape (self.V) or (self.V,p)
            and (self.W) or (self.W,p) respectively
            where p = dimension of the features
        k=1, int  is the number of neighbours considered
        
        Returns
        -------
        self.E, int the number of edges of the resulting graph
        
        Note
        ----
        It is assumed that the features are embedded in a
           (locally) Euclidian space 
        for the sake of speed it is advisable to give 
            PCA-preprocessed matrices X and Y.
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

    def subgraph_left(self, valid, renumb=True):
        """
        Extraction of a subgraph 
        
        Parameters
        ----------
        valid, boolean array of shape self.V
        renumb, boolean: renumbering of the (left) edges

        Return
        ------
        A new BipartiteGraph instance with only the left vertices that are True
        if sum(valid)==0, None is returned
        """
        if np.size(valid)!=self.V:
            raise ValueError, 'valid does not have the correct size'

        if np.sum(valid>0)==0:
            return None
        
        if self.E>0:
            win_edges = valid[self.edges[:,0]]
            edges = self.edges[win_edges,:]
            weights = self.weights[win_edges]
            if renumb:
                rindex = np.hstack((0,np.cumsum(valid>0)))
                edges[:,0] = rindex[edges[:,0]]
                G = BipartiteGraph(np.sum(valid),self.W, edges, weights)
            else:
                G = BipartiteGraph(self.V,self.W, edges, weights)
            
        else:
            G = self.copy()
    
        return G

    def subgraph_right(self, valid, renumb=True):
        """
        Extraction of a subgraph 
        
        Parameters
        ----------
        valid, boolean array of shape self.V
        renumb, boolean: renumbering of the (right) edges

        Return
        ------
        A new BipartiteGraph instance with only the right vertices
        that are True
        if sum(valid)==0, None is returned
        """
        if np.size(valid)!=self.V:
            raise ValueError, 'valid does not have the correct size'

        if np.sum(valid>0)==0:
            return None
        
        if self.E>0:
            win_edges = valid[self.edges[:,1]]
            edges = self.edges[win_edges,:]
            weights = self.weights[win_edges]
            if renumb:
                rindex = np.hstack((0,np.cumsum(valid>0)))
                edges[:,1] = rindex[edges[:,1]]
                G = BipartiteGraph(np.sum(valid),self.W, edges, weights)
            else:
                G = BipartiteGraph(self.V,self.W, edges, weights)
            
        else:
            G = self.copy()
    
        return G

    

def concatenate_graphs(G1,G2):
    """
    Sets G as  the concatenation of the graphs G1 and G2
    It is thus assumed that the vertices of G1 and G2 are disjoint sets
    
    Parameters
    ----------
    G1,G2: the two WeightedGraph instances  to be concatenated
    
    Returns
    -------
    G, WeightedGraph, the concatenated graph
    
    Note
    ----
    this implies that the vertices of G corresponding to G2
    are labeled [G1.V .. G1.V+G2.V]
    """
    V = G1.V+G2.V
    edges = np.vstack((G1.edges,G1.V+G2.edges))
    weights = np.hstack((G1.weights,G2.weights))
    G = WeightedGraph(V,edges,weights)
    return G

