# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This module implements two graph classes:

Graph: basic topological graph, i.e. vertices and edges. This kind of
object only has topological properties

WeightedGraph (Graph): also has a value associated with edges, called
weights, that are used in some computational procedures (e.g. path
length computation).  Importantly these objects are equivalent to
square sparse matrices, which is used to perform certain computations.

This module also provides several functions to
instantiate WeightedGraphs from data:
- k nearest neighbours (where samples are rows of a 2D-array)
- epsilon-neighbors (where sample rows of a 2D-array)
- representation of the neighbors on a 3d grid (6-, 18- and 26-neighbors)
- Minimum Spanning Tree (where samples are rows of a 2D-array)

Author: Bertrand Thirion, 2006--2011
"""
import numpy as np
from scipy.sparse import coo_matrix

class Graph(object):
    """ Basic topological (non-weighted) directed Graph class

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
            self.set_edges(np.zeros((self.E, 2), dtype=int))

    ### Accessors

    def get_vertices(self):
        """ To get the graph's vertices (as id)
        """
        return self.vertices

    def get_edges(self):
        """To get the graph's edges
        """
        try:
            temp = self.edges
        except:
            temp = []
        return temp

    def get_V(self):
        """To get the number of vertices in the graph
        """
        return self.V

    def get_E(self):
        """To get the number of edges in the graph
        """
        return self.E

    ### Mutators
    def __set_V(self, V):
        """ Sets the graph's number of vertices.
        This methods is defined as private since we don't want
        the number of vertices to be modified outside the graph object methods.
        """
        self.V = int(V)
        if self.V < 1:
            raise ValueError('Empty graphs cannot be created')

    def __set_E(self, E):
        """Sets the graph's number of edges.
        This methods is defined as private since we don't want
        the number of edges to be modified outside the graph object methods.
        """
        self.E = int(E)
        if self.E < 0:
            self.E = 0

    def set_edges(self, edges):
        """Sets the graph's edges

        Preconditions
        -------------
        - edges has a correct size
        - edges take values in [1..V]
        """
        if (not isinstance(edges, None.__class__) and (edges.size != 0)):
            if ((np.shape(edges)[0] != self.E) or (np.shape(edges)[1] != 2)):
                raise ValueError('Incompatible size of the edge matrix')
            if edges.max() + 1 > self.V:
                raise ValueError('Incorrect edge specification')
            self.edges = edges
        else:
            self.edges = []

    ### Methods

    def adjacency(self):
        """returns the adjacency matrix of the graph as a sparse coo matrix

        Returns
        -------
        adj: scipy.sparse matrix instance,
            that encodes the adjacency matrix of self
        """
        if self.E > 0:
            i = self.edges[:, 0]
            j = self.edges[:, 1]
            adj = coo_matrix((np.ones(self.E), (i, j)),
                                shape=(self.V, self.V))
        else:
            adj = coo_matrix((self.V, self.V))
        return adj

    def cc(self):
        """Compte the different connected components of the graph.

        Returns
        -------
        label: array of shape(self.V), labelling of the vertices
        """
        try:
            from scipy.sparse import cs_graph_components
            _, label = cs_graph_components(self.adjacency())
        except:
            pass
        lil = self.to_coo_matrix().tolil().rows.tolist()
        label = lil_cc(lil)
        return label

    def degrees(self):
        """Returns the degree of the graph vertices.

        Returns
        -------
        rdegree: (array, type=int, shape=(self.V,)), the right degrees
        ldegree: (array, type=int, shape=(self.V,)), the left degrees
         """
        A = self.adjacency()
        A.data = np.ones(A.nnz)
        right = np.array(A.sum(1)).ravel()
        left = np.array(A.sum(0)).ravel()
        return right, left

    def main_cc(self):
        """Returns the indexes of the vertices within the main cc

        Returns
        -------
        idx: array of shape (sizeof main cc)
        """
        if self.E > 0:
            cc = self.cc()
            pop = np.array([np.sum(cc == k) for k in np.unique(cc)])
            idx = np.nonzero(cc == pop.argmax())[0]
        else:
            idx = 0
        return idx

    def to_coo_matrix(self):
        """ Return adjacency matrix as coo sparse

        Returns
        -------
        sp: scipy.sparse matrix instance,
            that encodes the adjacency matrix of self
        """
        if self.E > 0:
            i, j = self.edges.T
            sm = coo_matrix((np.ones(self.E), (i, j)),
                                shape=(self.V, self.V))
        else:
            sm = coo_matrix((self.V, self.V))
        return sm

    def show(self, ax=None):
        """Shows the graph as a planar one.

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

        t = (2 * np.pi * np.arange(self.V)) / self.V
        plt.plot(np.cos(t), np.sin(t), '.')
        planar_edges = np.ravel((self.edges * 2 * np.pi) / self.V)
        ax.plot(np.cos(planar_edges), np.sin(planar_edges), 'k')
        ax.axis('off')
        return ax


#####################################################################
# WeightedGraph
#####################################################################


def wgraph_from_coo_matrix(x):
    """
    Instantiates a weighted graph from a (sparse) coo_matrix

    Parameters
    ----------
    x: scipy.sparse.coo_matrix instance, the input matrix

    Returns
    -------
    wg: WeightedGraph instance
    """
    if x.shape[0] != x.shape[1]:
        raise ValueError("the input coo_matrix is not square")
    i, j = x.nonzero()
    edges = np.vstack((i, j)).T
    weights = x.data
    wg = WeightedGraph(x.shape[0], edges, weights)
    return wg


def wgraph_from_adjacency(x):
    """Instantiates a weighted graph from a square 2D array

    Parameters
    ----------
    x: 2D array instance, the input array

    Returns
    -------
    wg: WeightedGraph instance
    """
    a = coo_matrix(x)
    return wgraph_from_coo_matrix(a)


def complete_graph(n):
    """ returns a complete graph with n vertices
    """
    return wgraph_from_adjacency(np.ones((n, n)))


def mst(X):
    """  Returns the WeightedGraph that is the minimum Spanning Tree of X

    Parameters
    ----------
    X: data array, of shape(n_samples, n_features)

    Returns
    -------
    the corresponding WeightedGraph instance
    """
    n = X.shape[0]
    label = np.arange(n).astype(np.int)

    edges = np.zeros((0, 2)).astype(np.int)
    # upper bound on maxdist**2
    maxdist = 4 * np.sum((X - X[0]) ** 2, 1).max()
    nbcc = n
    while nbcc > 1:
        mindist = maxdist * np.ones(nbcc)
        link = - np.ones((nbcc, 2)).astype(np.int)

        # find nearest neighbors
        for n1 in range(n):
            j = label[n1]
            newdist = np.sum((X[n1] - X) ** 2, 1)
            newdist[label == j] = maxdist
            n2 = np.argmin(newdist)
            if newdist[n2] < mindist[j]:
                mindist[j] = newdist[n2]
                link[j] = np.array([n1, n2])

        # merge nearest neighbors
        nnbcc = nbcc
        idx = np.arange(nbcc)
        for i in range(nnbcc):
            k, j = label[link[i]]
            while k > idx[k]:
                k = idx[k]
            while j > idx[j]:
                j = idx[j]
            if k != j:
                edges = np.vstack((edges, link[i],
                                   np.array([link[i, 1], link[i, 0]])))
            idx[max(j, k)] = min(j, k)
            nbcc -= 1
        # relabel the graph
        label = WeightedGraph(n, edges, np.ones(edges.shape[0])).cc()
        nbcc = label.max() + 1

    d = np.sqrt(np.sum((X[edges[:, 0]] - X[edges[:, 1]]) ** 2, 1))
    return WeightedGraph(n, edges, d)


def knn(X, k=1):
    """returns the k-nearest-neighbours graph of the data

    Parameters
    ----------
    X, array of shape (n_samples, n_features): the input data
    k, int, optional:  is the number of neighbours considered

    Returns
    -------
    the corresponding WeightedGraph instance

    Note
    ----
    The knn system is symmeterized: if (ab) is one of the edges
    then (ba) is also included
    """
    from ..utils.fast_distance import euclidean_distance

    if np.size(X) == X.shape[0]:
        X = np.reshape(X, (np.size(X), 1))
    try:
        k = int(k)
    except:
        "k cannot be cast to an int"
    if np.isnan(k):
        raise ValueError('k is nan')
    if np.isinf(k):
        raise ValueError('k is inf')
    k = min(k, X.shape[0] - 1)

    # create the distance matrix
    dist = euclidean_distance(X)
    sorted_dist = dist.copy()
    sorted_dist.sort(0)

    # neighbour system
    bool_knn = dist < sorted_dist[k + 1]
    bool_knn += bool_knn.T
    bool_knn -= np.diag(np.diag(bool_knn))
    dist *= (bool_knn > 0)
    return wgraph_from_adjacency(dist)


def eps_nn(X, eps=1.):
    """Returns the eps-nearest-neighbours graph of the data

    Parameters
    ----------
    X, array of shape (n_samples, n_features), input data
    eps, float, optional: the neighborhood width

    Returns
    -------
    the resulting graph instance
    """
    from ..utils.fast_distance import euclidean_distance
    if np.size(X) == X.shape[0]:
        X = np.reshape(X, (np.size(X), 1))
    try:
        eps = float(eps)
    except:
        "eps cannot be cast to a float"
    if np.isnan(eps):
        raise ValueError('eps is nan')
    if np.isinf(eps):
        raise ValueError('eps is inf')
    dist = euclidean_distance(X)
    dist = np.maximum(dist, 1.e-16)
    dist[dist >= eps] = 0

    # this would is just for numerical reasons
    dist -= np.diag(np.diag(dist))
    return wgraph_from_adjacency(dist)


def lil_cc(lil):
    """ Returns the connected comonents of a graph represented as a
    list of lists

    Parameters
    ----------
    lil: a list of list representing the graph neighbors

    Returns
    -------
    label a vector of shape len(lil): connected components labelling

    Note
    ----
    dramatically slow for non-sparse graphs
    """
    n = len(lil)
    visited = np.zeros(n).astype(np.int)
    label = - np.ones(n).astype(np.int)
    k = 0
    while (visited == 0).any():
        front = [np.argmin(visited)]
        while len(front) > 0:
            pivot = front.pop(0)
            if visited[pivot] == 0:
                visited[pivot] = 1
                label[pivot] = k
                front += lil[pivot]
        k += 1
    return label


def graph_3d_grid(xyz, k=18):
    """ Utility that computes the six neighbors on a 3d grid

    Parameters
    ----------
    xyz: array of shape (n_samples, 3); grid coordinates of the points
    k: neighboring system, equal to 6, 18, or 26

    Returns
    -------
    i, j, d 3 arrays of shape (E),
            where E is the number of edges in the resulting graph
            (i, j) represent the edges, d their weights
    """
    if np.size(xyz) == 0:
        return None
    lxyz = xyz - xyz.min(0)
    m = 3 * lxyz.max(0).sum() + 2

    # six neighbours
    n6 = [np.array([1, m, m ** 2]), np.array([m ** 2, 1, m]),
         np.array([m, m ** 2, 1])]

    # eighteen neighbours
    n18 = [np.array([1 + m, 1 - m, m ** 2]),
           np.array([1 + m, m - 1, m ** 2]),
           np.array([m ** 2, 1 + m, 1 - m]),
           np.array([m ** 2, 1 + m, m - 1]),
           np.array([1 - m, m ** 2, 1 + m]),
           np.array([m - 1, m ** 2, 1 + m])]

    # twenty-six neighbours
    n26 = [np.array([1 + m + m ** 2, 1 - m, 1 - m ** 2]),
           np.array([1 + m + m ** 2, m - 1, 1 - m ** 2]),
           np.array([1 + m + m ** 2, 1 - m, m ** 2 - 1]),
           np.array([1 + m + m ** 2, m - 1, m ** 2 - 1])]

    # compute the edges in each possible direction
    def create_edges(lxyz, nn, l1dist=1, left=np.array([]), right=np.array([]),
                     weights=np.array([])):
        q = 0
        for nn_row in nn:
            v1 = np.dot(lxyz, nn_row)
            o1 = np.argsort(v1)
            sv1 = v1[o1]
            nz = np.squeeze(np.nonzero(sv1[: - 1] - sv1[1:] == - l1dist))
            o1z, o1z1 = o1[nz], o1[nz + 1]
            left = np.hstack((left, o1z, o1z1))
            right = np.hstack((right, o1z1, o1z))
            q += 2 * np.size(nz)
        weights = np.hstack((weights, np.sqrt(l1dist) * np.ones(q)))
        return left, right, weights

    i, j, d = create_edges(lxyz, n6, 1.)
    if k >= 18:
        i, j, d = create_edges(lxyz, n18, 2, i, j, d)
    if k == 26:
        i, j, d = create_edges(lxyz, n26, 3, i, j, d)
    i, j = i.astype(np.int), j.astype(np.int)

    # reorder the edges to have a more standard order
    order = np.argsort(i + j * (len(i) + 1))
    i, j, d = i[order], j[order], d[order]
    return i, j, d


def wgraph_from_3d_grid(xyz, k=18):
    """Create graph as the set of topological neighbours
    of the three-dimensional coordinates set xyz,
    in the k-connectivity scheme

    Parameters
    ----------
    xyz: array of shape (nsamples, 3) and type np.int,
    k = 18: the number of neighbours considered. (6, 18 or 26)

    Returns
    -------
    the WeightedGraph instance
    """
    if xyz.shape[1] != 3:
        raise ValueError('xyz should have shape n * 3')
    if k not in [6, 18, 26]:
        raise ValueError('k should be equal to 6, 18 or 26')

    i, j, d = graph_3d_grid(xyz, k)
    edges = np.vstack((i, j)).T
    return WeightedGraph(xyz.shape[0], edges, d)


def concatenate_graphs(G1, G2):
    """Returns  the concatenation of the graphs G1 and G2
    It is thus assumed that the vertices of G1 and G2 represent disjoint sets

    Parameters
    ----------
    G1, G2: the two WeightedGraph instances  to be concatenated

    Returns
    -------
    G, WeightedGraph, the concatenated graph

    Note
    ----
    this implies that the vertices of G corresponding to G2
    are labeled [G1.V .. G1.V+G2.V]
    """
    V = G1.V + G2.V
    edges = np.vstack((G1.edges, G1.V + G2.edges))
    weights = np.hstack((G1.weights, G2.weights))
    G = WeightedGraph(V, edges, weights)
    return G


class WeightedGraph(Graph):
    """Basic weighted, directed graph class

    Member variables
    ----------------
    - V (int): the number of vertices
    - E (int): the number of edges

    Methods
    -------
    - vertices (list, type=int, shape=(V,)): vertices id
    - edges (list, type=int, shape=(E,2)): edges as vertices id tuples
    - weights (list, type=int, shape=(E,)): weights/lenghts
        of the graph's edges
    """

    ### Constructor

    def __init__(self, V, edges=None, weights=None):
        """ Constructor

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

    def set_weights(self, weights):
        """ Set edge weights

        Parameters
        ----------
        weights: array
            array shape(self.V): edges weights
        """
        if np.size(weights) != self.E:
            raise ValueError('The weight size is not the edges size')
        else:
            self.weights = np.reshape(weights, (self.E))

    def get_weights(self):
        return self.weights

    def from_3d_grid(self, xyz, k=18):
        """Sets the graph to be the topological neighbours graph
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
            raise ValueError('xyz should have shape n * 3, with n = self.V')

        if xyz.shape[1] != 3:
            raise ValueError('xyz should have shape n * 3')

        graph = graph_3d_grid(xyz, k)
        if graph is not None:
            i, j, d = graph
        else:
            raise TypeError('Creating graph from grid failed. '\
                'Maybe the grid is too big')
        self.E = np.size(i)
        self.edges = np.zeros((self.E, 2), np.int)
        self.edges[:, 0] = i
        self.edges[:, 1] = j
        self.weights = np.array(d)
        return self.E

    def cut_redundancies(self):
        """ Returns a graph with redundant edges removed:
        ecah edge (ab) is present ony once in the edge matrix:
        the correspondng weights are added.

        Returns
        -------
        the resulting WeightedGraph
        """
        A = self.to_coo_matrix().tocsr().tocoo()
        return wgraph_from_coo_matrix(A)

    def dijkstra(self, seed=0):
        """ Returns all the [graph] geodesic distances starting from seed
x
        Parameters
        ----------
        seed (int, >-1, <self.V) or array of shape(p)
             edge(s) from which the distances are computed

        Returns
        -------
        dg: array of shape (self.V),
            the graph distance dg from ant vertex to the nearest seed

        Note
        ----
        it is mandatory that the graph weights are non-negative
        """
        import heapq
        if hasattr(seed, '__iter__') == False:
            seed = [seed]
        try:
            if (self.weights < 0).any():
                raise ValueError('some weights are non-positive')
        except:
            raise ValueError('undefined weights')
        dist, active = np.inf * np.ones(self.V), np.ones(self.V)
        idx, neighb, weight = self.compact_neighb()
        dist[seed] = 0
        dg = zip(np.zeros_like(seed), seed)
        heapq.heapify(dg)
        for j in range(self.V):
            end = False
            while True:
                if len(dg) == 0:
                    end = True
                    break
                node = heapq.heappop(dg)
                if active[node[1]]:
                    break
            if end:
                break
            dwin, win = node
            active[win] = False
            # the folllowing loop might be vectorized
            l = neighb[idx[win]: idx[win + 1]]
            newdist = dwin + weight[idx[win]: idx[win + 1]]
            who = newdist < dist[l]
            for  z in zip(newdist[who], l[who]):
                heapq.heappush(dg, z)
            dist[l[who]] = newdist[who]
        return dist

    def compact_neighb(self):
        """ returns a compact representation of self

        Returns
        -------
        idx: array of of shape(self.V + 1):
             the positions where to find the neighors of each node
             within neighb and weights
        neighb: array of shape(self.E), concatenated list of neighbors
        weights: array of shape(self.E), concatenated list of weights
        """
        order = np.argsort(self.edges[:, 0] * float(self.V) + self.edges[:, 1])
        neighb = self.edges[order, 1].astype(np.int)
        weights = self.weights[order]
        degree, _ = self.degrees()
        idx = np.hstack((0, np.cumsum(degree))).astype(np.int)
        return idx, neighb, weights

    def floyd(self, seed=None):
        """ Compute all the geodesic distances starting from seeds

        Parameters
        ----------
        seed= None: array of shape (nbseed), type np.int
             vertex indexes from which the distances are computed
             if seed==None, then every edge is a seed point

        Returns
        -------
        dg array of shape (nbseed, self.V)
                the graph distance dg from each seed to any vertex

        Note
        ----
        It is mandatory that the graph weights are non-negative
        The algorithm  proceeds byr epeating dijkstra's algo for each
            seed. floyd's algo is not used (O(self.V)^3 complexity...)
        """
        if seed == None:
            seed = np.arange(self.V)
        dg = None
        for s in seed:
            if dg == None:
                dg = self.dijkstra(s)
            else:
                dg = np.vstack((dg, self.dijkstra(s)))
        return dg

    def normalize(self, c=0):
        """ Normalize the graph according to the index c
        Normalization means that the sum of the edges values
        that go into or out each vertex must sum to 1

        Parameters
        ----------
        c=0 in {0, 1, 2}, optional: index that designates the way
            according to which D is normalized
            c == 0 => for each vertex a, sum{edge[e, 0]=a} D[e]=1
            c == 1 => for each vertex b, sum{edge[e, 1]=b} D[e]=1
            c == 2 => symmetric ('l2') normalization

        Note
        ----
        Note that when sum_{edge[e, .] == a } D[e] = 0, nothing is performed
        """
        from scipy.sparse import dia_matrix
        c = int(c)
        if not c in [0, 1, 2]:
            raise ValueError('c must be equal to 0, 1 or 2')

        if self.E == 0:
            if c < 2:
                return np.zeros(self.V)
            else:
                return np.zeros(self.V), np.zeros(self.V)
        adj = self.to_coo_matrix().tocsr()
        s1 = adj.sum(0)
        s2 = adj.sum(1)
        if c == 1:
            s = dia_matrix((1. / s1, 0), shape=(self.V, self.V))
            adj = adj * s
            self.weights = wgraph_from_adjacency(adj).get_weights()
            return np.asarray(s1)
        if c == 0:
            s = dia_matrix((1. / s2.T, 0), shape=(self.V, self.V))
            adj = s * adj
            self.weights = wgraph_from_adjacency(adj).get_weights()
            return np.asarray(s2)
        if c == 2:
            s1 = dia_matrix((1. / np.sqrt(s1), 0),
                            shape=(self.V, self.V))
            s2 = dia_matrix((1. / np.sqrt(adj.sum(1)), 0),
                            shape=(self.V, self.V))
            adj = (s1 * adj) * s2
            self.weights = wgraph_from_adjacency(adj).get_weights()
            return np.asarray(s1), np.asarray(s2)

    def set_euclidian(self, X):
        """
        Compute the weights of the graph as the distances between the
        corresponding rows of X, which represents an embdedding of self

        Parameters
        ----------
        X array of shape (self.V, edim),
          the coordinate matrix of the embedding
        """
        if np.size(X) == X.shape[0]:
            X = np.reshape(X, (np.size(X), 1))
        if X.shape[0] != self.V:
            raise ValueError('X.shape[0] != self.V')
        if self.E > 0:
            d = np.sum((X[self.edges[:, 0]] - X[self.edges[:, 1]]) ** 2, 1)
        self.weights = np.sqrt(d)

    def set_gaussian(self, X, sigma=0):
        """
        Compute the weights  of the graph as a gaussian function
        of the distance  between the corresponding rows of X,
        which represents an embdedding of self

        Parameters
        ----------
        X array of shape (self.V, dim)
          the coordinate matrix of the embedding
        sigma=0, float: the parameter of the gaussian function

        Note
        ----
        when sigma = 0, the following value is used:
        sigma = sqrt(mean(||X[self.edges[:, 0], :]-X[self.edges[:, 1], :]||^2))
        """
        sigma = float(sigma)
        if sigma < 0:
            raise ValueError('sigma should be positive')
        self.set_euclidian(X)
        d = self.weights

        if sigma == 0:
            sigma = (d ** 2).mean()

        w = np.exp(- (d ** 2) / (2 * sigma))
        self.weights = w

    def symmeterize(self):
        """Symmeterize self, modify edges and weights so that
        self.adjacency becomes the symmetric part of the current
        self.adjacency.
        """
        A = self.to_coo_matrix()
        symg = wgraph_from_adjacency((A + A.T) / 2)
        self.E = symg.E
        self.edges = symg.edges
        self.weights = symg.weights
        return self

    def anti_symmeterize(self):
        """anti-symmeterize self, i.e. produces the graph
        whose adjacency matrix would be the antisymmetric part of
        its current adjacency matrix
        """
        A = self.to_coo_matrix()
        symg = wgraph_from_adjacency((A - A.T) / 2)
        self.E = symg.E
        self.edges = symg.edges
        self.weights = symg.weights
        return self.E

    def voronoi_labelling(self, seed):
        """ Performs a voronoi labelling of the graph

        Parameters
        ----------
        seed: array of shape (nseeds), type (np.int),
              vertices from which the cells are built

        Returns
        -------
        labels: array of shape (self.V) the labelling of the vertices
        """
        import heapq
        if hasattr(seed, '__iter__') == False:
            seed = [seed]
        try:
            if (self.weights < 0).any():
                raise ValueError('some weights are non-positive')
        except:
            raise ValueError('undefined weights')
        dist, active = np.inf * np.ones(self.V), np.ones(self.V)
        label = - np.ones(self.V, np.int)
        idx, neighb, weight = self.compact_neighb()
        dist[seed] = 0
        label[seed] = np.arange(len(seed))
        dg = zip(np.zeros_like(seed), seed)
        heapq.heapify(dg)
        for j in range(self.V):
            end = False
            while True:
                if len(dg) == 0:
                    end = True
                    break
                node = heapq.heappop(dg)
                if active[node[1]]:
                    break
            if end:
                break
            dwin, win = node
            active[win] = False
            # the folllowing loop might be vectorized
            for i in range(idx[win], idx[win + 1]):
                l, newdist = neighb[i], dwin + weight[i]
                if  newdist < dist[l]:
                    heapq.heappush(dg, (newdist, l))
                    dist[l] = newdist
                    label[l] = label[win]
        return label

    def cliques(self):
        """ Extraction of the graphe cliques
        these are defined using replicator dynamics equations

        Returns
        -------
        cliques: array of shape (self.V), type (np.int)
          labelling of the vertices according to the clique they belong to
        """
        if (self.weights < 0).any():
            raise ValueError('cliques definition require a positive graph')

        cliques, size = - np.ones(self.V), np.zeros(self.V)
        adj = self.to_coo_matrix()

        for k in range(self.V):
            u = cliques < 0
            w = np.zeros_like(u)
            # replicator dynamics iterations
            for q in range(self.V):
                w = u.copy()
                u = (adj * u) * w
                if u.sum() == 0:
                    break
                u /= u.sum()
                if ((w - u) ** 2).sum() < 1.e-12:
                    break

            # threshold the result
            threshold = 1. / max(2., 1. * np.sum(cliques == - 1))
            cliques[u > threshold] = k
            if np.sum(u > threshold) == 0:
                break
            size[k] = np.sum(u > threshold)
            if cliques.min() > - 1:
                break
        # sort the labels
        size = size[size > 0]
        order = np.argsort(- size)
        label = cliques.copy()
        for k, vv in enumerate(order):
            cliques[label == vv] = k
        return cliques

    def remove_trivial_edges(self):
        """ Removes trivial edges, i.e. edges that are (vv)-like
        self.weights and self.E are corrected accordingly

        Returns
        -------
        self.E (int): The number of edges
        """
        if self.E > 0:
            valid = self.edges[:, 0] != self.edges[:, 1]
            self.edges = self.edges[valid]
            self.weights = self.weights[valid]
            self.E = np.sum(valid)
        return self.E

    def subgraph(self, valid):
        """ Creates a subgraph with the vertices for which valid>0
        and with the correponding set of edges

        Parameters
        ----------
        valid, array of shape (self.V): nonzero for vertices to be retained

        Returns
        -------
        G, WeightedGraph instance, the desired subgraph of self

        Note
        ----
        The vertices are renumbered as [1..p] where p = sum(valid>0)
        when sum(valid==0) then None is returned
        """
        if np.size(valid) != self.V:
            raise ValueError("incompatible size for self anf valid")

        if np.sum(valid > 0) == 0:
            return None

        if self.E > 0:
            win_edges = (valid[self.edges]).min(1) > 0
            edges = self.edges[win_edges]
            weights = self.weights[win_edges]
            renumb = np.hstack((0, np.cumsum(valid > 0)))
            edges = renumb[edges]
            G = WeightedGraph(np.sum(valid > 0), edges, weights)
        else:
            G = WeightedGraph(np.sum(valid > 0))

        return G

    def kruskal(self):
        """ Creates the Minimum Spanning Tree of self using Kruskal's algo.
        efficient is self is sparse

        Returns
        -------
        K, WeightedGraph instance: the resulting MST

        Note
        ----
        if self contains several connected components,
        will have the same number k of connected components
        """
        k = self.cc().max() + 1
        E = 2 * self.V - 2
        V = self.V
        Kedges = np.zeros((E, 2)).astype(np.int)
        Kweights = np.zeros(E)
        iw = np.argsort(self.weights)
        label = np.arange(V)
        j = 0
        for i in range(V - k):
            a, b = self.edges[iw[j]]
            d = self.weights[iw[j]]
            while label[a] == label[b]:
                    j = j + 1
                    a, b = self.edges[iw[j]]
                    d = self.weights[iw[j]]

            if label[a] != label[b]:
                lb = label[b]
                label[label == lb] = label[a]
                Kedges[2 * i] = np.array([a, b])
                Kedges[2 * i + 1] = np.array([b, a])
                Kweights[2 * i: 2 * i + 2] = d

        K = WeightedGraph(V, Kedges, Kweights)
        return K

    def voronoi_diagram(self, seeds, samples):
        """ Defines the graph as the Voronoi diagram (VD)
        that links the seeds.
        The VD is defined using the sample points.

        Parameters
        ----------
        seeds: array of shape (self.V, dim)
        samples: array of shape (nsamples, dim)

        Note
        ----
        by default, the weights are a Gaussian function of the distance
        The implementation is not optimal
        """
        from bipartite_graph import cross_knn
        # checks
        if seeds.shape[0] != self.V:
            raise ValueError("The numberof seeds is not as expected")
        if np.size(seeds) == self.V:
            seeds = np.reshape(seeds, (np.size(seeds), 1))
        if np.size(samples) == samples.shape[0]:
            samples = np.reshape(samples, (np.size(samples), 1))
        if seeds.shape[1] != samples.shape[1]:
            raise ValueError("The seeds and samples do not belong \
                                  to the same space")

        #1. define the graph knn(samples, seeds, 2)
        j = cross_knn(samples, seeds, 2).edges[:, 1]

        #2. put all the pairs i the target graph
        Ns = np.shape(samples)[0]
        self.E = Ns
        self.edges = np.array(
            [j[2 * np.arange(Ns)], j[2 * np.arange(Ns) + 1]]).T
        self.weights = np.ones(self.E)

        #3. eliminate the redundancies and set the weights
        self.cut_redundancies()
        self.symmeterize()
        self.set_gaussian(seeds)

    def show(self, X=None, ax=None):
        """plots the current graph in 2D

        Parameters
        ----------
        X=None, array of shape (self.V, 2)
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
        if np.size(self.weights) == 0:
            return Graph.show()

        wm = self.weights.max()
        import matplotlib.pylab as mp
        if ax == None:
            mp.figure()
            ax = mp.subplot(1, 1, 1)

        ml = 5.
        if (X == None):
            for e in range(self.E):
                A = (self.edges[e, 0] * 2 * np.pi) / self.V
                B = (self.edges[e, 1] * 2 * np.pi) / self.V
                C = max(1, int(self.weights[e] * ml / wm))
                mp.plot([np.cos(A), np.cos(B)], [np.sin(A), np.sin(B)], 'k',
                        linewidth=C)
                t = (2 * np.pi * np.arange(self.V)) / self.V
            mp.plot(np.cos(t), np.sin(t), 'o', linewidth=ml)
            mp.axis([-1.1, 1.1, -1.1, 1.1])
            return ax

        if (X.shape[0] != self.V):
            raise ValueError('X.shape(0)!=self.V')
        if np.size(X) == self.V:
            X = np.reshape(X, (self.V, 1))
        if X.shape[1] == 1:
            # plot the graph on a circle
            x = np.pi * (X - X.min()) / (X.max() - X.min())
            for e in range(self.E):
                A = x[self.edges[e, 0]]
                B = x[self.edges[e, 1]]
                C = max(1, int(self.weights[e] * ml / wm))
                mp.plot([np.cos(A), np.cos(B)], [np.sin(A), np.sin(B)],
                         'k', linewidth=C)

            mp.plot(np.cos(x), np.sin(x), 'o', linewidth=ml)
            mp.axis([-1.1, 1.1, -0.1, 1.1])

        if X.shape[1] > 2:
            Y = X.copy()
            from numpy.linalg import svd
            M1, M2, M3 = svd(Y, 0)
            Y = np.dot(M1, np.diag(M2))
            Y = Y[:, :1]
        if X.shape[1] < 3:
            Y = X

        if Y.shape[1] == 2:
            for e in range(self.E):
                A = self.edges[e, 0]
                B = self.edges[e, 1]
                C = max(1, int(self.weights[e] * ml / wm))
                mp.plot([Y[A, 0], Y[B, 0]], [Y[A, 1], Y[B, 1]], 'k',
                        linewidth=C)

            mp.plot(Y[:, 0], Y[:, 1], 'o', linewidth=ml)
            xmin, xmax = Y[:, 0].min(), Y[:, 0].max()
            ymin, ymax = Y[:, 1].min(), Y[:, 1].max()
            xmin = 1.1 * xmin - 0.1 * xmax
            xmax = 1.1 * xmax - 0.1 * xmin
            ymin = 1.1 * ymin - 0.1 * ymax
            ymax = 1.1 * ymax - 0.1 * ymin
            mp.axis([xmin, xmax, ymin, ymax])

        return ax

    def remove_edges(self, valid):
        """ Removes all the edges for which valid==0

        Parameters
        ----------
        valid, an array of shape (self.E)
        """
        if np.size(valid) != self.E:
            raise ValueError("the input vector does not have the correct size")
        valid = np.reshape(valid, np.size(valid))
        self.E = int(valid.sum())
        self.edges = self.edges[valid != 0]
        self.weights = self.weights[valid != 0]

    def list_of_neighbors(self):
        """ returns the set of neighbors of self as a list of arrays
        """
        return self.to_coo_matrix().tolil().rows.tolist()

    def copy(self):
        """ returns a copy of self
        """
        G = WeightedGraph(self.V, self.edges.copy(), self.weights.copy())
        return G

    def left_incidence(self):
        """ Return left incidence matrix

        Returns
        -------
        left_incid: list
            the left incidence matrix of self as a list of lists: i.e. the
            list[[e.0.0, .., e.0.i(0)], .., [e.V.0, E.V.i(V)]] where e.i.j is
            the set of edge indexes so that e.i.j[0] = i
        """
        linc = []
        for i in range(self.V):
            linc.append([])
        for e in range(self.E):
            i = self.edges[e, 0]
            a = linc[i]
            a.append(e)
        return linc

    def right_incidence(self):
        """ Return right incidence matrix

        Returns
        -------
        right_incid: list
            the right incidence matrix of self as a list of lists: i.e. the
            list[[e.0.0, .., e.0.i(0)], .., [e.V.0, E.V.i(V)]] where e.i.j is
            the set of edge indexes so that e.i.j[1] = i
        """
        rinc = []
        for i in range(self.V):
            rinc.append([])
        for e in range(self.E):
            i = self.edges[e, 1]
            a = rinc[i]
            a.append(e)
        return rinc

    def is_connected(self):
        """ States whether self is connected or not
        """
        if self.V < 1:
            raise ValueError("empty graph")
        if self.V < 2:
            return True
        if self.E == 0:
            return False
        cc = self.cc()
        return int(cc.max() == 0)

    def to_coo_matrix(self):
        """ Return adjacency matrix as coo sparse

        Returns
        -------
        sp: scipy.sparse matrix instance
            that encodes the adjacency matrix of self
        """
        if self.E > 0:
            i, j = self.edges.T
            sm = coo_matrix((self.weights, (i, j)), shape=(self.V, self.V))
        else:
            sm = coo_matrix((self.V, self.V))
        return sm
