# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This module implements the BipartiteGraph class, used to represent
weighted bipartite graph: it contains two types of vertices, say
'left' and 'right'; then edges can only exist between 'left' and
'right' vertices. For simplicity the vertices of either side are
labeled [1..V] and [1..W] respectively.

Author: Bertrand Thirion, 2006--2011
"""

import numpy as np


def check_feature_matrices(X, Y):
    """ checks wether the dismension of X and Y are consistent

    Parameters
    ----------
    X, Y arrays of shape (n1, p) and (n2, p)
    where p = common dimension of the features
    """
    if np.size(X) == X.shape[0]:
        X = np.reshape(X, (np.size(X), 1))
    if np.size(Y) == Y.shape[0]:
        Y = np.reshape(Y, (np.size(Y), 1))
    if X.shape[1] != Y.shape[1]:
        raise ValueError('X.shape[1] should = Y.shape[1]')


def bipartite_graph_from_coo_matrix(x):
    """
    Instantiates a weighted graph from a (sparse) coo_matrix

    Parameters
    ----------
    x: scipy.sparse.coo_matrix instance, the input matrix

    Returns
    -------
    bg: BipartiteGraph instance
    """
    i, j = x.nonzero()
    edges = np.vstack((i, j)).T
    weights = x.data
    wg = BipartiteGraph(x.shape[0], x.shape[1], edges, weights)
    return wg


def bipartite_graph_from_adjacency(x):
    """Instantiates a weighted graph from a square 2D array

    Parameters
    ----------
    x: 2D array instance, the input array

    Returns
    -------
    wg: BipartiteGraph instance
    """
    from scipy.sparse import coo_matrix
    return bipartite_graph_from_coo_matrix(coo_matrix(x))


def cross_eps(X, Y, eps=1.):
    """Return the eps-neighbours graph of from X to Y

    Parameters
    ----------
    X, Y arrays of shape (n1, p) and (n2, p)
    where p = common dimension of the features
    eps=1, float: the neighbourhood size considered

    Returns
    -------
    the resulting bipartite graph instance

    Notes
    -----
    for the sake of speed it is advisable to give PCA-preprocessed matrices X
    and Y.
    """
    from scipy.sparse import coo_matrix
    check_feature_matrices(X, Y)
    try:
        eps = float(eps)
    except:
        "eps cannot be cast to a float"
    if np.isnan(eps):
        raise ValueError('eps is nan')
    if np.isinf(eps):
        raise ValueError('eps is inf')
    ij = np.zeros((0, 2))
    data = np.zeros(0)
    for i, x in enumerate(X):
        dist = np.sum((Y - x) ** 2, 1)
        idx = np.asanyarray(np.where(dist < eps))
        data = np.hstack((data, dist[idx.ravel()]))
        ij = np.vstack((ij, np.hstack((
                        i * np.ones((idx.size, 1)), idx.T)))).astype(np.int)

    data = np.maximum(data, 1.e-15)
    adj = coo_matrix((data, ij.T), shape=(X.shape[0], Y.shape[0]))
    return bipartite_graph_from_coo_matrix(adj)


def cross_knn(X, Y, k=1):
    """return the k-nearest-neighbours graph of from X to Y

    Parameters
    ----------
    X, Y arrays of shape (n1, p) and (n2, p)
    where p = common dimension of the features
    eps=1, float: the neighbourhood size considered

    Returns
    -------
    BipartiteGraph instance

    Notes
    -----
    For the sake of speed it is advised to give PCA-transformed matrices X and
    Y.
    """
    from scipy.sparse import coo_matrix
    check_feature_matrices(X, Y)
    try:
        k = int(k)
    except:
        "k cannot be cast to an int"
    if np.isnan(k):
        raise ValueError('k is nan')
    if np.isinf(k):
        raise ValueError('k is inf')
    k = min(k, Y.shape[0] -1)

    ij = np.zeros((0, 2))
    data = np.zeros(0)
    for i, x in enumerate(X):
        dist = np.sum((Y - x) ** 2, 1)
        idx = np.argsort(dist)[:k]
        data = np.hstack((data, dist[idx]))
        ij = np.vstack((ij, np.hstack((
                        i * np.ones((k, 1)), np.reshape(idx, (k, 1))))))

    data = np.maximum(data, 1.e-15)
    adj = coo_matrix((data, ij.T), shape=(X.shape[0], Y.shape[0]))
    return bipartite_graph_from_coo_matrix(adj)


class BipartiteGraph(object):
    """ Bipartite graph class

    A graph for which there are two types of nodes, such that
    edges can exist only between nodes of type 1 and type 2 (not within)
    fields of this class:
    V (int, > 0) the number of type 1 vertices
    W (int, > 0) the number of type 2 vertices
    E: (int) the number of edges
    edges: array of shape (self.E, 2) reprensenting pairwise neighbors
    weights, array of shape (self.E), +1/-1 for scending/descending links
    """

    def __init__(self, V, W, edges=None, weights=None):
        """ Constructor

        Parameters
        ----------
        V (int), the number of vertices of subset 1
        W (int), the number of vertices of subset 2
        edges=None: array of shape (self.E, 2)
                    the edge array of the graph
        weights=None: array of shape (self.E)
                      the asociated weights array
        """
        V = int(V)
        W = int(W)
        if (V < 1) or (W < 1):
            raise ValueError('cannot create graph with no vertex')
        self.V = V
        self.W = W
        self.E = 0
        if (edges == None) & (weights == None):
            self.edges = np.array([], np.int)
            self.weights = np.array([])
        else:
            if edges.shape[0] == np.size(weights):
                E = edges.shape[0]
                self.E = E
                self.edges = - np.ones((E, 2), np.int)
                self.set_edges(edges)
                self.set_weights(weights)
            else:
                raise ValueError('Incompatible size of the edges and \
                                  weights matrices')

    def set_weights(self, weights):
        """ Set weights `weights` to edges

        Parameters
        ----------
        weights, array of shape(self.V): edges weights
        """
        if np.size(weights) != self.E:
            raise ValueError('The weight size is not the edges size')
        else:
            self.weights = np.reshape(weights, (self.E))

    def set_edges(self, edges):
        """ Set edges to graph

        sets self.edges=edges if
             1. edges has a correct size
             2. edges take values in [0..V-1]*[0..W-1]

        Parameters
        ----------
        edges: array of shape(self.E, 2): set of candidate edges
        """
        if np.shape(edges) != np.shape(self.edges):
            raise ValueError('Incompatible size of the edge matrix')

        if np.size(edges) > 0:
            if edges.max(0)[0] + 1 > self.V:
                raise ValueError('Incorrect edge specification')
            if edges.max(0)[1] + 1 > self.W:
                raise ValueError('Incorrect edge specification')
        self.edges = edges

    def copy(self):
        """
        returns a copy of self
        """
        G = BipartiteGraph(self.V, self.W, self.edges.copy(),
                        self.weights.copy())
        return G

    def subgraph_left(self, valid, renumb=True):
        """Extraction of a subgraph

        Parameters
        ----------
        valid, boolean array of shape self.V
        renumb, boolean: renumbering of the (left) edges

        Returns
        -------
        G : None or ``BipartiteGraph`` instance
            A new BipartiteGraph instance with only the left vertices that are
            True.  If sum(valid)==0, None is returned
        """
        if np.size(valid) != self.V:
            raise ValueError('valid does not have the correct size')

        if np.sum(valid > 0) == 0:
            return None

        if self.E > 0:
            win_edges = valid[self.edges[:, 0]]
            edges = self.edges[win_edges]
            weights = self.weights[win_edges]
            if renumb:
                rindex = np.hstack((0, np.cumsum(valid > 0)))
                edges[:, 0] = rindex[edges[:, 0]]
                G = BipartiteGraph(np.sum(valid), self.W, edges, weights)
            else:
                G = BipartiteGraph(self.V, self.W, edges, weights)

        else:
            G = self.copy()

        return G

    def subgraph_right(self, valid, renumb=True):
        """
        Extraction of a subgraph

        Parameters
        ----------
        valid : bool array of shape self.V
        renumb : bool, optional
            renumbering of the (right) edges

        Returns
        -------
        G : None or ``BipartiteGraph`` instance.
            A new BipartiteGraph instance with only the right vertices that are
            True.  If sum(valid)==0, None is returned
        """
        if np.size(valid) != self.V:
            raise ValueError('valid does not have the correct size')

        if np.sum(valid > 0) == 0:
            return None

        if self.E > 0:
            win_edges = valid[self.edges[:, 1]]
            edges = self.edges[win_edges]
            weights = self.weights[win_edges]
            if renumb:
                rindex = np.hstack((0, np.cumsum(valid > 0)))
                edges[:, 1] = rindex[edges[:, 1]]
                G = BipartiteGraph(self.V, np.sum(valid), edges, weights)
            else:
                G = BipartiteGraph(self.V, self.W, edges, weights)

        else:
            G = self.copy()

        return G
