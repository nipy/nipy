# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
These routines perform some hierrachical agglomerative clustering
of some input data. The following alternatives are proposed:
- Distance based average-link
- Similarity-based average-link
- Distance based maximum-link
- Ward's algorithm under graph constraints
- Ward's algorithm without graph constraints

In this latest version, the results are returned in a 'WeightedForest'
structure, which gives access to the clustering hierarchy, facilitates
the plot of the result etc.

For back-compatibility, *_segment versions of the algorithms have been
appended, with the old API (except the qmax parameter, which now
represents the number of wanted clusters)

Author : Bertrand Thirion,Pamela Guevara, 2006-2009
"""
from __future__ import print_function
from __future__ import absolute_import

#---------------------------------------------------------------------------
# ------ Routines for Agglomerative Hierarchical Clustering ----------------
# --------------------------------------------------------------------------

import numpy as np
from warnings import warn

from ..graph.graph import WeightedGraph
from ..graph.forest import Forest


class WeightedForest(Forest):
    """
    This is a weighted Forest structure, i.e. a tree
    - each node has one parent and children
    (hierarchical structure)
    - some of the nodes can be viewed as leaves, other as roots
    - the edges within a tree are associated with a weight:
    +1 from child to parent
    -1 from parent to child
    - additionally, the nodes have a value, which is called 'height',
    especially useful from dendrograms

    members
    -------
    V : (int, >0) the number of vertices
    E : (int) the number of edges
    parents: array of shape (self.V) the parent array
    edges: array of shape (self.E,2) reprensenting pairwise neighbors
    weights, array of shape (self.E), +1/-1 for scending/descending links
    children: list of arrays that represents the childs of any node
    height: array of shape(self.V)
    """

    def __init__(self, V, parents=None, height=None):
        """
        Parameters
        ----------
        V: the number of edges of the graph
        parents=None: array of shape (V)
                the parents of the graph
                by default, the parents are set to range(V), i.e. each
                node is its own parent, and each node is a tree
        height=None: array of shape(V)
                     the height of the nodes
        """
        V = int(V)
        if V < 1:
            raise ValueError('cannot create graphs with no vertex')
        self.V = int(V)

        # define the parents
        if parents is None:
            self.parents = np.arange(self.V)
        else:
            if np.size(parents) != V:
                raise ValueError('Incorrect size for parents')
            if parents.max() > self.V:
                raise ValueError('Incorrect value for parents')
            self.parents = np.reshape(parents, self.V)

        self.define_graph_attributes()

        if self.check() == 0:
            raise ValueError('The proposed structure is not a forest')
        self.children = []

        if height is None:
            height = np.zeros(self.V)
        else:
            if np.size(height) != V:
                raise ValueError('Incorrect size for height')
            self.height = np.reshape(height, self.V)

    def set_height(self, height=None):
        """Set the height array
        """
        if height is None:
            height = np.zeros(self.V)

        if np.size(height) != self.V:
            raise ValueError('Incorrect size for height')

        self.height = np.reshape(height, self.V)

    def get_height(self):
        """Get the height array
        """
        return self.height

    def check_compatible_height(self):
        """Check that height[parents[i]]>=height[i] for all nodes
        """
        OK = True
        for i in range(self.V):
            if self.height[self.parents[i]] < self.height[i]:
                OK = False
        return OK

    def plot(self, ax=None):
        """Plot the dendrogram associated with self
        the rank of the data in the dendogram is returned

        Parameters
        ----------
        ax: axis handle, optional

        Returns
        -------
        ax, the axis handle
        """
        import matplotlib.pylab as mp
        if self.check_compatible_height() == False:
            raise ValueError('cannot plot myself in my current state')

        n = np.sum(self.isleaf())

        # 1. find a permutation of the leaves that makes it nice
        aux = _label(self.parents)
        temp = np.zeros(self.V)
        rank = np.arange(self.V)
        temp[:n] = np.argsort(aux[:n])
        for i in range(n):
            rank[temp[i]] = i

        # 2. derive the abscissa in the dendrogram
        idx = np.zeros(self.V)
        temp = np.argsort(rank[:n])
        for i in range(n):
            idx[temp[i]] = i
        for i in range(n, self.V):
            j = np.nonzero(self.parents == i)[0]
            idx[i] = np.mean(idx[j])

        # 3. plot
        if ax is None:
            mp.figure()
            ax = mp.subplot(1, 1, 1)

        for i in range(self.V):
            h1 = self.height[i]
            h2 = self.height[self.parents[i]]
            mp.plot([idx[i], idx[i]], [h1, h2], 'k')

        ch = self.get_children()
        for i in range(self.V):
            if np.size(ch[i]) > 0:
                lidx = idx[ch[i]]
                m = lidx.min()
                M = lidx.max()
                h = self.height[i]
                mp.plot([m, M], [h, h], 'k')

        cM = 1.05 * self.height.max() - 0.05 * self.height.min()
        cm = 1.05 * self.height.min() - 0.05 * self.height.max()
        mp.axis([-1, idx.max() + 1, cm, cM])
        return ax

    def partition(self, threshold):
        """ Partition the tree according to a cut criterion
        """
        valid = self.height < threshold
        f = self.subforest(valid)
        u = f.cc()
        return u[f.isleaf()]

    def split(self, k):
        """
        idem as partition, but a number of components are supplied instead
        """
        k = int(k)
        if k > self.V:
            k = self.V
        nbcc = self.cc().max() + 1

        if k <= nbcc:
            u = self.cc()
            return u[self.isleaf()]

        sh = np.sort(self.height)
        th = sh[nbcc - k]
        u = self.partition(th)
        return u

    def plot_height(self):
        """Plot the height of the non-leaves nodes
        """
        import matplotlib.pylab as mp
        mp.figure()
        sh = np.sort(self.height[self.isleaf() == False])
        n = np.sum(self.isleaf() == False)
        mp.bar(np.arange(n), sh)

    def list_of_subtrees(self):
        """
        returns the list of all non-trivial subtrees in the graph
        Caveat: theis function assumes that the vertices are sorted in a
        way such that parent[i]>i forall i
        Only the leaves are listeed, not the subtrees themselves
        """
        lst = []
        n = np.sum(self.isleaf())
        for i in range(self.V):
            lst.append(np.array([], np.int))
        for i in range(n):
            lst[i] = np.array([i], np.int)
        for i in range(self.V - 1):
            j = self.parents[i]
            lst[j] = np.hstack((lst[i], lst[j]))

        return lst[n:self.V]


#--------------------------------------------------------------------------
#------------- Average link clustering on a graph -------------------------
# -------------------------------------------------------------------------


def fusion(K, pop, i, j, k):
    """ Modifies the graph K to merge nodes i and  j into nodes k

    The similarity values are weighted averaged, where pop[i] and pop[j]
    yield the relative weights.
    this is used in average_link_slow (deprecated)
    """
    #
    fi = float(pop[i]) / (pop[k])
    fj = 1.0 - fi
    #
    # replace i ny k
    #
    idxi = np.nonzero(K.edges[:, 0] == i)
    K.weights[idxi] = K.weights[idxi] * fi
    K.edges[idxi, 0] = k
    idxi = np.nonzero(K.edges[:, 1] == i)
    K.weights[idxi] = K.weights[idxi] * fi
    K.edges[idxi, 1] = k
    #
    # replace j by k
    #
    idxj = np.nonzero(K.edges[:, 0] == j)
    K.weights[idxj] = K.weights[idxj] * fj
    K.edges[idxj, 0] = k
    idxj = np.nonzero(K.edges[:, 1] == j)
    K.weights[idxj] = K.weights[idxj] * fj
    K.edges[idxj, 1] = k
    #
    #sum/remove double edges
    #
    #left side
    idxk = np.nonzero(K.edges[:, 0] == k)[0]
    corr = K.edges[idxk, 1]
    scorr = np.sort(corr)
    acorr = np.argsort(corr)
    for a in range(np.size(scorr) - 1):
        if scorr[a] == scorr[a + 1]:
            i1 = idxk[acorr[a]]
            i2 = idxk[acorr[a + 1]]
            K.weights[i1] = K.weights[i1] + K.weights[i2]
            K.weights[i2] = - np.inf
            K.edges[i2] = -1

    #right side
    idxk = np.nonzero(K.edges[:, 1] == k)[0]
    corr = K.edges[idxk, 0]
    scorr = np.sort(corr)
    acorr = np.argsort(corr)
    for a in range(np.size(scorr) - 1):
        if scorr[a] == scorr[a + 1]:
            i1 = idxk[acorr[a]]
            i2 = idxk[acorr[a + 1]]
            K.weights[i1] = K.weights[i1] + K.weights[i2]
            K.weights[i2] = - np.inf
            K.edges[i2] = - 1


def average_link_graph(G):
    """
    Agglomerative function based on a (hopefully sparse) similarity graph

    Parameters
    ----------
    G the input graph

    Returns
    -------
    t a weightForest structure that represents the dendrogram of the data

    CAVEAT
    ------
    In that case, the homogeneity is associated with high similarity
    (as opposed to low cost as in most clustering procedures,
    e.g. distance-based procedures).  Thus the tree is created with
    negated affinity values, in roder to respect the traditional
    ordering of cluster potentials. individual points have the
    potential (-np.inf).
    This problem is handled transparently inthe associated segment functionp.
    """
    warn('Function average_link_graph deprecated, will be removed',
         FutureWarning,
         stacklevel=2)
    # prepare a graph with twice the number of vertices
    n = G.V
    nbcc = G.cc().max() + 1
    K = WeightedGraph(2 * G.V)
    K.E = G.E
    K.edges = G.edges.copy()
    K.weights = G.weights.copy()

    parent = np.arange(2 * n - nbcc, dtype=np.int)
    pop = np.ones(2 * n - nbcc, np.int)
    height = np.inf * np.ones(2 * n - nbcc)

    # iteratively merge clusters
    for q in range(n - nbcc):

        # 1. find the heaviest edge
        m = (K.weights).argmax()
        cost = K.weights[m]
        k = q + n
        height[k] = cost
        i = K.edges[m, 0]
        j = K.edges[m, 1]

        # 2. remove the current edge
        K.edges[m] = -1
        K.weights[m] = - np.inf
        m = np.nonzero((K.edges[:, 0] == j) * (K.edges[:, 1] == i))[0]
        K.edges[m] = - 1
        K.weights[m] = - np.inf

        # 3. merge the edges with third part edges
        parent[i] = k
        parent[j] = k
        pop[k] = pop[i] + pop[j]
        fusion(K, pop, i, j, k)

    height[height < 0] = 0
    height[np.isinf(height)] = height[n] + 1
    t = WeightedForest(2 * n - nbcc, parent, - height)
    return t


def average_link_graph_segment(G, stop=0, qmax=1, verbose=False):
    """Agglomerative function based on a (hopefully sparse) similarity graph

    Parameters
    ----------
    G the input graph
    stop: float
        the stopping criterion
    qmax: int, optional
        the number of desired clusters (in the limit of the stopping criterion)
    verbose : bool, optional
        If True, print diagnostic information

    Returns
    -------
    u: array of shape (G.V)
       a labelling of the graph vertices according to the criterion
    cost: array of shape (G.V (?))
          the cost of each merge step during the clustering procedure
    """
    warn('Function average_link_graph_segment deprecated, will be removed',
         FutureWarning,
         stacklevel=2)

    # prepare a graph with twice the number of vertices
    n = G.V
    if qmax == - 1:
        qmax = n
    qmax = int(np.minimum(qmax, n))

    t = average_link_graph(G)

    if verbose:
        t.plot()

    u1 = np.zeros(n, np.int)
    u2 = np.zeros(n, np.int)
    if stop >= 0:
        u1 = t.partition( - stop)
    if qmax > 0:
        u2 = t.split(qmax)

    if u1.max() < u2.max():
        u = u2
    else:
        u = u1

    cost = - t.get_height()
    cost = cost[t.isleaf() == False]

    return u, cost


#--------------------------------------------------------------------------
#------------- Ward's algorithm with graph constraints --------------------
# -------------------------------------------------------------------------


def _inertia_(i, j, Features):
    """
    Compute the variance of the set which is
    the concatenation of Feature[i] and Features[j]
    """
    if np.size(np.shape(Features[i])) < 2:
        print(i, np.shape(Features[i]), Features[i])
    if np.size(np.shape(Features[i])) < 2:
        print(j, np.shape(Features[j]), Features[j])
    if np.shape(Features[i])[1] != np.shape(Features[j])[1]:
        print(i, j, np.shape(Features[i]), np.shape(Features[j]))
    localset = np.vstack((Features[i], Features[j]))
    return np.var(localset, 0).sum()


def _inertia(i, j, Features):
    """
    Compute the variance of the set which is
    the concatenation of Feature[i] and Features[j]
    """
    n = Features[0][i] + Features[0][j]
    s = Features[1][i] + Features[1][j]
    q = Features[2][i] + Features[2][j]
    return np.sum(q - (s ** 2 / n))


def _initial_inertia(K, Features, seeds=None):
    """ Compute the variance associated with each
    edge-related pair of vertices
    Thre sult is written in K;weights
    if seeds if provided (seeds!=None)
    this is done only for vertices adjacent to the seeds
    """
    if seeds is None:
        for e in range(K.E):
            i = K.edges[e, 0]
            j = K.edges[e, 1]
            ESS = _inertia(i, j, Features)
            K.weights[e] = ESS
    else:
        aux = np.zeros(K.V).astype('bool')
        aux[seeds] = 1
        for e in range(K.E):
            i = K.edges[e, 0]
            j = K.edges[e, 1]
            if (aux[i] or aux[j]):
                K.weights[e] = _inertia(i, j, Features)
            else:
                K.weights[e] = np.inf


def _auxiliary_graph(G, Features):
    """
    prepare a graph with twice the number of vertices
    this graph will contain the connectivity information
    along the merges.
    """
    K = WeightedGraph(2 * G.V - 1)
    K.E = G.E
    K.edges = G.edges.copy()
    K.weights = np.ones(K.E)
    K.symmeterize()
    if K.E > 0:    
        valid = K.edges[:, 0] < K.edges[:, 1]
        K.remove_edges(valid)
    #
    K.remove_trivial_edges()
    _initial_inertia(K, Features)
    return K


def _remap(K, i, j, k, Features, linc, rinc):
    """Modifies the graph K to merge nodes i and  j into nodes k
    the graph weights are modified accordingly

    Parameters
    ----------
    K graph instance:
      the existing graphical model
    i,j,k: int
           indexes of the nodes to be merged and of the parent respectively
    Features: list of node-per-node features
    linc: array of shape(K.V)
          left incidence matrix
    rinc: array of shape(K.V)
          right incidencematrix
    """
    # -------
    # replace i by k
    # --------
    idxi = np.array(linc[i]).astype(np.int)
    if np.size(idxi) > 1:
        for l in idxi:
            K.weights[l] = _inertia(k, K.edges[l, 1], Features)
    elif np.size(idxi) == 1:
        K.weights[idxi] = _inertia(k, K.edges[idxi, 1], Features)
    if np.size(idxi) > 0:
        K.edges[idxi, 0] = k

    idxi = np.array(rinc[i]).astype(np.int)
    if np.size(idxi) > 1:
        for l in idxi:
            K.weights[l] = _inertia(K.edges[l, 0], k, Features)
    elif np.size(idxi) == 1:
        K.weights[idxi] = _inertia(K.edges[idxi, 0], k, Features)
    if np.size(idxi) > 0:
        K.edges[idxi, 1] = k

    #------
    # replace j by k
    #-------
    idxj = np.array(linc[j]).astype(np.int)
    if np.size(idxj) > 1:
        for l in idxj:
            K.weights[l] = _inertia(k, K.edges[l, 1], Features)
    elif np.size(idxj) == 1:
        K.weights[idxj] = _inertia(k, K.edges[idxj, 1], Features)
    if np.size(idxj) > 0:
        K.edges[idxj, 0] = k

    idxj = np.array(rinc[j]).astype(np.int)
    if np.size(idxj) > 1:
        for l in idxj:
            K.weights[l] = _inertia(k, K.edges[l, 0], Features)
    elif np.size(idxj) == 1:
        K.weights[idxj] = _inertia(k, K.edges[idxj, 0], Features)
    if np.size(idxj) > 0:
        K.edges[idxj, 1] = k

    #------
    # update linc,rinc
    #------
    lidxk = list(np.concatenate((linc[j], linc[i])))
    for l in lidxk:
        if K.edges[l, 1] == - 1:
            lidxk.remove(l)

    linc[k] = lidxk
    linc[i] = []
    linc[j] = []
    ridxk = list(np.concatenate((rinc[j], rinc[i])))
    for l in ridxk:
        if K.edges[l, 0] == - 1:
            ridxk.remove(l)

    rinc[k] = ridxk
    rinc[i] = []
    rinc[j] = []

    #------
    #remove double edges
    #------
    #left side
    idxk = np.array(linc[k]).astype(np.int)
    if np.size(idxk) > 0:
        corr = K.edges[idxk, 1]
        scorr = np.sort(corr)
        acorr = np.argsort(corr)
        for a in range(np.size(scorr) - 1):
            if scorr[a] == scorr[a + 1]:
                i2 = idxk[acorr[a + 1]]
                K.weights[i2] = np.inf
                rinc[K.edges[i2, 1]].remove(i2)
                K.edges[i2] = - 1
                linc[k].remove(i2)

    #right side
    idxk = np.array(rinc[k]).astype(np.int)
    if np.size(idxk) > 0:
        corr = K.edges[idxk, 0]
        scorr = np.sort(corr)
        acorr = np.argsort(corr)
        for a in range(np.size(scorr) - 1):
            if scorr[a] == scorr[a + 1]:
                i2 = idxk[acorr[a + 1]]
                K.weights[i2] = np.inf
                linc[K.edges[i2, 0]].remove(i2)
                K.edges[i2] = - 1
                rinc[k].remove(i2)
    return linc, rinc


def ward_quick(G, feature, verbose=False):
    """ Agglomerative function based on a topology-defining graph
    and a feature matrix.

    Parameters
    ----------
    G : graph instance
        topology-defining graph
    feature: array of shape (G.V,dim_feature)
        some vectorial information related to the graph vertices
    verbose : bool, optional
         If True, print diagnostic information

    Returns
    -------
    t: weightForest instance,
       that represents the dendrogram of the data

    Notes
    ----
    Hopefully a quicker version

    A euclidean distance is used in the feature space

    Caveat : only approximate
    """
    warn('Function ward_quick from ' + 
         'nipy.algorithms.clustering.hierrachical_clustering ' + 
         'deprecated, will be removed',
         FutureWarning,
         stacklevel=2)

    # basic check
    if feature.ndim == 1:
        feature = np.reshape(feature, (-1, 1))

    if feature.shape[0] != G.V:
        raise ValueError(
            "Incompatible dimension for the feature matrix and the graph")

    Features = [np.ones(2 * G.V), np.zeros((2 * G.V, feature.shape[1])),
                np.zeros((2 * G.V, feature.shape[1]))]
    Features[1][:G.V] = feature
    Features[2][:G.V] = feature ** 2
    n = G.V
    nbcc = G.cc().max() + 1

    # prepare a graph with twice the number of vertices
    K = _auxiliary_graph(G, Features)
    parent = np.arange(2 * n - nbcc).astype(np.int)
    height = np.zeros(2 * n - nbcc)
    linc = K.left_incidence()
    rinc = K.right_incidence()

    # iteratively merge clusters
    q = 0
    while (q < n - nbcc):
        # 1. find the lightest edges
        aux = np.zeros(2 * n)
        ape = np.nonzero(K.weights < np.inf)
        ape = np.reshape(ape, np.size(ape))
        idx = np.argsort(K.weights[ape])

        for e in range(n - nbcc - q):
            i, j = K.edges[ape[idx[e]], 0], K.edges[ape[idx[e]], 1]
            if (aux[i] == 1) or (aux[j] == 1):
                break
            aux[i] = 1
            aux[j] = 1

        emax = np.maximum(e, 1)

        for e in range(emax):
            m = ape[idx[e]]
            cost = K.weights[m]
            k = q + n
            i = K.edges[m, 0]
            j = K.edges[m, 1]
            height[k] = cost
            if verbose:
                print(q, i, j, m, cost)

            # 2. remove the current edge
            K.edges[m] = -1
            K.weights[m] = np.inf
            linc[i].remove(m)
            rinc[j].remove(m)

            ml = linc[j]
            if np.sum(K.edges[ml, 1] == i) > 0:
                m = ml[np.flatnonzero(K.edges[ml, 1] == i)]
                K.edges[m] = -1
                K.weights[m] = np.inf
                linc[j].remove(m)
                rinc[i].remove(m)

            # 3. merge the edges with third part edges
            parent[i] = k
            parent[j] = k
            for p in range(3):
                Features[p][k] = Features[p][i] + Features[p][j]

            linc, rinc = _remap(K, i, j, k, Features, linc, rinc)
            q += 1

    # build a tree to encode the results
    t = WeightedForest(2 * n - nbcc, parent, height)
    return t


def ward_field_segment(F, stop=-1, qmax=-1, verbose=False):
    """Agglomerative function based on a field structure

    Parameters
    ----------
    F the input field (graph+feature)
    stop: float, optional
        the stopping crterion.  if stop==-1, then no stopping criterion is used
    qmax: int, optional
        the maximum number of desired clusters (in the limit of the stopping
        criterion)
    verbose : bool, optional
        If True, print diagnostic information

    Returns
    -------
    u: array of shape (F.V)
       labelling of the graph vertices according to the criterion
    cost array of shape (F.V - 1)
         the cost of each merge step during the clustering procedure


    Notes
    -----
    See ward_quick_segment for more information

    Caveat : only approximate
    """
    u, cost = ward_quick_segment(F, F.field, stop, qmax, verbose)
    return u, cost


def ward_quick_segment(G, feature, stop=-1, qmax=1, verbose=False):
    """
    Agglomerative function based on a topology-defining graph
    and a feature matrix.

    Parameters
    ----------
    G: labs.graph.WeightedGraph instance
       the input graph (a topological graph essentially)
    feature array of shape (G.V,dim_feature)
        vectorial information related to the graph vertices
    stop1 : int or float, optional
        the stopping crterion if stop==-1, then no stopping criterion is used
    qmax : int, optional
        the maximum number of desired clusters (in the limit of the stopping
        criterion)
    verbose : bool, optional
        If True, print diagnostic information

    Returns
    -------
    u: array of shape (G.V)
       labelling of the graph vertices according to the criterion
    cost: array of shape (G.V - 1)
          the cost of each merge step during the clustering procedure

    Notes
    -----
    Hopefully a quicker version

    A euclidean distance is used in the feature space

    Caveat : only approximate
    """
    # basic check
    if feature.ndim == 1:
        feature = np.reshape(feature, (-1, 1))

    if feature.shape[0] != G.V:
        raise ValueError(
            "Incompatible dimension for the feature matrix and the graph")

    n = G.V
    if stop == - 1:
        stop = np.inf
    qmax = int(np.minimum(qmax, n - 1))
    t = ward_quick(G, feature, verbose)
    if verbose:
        t.plot()

    u1 = np.zeros(n, np.int)
    u2 = np.zeros(n, np.int)
    if stop >= 0:
        u1 = t.partition(stop)
    if qmax > 0:
        u2 = t.split(qmax)

    if u1.max() < u2.max():
        u = u2
    else:
        u = u1

    cost = t.get_height()
    cost = cost[t.isleaf() == False]
    return u, cost


def ward_segment(G, feature, stop=-1, qmax=1, verbose=False):
    """
    Agglomerative function based on a topology-defining graph
    and a feature matrix.

    Parameters
    ----------
    G : graph object
        the input graph (a topological graph essentially)
    feature : array of shape (G.V,dim_feature)
        some vectorial information related to the graph vertices
    stop : int or float, optional
        the stopping crterion.  if stop==-1, then no stopping criterion is used
    qmax : int, optional
        the maximum number of desired clusters (in the limit of the stopping
        criterion)
    verbose : bool, optional
        If True, print diagnostic information

    Returns
    -------
    u: array of shape (G.V):
       a labelling of the graph vertices according to the criterion
    cost: array of shape (G.V - 1)
          the cost of each merge step during the clustering procedure

    Notes
    -----
    A euclidean distance is used in the feature space

    Caveat : when the number of cc in G (nbcc) is greter than qmax, u contains
    nbcc values, not qmax !
    """
    # basic check
    if feature.ndim == 1:
        feature = np.reshape(feature, (-1, 1))

    if feature.shape[0] != G.V:
        raise ValueError(
            "Incompatible dimension for the feature matrix and the graph")

    # prepare a graph with twice the number of vertices
    n = G.V
    if qmax == -1:
        qmax = n - 1
    if stop == -1:
        stop = np.inf
    qmax = int(np.minimum(qmax, n - 1))

    t = ward(G, feature, verbose)
    u1 = np.zeros(n, np.int)
    u2 = np.zeros(n, np.int)
    if stop >= 0:
        u1 = t.partition(stop)
    if qmax > 0:
        u2 = t.split(qmax)

    if u1.max() < u2.max():
        u = u2
    else:
        u = u1

    cost = t.get_height()
    cost = cost[t.isleaf() == False]
    return u, cost


def ward(G, feature, verbose=False):
    """
    Agglomerative function based on a topology-defining graph
    and a feature matrix.

    Parameters
    ----------
    G : graph
        the input graph (a topological graph essentially)
    feature : array of shape (G.V,dim_feature)
        vectorial information related to the graph vertices
    verbose : bool, optional
        If True, print diagnostic information

    Returns
    --------
    t : ``WeightedForest`` instance
        structure that represents the dendrogram

    Notes
    -----
    When G has more than 1 connected component, t is no longer a tree.  This
    case is handled cleanly now
    """
    warn('Function ward from ' + 
         'nipy.algorithms.clustering.hierrachical_clustering ' + 
         'deprecated, will be removed',
         FutureWarning,
         stacklevel=2)

    # basic check
    if feature.ndim == 1:
        feature = np.reshape(feature, (-1, 1))

    if feature.shape[0] != G.V:
        raise ValueError(
            "Incompatible dimension for the feature matrix and the graph")

    Features = [np.ones(2 * G.V), np.zeros((2 * G.V, feature.shape[1])),
                np.zeros((2 * G.V, feature.shape[1]))]
    Features[1][:G.V] = feature
    Features[2][:G.V] = feature ** 2

    # prepare a graph with twice the number of vertices
    # this graph will contain the connectivity information
    # along the merges.
    n = G.V
    nbcc = G.cc().max() + 1
    K = _auxiliary_graph(G, Features)

    # prepare some variables that are useful tp speed up the algorithm
    parent = np.arange(2 * n - nbcc).astype(np.int)
    height = np.zeros(2 * n - nbcc)
    linc = K.left_incidence()
    rinc = K.right_incidence()

    # iteratively merge clusters
    for q in range(n - nbcc):
        # 1. find the lightest edge
        m = (K.weights).argmin()
        cost = K.weights[m]
        k = q + n
        i = K.edges[m, 0]
        j = K.edges[m, 1]
        height[k] = cost
        if verbose:
            print(q, i, j, m, cost)

        # 2. remove the current edge
        K.edges[m] = - 1
        K.weights[m] = np.inf
        linc[i].remove(m)
        rinc[j].remove(m)

        ml = linc[j]
        if np.sum(K.edges[ml, 1] == i) > 0:
            m = ml[np.flatnonzero(K.edges[ml, 1] == i)]
            K.edges[m] = -1
            K.weights[m] = np.inf
            linc[j].remove(m)
            rinc[i].remove(m)

        # 3. merge the edges with third part edges
        parent[i] = k
        parent[j] = k
        for p in range(3):
            Features[p][k] = Features[p][i] + Features[p][j]

        linc, rinc = _remap(K, i, j, k, Features, linc, rinc)

    # build a tree to encode the results
    t = WeightedForest(2 * n - nbcc, parent, height)
    return t


#--------------------------------------------------------------------------
#----------------------- Visualization ------------------------------------
# -------------------------------------------------------------------------


def _label_(f, parent, left, labelled):
    temp = np.nonzero(parent == f)

    if np.size(temp) > 0:
        i = temp[0][np.nonzero(left[temp[0]] == 1)]
        j = temp[0][np.nonzero(left[temp[0]] == 0)]
        labelled = _label_(i, parent, left, labelled)
        labelled[f] = labelled.max() + 1
        labelled = _label_(j, parent, left, labelled)

    if labelled[f] < 0:
        labelled[f] = labelled.max() + 1

    return labelled


def _label(parent):
    # find the root
    root = np.nonzero(parent == np.arange(np.size(parent)))[0]

    # define left
    left = np.zeros(np.size(parent))
    for f in range(np.size(parent)):
        temp = np.nonzero(parent == f)
        if np.size(temp) > 0:
            left[temp[0][0]] = 1

    left[root] = .5

    # define labelled
    labelled = - np.ones(np.size(parent))

    # compute labelled
    for j in range(np.size(root)):
        labelled = _label_(root[j], parent, left, labelled)

    return labelled
