# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This module implements the Field class, which simply a WeightedGraph
(see the graph.py) module, plus an arrray that yields (possibly
multi-dimnesional) features associated with graph vertices. This
allows some kinds of computations (all thoses relating to mathematical
morphology, diffusion etc.)

Certain functions are provided to Instantiate Fields easily, given a
WeightedGraph and feature data.

Author:Bertrand Thirion, 2006--2011
"""
from warnings import warn
import numpy as np

from .graph import WeightedGraph

NEGINF = -np.inf


def field_from_coo_matrix_and_data(x, data):
    """ Instantiates a weighted graph from a (sparse) coo_matrix

    Parameters
    ----------
    x: (V, V) scipy.sparse.coo_matrix instance,
       the input matrix
    data: array of shape (V, dim),
          the field data

    Returns
    -------
    ifield: resulting Field instance
    """
    if x.shape[0] != x.shape[1]:
        raise ValueError("the input coo_matrix is not square")
    if data.shape[0] != x.shape[0]:
        raise ValueError("data and x do not have consistent shapes")
    i, j = x.nonzero()
    edges = np.vstack((i, j)).T
    weights = x.data
    ifield = Field(x.shape[0], edges, weights, data)
    return ifield


def field_from_graph_and_data(g, data):
    """ Instantiate a Fieldfrom a WeightedGraph plus some feature data
    Parameters
    ----------
    x: (V, V) scipy.sparse.coo_matrix instance,
       the input matrix
    data: array of shape (V, dim),
          the field data

    Returns
    -------
    ifield: resulting field instance
    """
    if data.shape[0] != g.V:
        raise ValueError("data and g do not have consistent shapes")
    ifield = Field(g.V, g.edges, g.weights, data)
    return ifield


class Field(WeightedGraph):
    """
    This is the basic field structure,
         which contains the weighted graph structure
         plus an array of data (the 'field')
    field is an array of size(n, p)
          where n is the number of vertices of the graph
          and p is the field dimension
    """

    def __init__(self, V, edges=None, weights=None, field=None):
        """
        Parameters
        ----------
        V (int > 0) the number of vertices of the graph
        edges=None: the edge array of the graph
        weights=None: the asociated weights array
        field=None: the field data itself
        """
        V = int(V)
        if V < 1:
            raise ValueError('cannot create graph with no vertex')
        self.V = int(V)
        self.E = 0
        self.edges = []
        self.weights = []
        if (edges is not None) or (weights is not None):
            if len(edges) == 0:
                E = 0
            elif edges.shape[0] == np.size(weights):
                E = edges.shape[0]
            else:
                raise ValueError('Incompatible size of the edges \
                                  and weights matrices')
            self.V = V
            self.E = E
            self.edges = edges
            self.weights = weights
        self.field = []
        if field == None:
            pass
        else:
            if np.size(field) == self.V:
                field = np.reshape(field, (self.V, 1))
            if field.shape[0] != self.V:
                raise ValueError('field does not have a correct size')
            else:
                self.field = field

    def get_field(self):
        return self.field

    def set_field(self, field):
        if np.size(field) == self.V:
            field = np.reshape(field, (self.V, 1))
        if field.shape[0] != self.V:
            raise ValueError('field does not have a correct size')
        else:
            self.field = field

    def closing(self, nbiter=1):
        """Morphological closing of the field data. 
        self.field is changed inplace

        Parameters
        ----------
        nbiter=1 : the number of iterations required
        """
        nbiter = int(nbiter)
        self.dilation(nbiter)
        self.erosion(nbiter)

    def opening(self, nbiter=1):
        """Morphological opening of the field data.
        self.field is changed inplace

        Parameters
        ----------
        nbiter: int, optional, the number of iterations required
        """
        nbiter = int(nbiter)
        self.erosion(nbiter)
        self.dilation(nbiter)

    def dilation(self, nbiter=1, fast=True):
        """Morphological dilation of the field data, changed in place

        Parameters
        ----------
        nbiter: int, optional, the number of iterations required

        Note
        ----
        When data dtype is not float64, a slow version of the code is used
        """
        nbiter = int(nbiter)
        if self.field.dtype != np.float64:
            warn('data type is not float64; a slower version is used')
            fast = False
        if fast:
            from ._graph import dilation
            if self.E > 0:
                if (self.field.size == self.V):
                    self.field = self.field.reshape((self.V, 1))
                idx, neighb, _ = self.compact_neighb()
                for i in range(nbiter):
                    dilation(self.field, idx, neighb)
        else:
            from scipy.sparse import dia_matrix
            adj = self.to_coo_matrix() + dia_matrix(
                (np.ones(self.V), 0), (self.V, self.V))
            rows = adj.tolil().rows
            for i in range(nbiter):
                self.field = np.array([self.field[row].max(0) for row in rows])

    def highest_neighbor(self, refdim=0):
        """Computes the neighbor with highest field value along refdim

        Parameters
        ----------
        refdim: int, optional,
                the dimension of the field under consideration

        Returns
        -------
        hneighb: array of shape(self.V), 
                 index of the neighbor with highest value
        """
        from scipy.sparse import dia_matrix
        refdim = int(refdim)
        # add self-edges to avoid singularities, when taking the maximum
        adj = self.to_coo_matrix() + dia_matrix(
            (np.ones(self.V), 0), (self.V, self.V))
        rows = adj.tolil().rows
        hneighb = np.array([row[self.field[row].argmax()] for row in rows])
        return hneighb

    def erosion(self, nbiter=1):
        """Morphological openeing of the field

        Parameters
        ----------
        nbiter: int, optional, the number of iterations required
        """
        nbiter = int(nbiter)
        lil = self.to_coo_matrix().tolil().rows.tolist()
        for i in range(nbiter):
            nf = np.zeros_like(self.field)
            for k, neighbors in enumerate(lil):
                nf[k] = self.field[neighbors].min(0)
            self.field = nf

    def get_local_maxima(self, refdim=0, th=NEGINF):
        """
        Look for the local maxima of one dimension (refdim) of self.field

        Parameters
        ----------
        refdim (int) the field dimension over which the maxima are looked after
        th = float, optional
            threshold so that only values above th are considered

        Returns
        -------
        idx: array of shape (nmax)
             indices of the vertices that are local maxima
        depth: array of shape (nmax)
               topological depth of the local maxima :
               depth[idx[i]] = q means that idx[i] is a q-order maximum
        """
        depth_all = self.local_maxima(refdim, th)
        idx = np.ravel(np.where(depth_all))
        depth = depth_all[idx]
        return idx, depth

    def local_maxima(self, refdim=0, th=NEGINF):
        """Returns all the local maxima of a field

        Parameters
        ----------
        refdim (int) field dimension over which the maxima are looked after
        th: float, optional
            threshold so that only values above th are considered

        Returns
        -------
        depth: array of shape (nmax)
               a labelling of the vertices such that
               depth[v] = 0 if v is not a local maximum
               depth[v] = 1 if v is a first order maximum
               ...
               depth[v] = q if v is a q-order maximum
        """
        refdim = int(refdim)
        if np.size(self.field) == 0:
            raise ValueError('No field has been defined so far')
        if self.field.shape[1] - 1 < refdim:
            raise ValueError(refdim > self.shape[1])
        depth = np.zeros(self.V, np.int)

        # create a subfield(thresholding)
        sf = self.subfield(self.field.T[refdim] >= th)
        initial_field = sf.field.T[refdim]
        sf.field = initial_field.astype(np.float64)

        # compute the depth in the subgraph
        ldepth = sf.V * np.ones(sf.V, np.int)
        for k in range(sf.V):
            dilated_field_old = sf.field.ravel().copy()
            sf.dilation(1)
            non_max = sf.field.ravel() > dilated_field_old
            ldepth[non_max] = np.minimum(k, ldepth[non_max])
            if (non_max == False).all():
                ldepth[sf.field.ravel() == initial_field] = np.maximum(k, 1)
                break

        # write all the depth values
        depth[self.field[:, refdim] >= th] = ldepth
        return depth

    def diffusion(self, nbiter=1):
        """diffusion of the field data in the weighted graph structure
        self.field is changed inplace

        Parameters
        ----------
        nbiter: int, optional the number of iterations required

        Notes
        -----
        The process is run for all the dimensions of the field
        """
        nbiter = int(nbiter)
        adj = self.to_coo_matrix()
        for i in range(nbiter):
            self.field = adj * self.field

    def custom_watershed(self, refdim=0, th=NEGINF):
        """ customized watershed analysis of the field.
        Note that bassins are found around each maximum
        (and not minimum as conventionally)

        Parameters
        ----------
        refdim: int, optional
        th: float optional, threshold of the field

        Returns
        -------
        idx: array of shape (nbassins)
             indices of the vertices that are local maxima
        label : array of shape (self.V)
              labelling of the vertices according to their bassin
        """
        import numpy.ma as ma
        from graph import Graph

        if (np.size(self.field) == 0):
            raise ValueError('No field has been defined so far')
        if self.field.shape[1] - 1 < refdim:
            raise ValueError('refdim>field.shape[1]')

        label = - np.ones(self.V, np.int)

        # create a subfield(thresholding)
        sf = self.subfield(self.field[:, refdim] >= th)

        # compute the basins
        hneighb = sf.highest_neighbor()
        edges = np.vstack((hneighb, np.arange(sf.V))).T
        edges = np.vstack((edges, np.vstack((np.arange(sf.V), hneighb)).T))
        aux = Graph(sf.V, edges.shape[0], edges)
        llabel = aux.cc()
        n_bassins = len(np.unique(llabel))

        # write all the depth values
        label[self.field[:, refdim] >= th] = llabel
        idx = np.array([ma.array(
                    self.field[:, refdim], mask=(label != c)).argmax()
                        for c in range(n_bassins)])
        return idx, label

    def threshold_bifurcations(self, refdim=0, th=NEGINF):
        """Analysis of the level sets of the field:
        Bifurcations are defined as changes in the topology in the level sets
        when the level (threshold) is varied
        This can been thought of as a kind of Morse analysis

        Parameters
        ----------
        th: float, optional,
            threshold so that only values above th are considered

        Returns
        -------
        idx: array of shape (nlsets)
             indices of the vertices that are local maxima
        height: array of shape (nlsets)
                the depth of the local maxima
                depth[idx[i]] = q means that idx[i] is a q-order maximum
                Note that this is also the diameter of the basins
                associated with local maxima
        parents: array of shape (nlsets)
                 the label of the maximum which dominates each local maximum
                 i.e. it describes the hierarchy of the local maxima
        label: array of shape (self.V)
               a labelling of thevertices according to their bassin
        """
        import numpy.ma as ma
        if (np.size(self.field) == 0):
            raise ValueError('No field has been defined so far')
        if self.field.shape[1] - 1 < refdim:
            raise ValueError('refdim>field.shape[1]')

        label = - np.ones(self.V, np.int)

        # create a subfield(thresholding)
        sf = self.subfield(self.field[:, refdim] >= th)
        initial_field = sf.field[:, refdim].copy()
        sf.field = initial_field.copy()

        # explore the subfield
        order = np.argsort(- initial_field)
        rows = sf.to_coo_matrix().tolil().rows
        llabel = - np.ones(sf.V, np.int)
        parent, root =  np.arange(2 * self.V), np.arange(2 * self.V)
        # q will denote the region index
        q = 0
        for i in order:
            if (llabel[rows[i]] > - 1).any():
                nlabel = np.unique(llabel[rows[i]])
                if nlabel[0] == -1:
                    nlabel = nlabel[1:]
                nlabel = np.unique(root[nlabel])
                if len(nlabel) == 1:
                    # we are at a regular point
                    llabel[i] = nlabel[0]
                else:
                    # we are at a saddle point
                    llabel[i] = q
                    parent[nlabel] = q
                    root[nlabel] = q
                    for j in nlabel:
                        root[root == j] = q
                    q += 1
            else:
                # this is a new component
                llabel[i] = q
                q += 1
        parent = parent[:q]

        # write all the depth values
        label[self.field[:, refdim] >= th] = llabel
        idx = np.array([ma.array(
                    self.field[:, refdim], mask=(label != c)).argmax()
                         for c in range(q)])
        return idx, parent, label

    def constrained_voronoi(self, seed):
        """Voronoi parcellation of the field starting from the input seed

        Parameters
        ----------
        seed: int array of shape(p), the input seeds

        Returns
        -------
        label: The resulting labelling of the data

        Fixme
        -----
        deal with graphs with several ccs
        """
        if np.size(self.field) == 0:
            raise ValueError('No field has been defined so far')
        seed = seed.astype(np.int)
        weights = np.sqrt(np.sum((self.field[self.edges.T[0]] -
                                  self.field[self.edges.T[1]]) ** 2, 1))
        g = WeightedGraph(self.V, self.edges, weights)
        label = g.voronoi_labelling(seed)
        return label

    def geodesic_kmeans(self, seeds=None, label=None, maxiter=100, eps=1.e-4,
                        verbose=0):
        """ Geodesic k-means algorithm
        i.e. obtention of clusters that are topologically
        connected and minimally variable concerning the information
        of self.field

        Parameters
        ----------
        seeds: array of shape(p), optional,
               initial indices of the seeds within the field
               if seeds==None the labels are used as initialization
        labels: array of shape(self.V) initial labels, optional,
                it is expected that labels take their values
                in a certain range (0..lmax)
                if Labels==None, this is not used
                if seeds==None and labels==None,  an ewxception is raised
        maxiter: int, optional,
                 maximal number of iterations
        eps: float, optional,
             increase of inertia at which convergence is declared

        Returns
        -------
        seeds: array of shape (p), the final seeds
        label : array of shape (self.V), the resulting field label
        J: float, inertia value
        """
        if np.size(self.field) == 0:
            raise ValueError('No field has been defined so far')

        if (seeds == None) and (label == None):
            raise ValueError('No initialization has been provided')
        k = np.size(seeds)
        inertia_old = NEGINF
        if seeds == None:
            k = label.max() + 1
            if np.size(np.unique(label)) != k:
                raise ValueError('missing values, cannot proceed')
            seeds = np.zeros(k).astype(np.int)
            for  j in range(k):
                lj = np.nonzero(label == j)[0]
                cent = np.mean(self.field[lj], 0)
                tj = np.argmin(np.sum((cent - self.field[lj]) ** 2, 1))
                seeds[j] = lj[tj]
        else:
            k = np.size(seeds)

        for i in range(maxiter):
            # voronoi labelling
            label = self.constrained_voronoi(seeds)
            # update the seeds
            inertia = 0
            pinteria = 0
            for  j in range(k):
                lj = np.nonzero(label == j)[0]
                pinteria += np.sum(
                    (self.field[seeds[j]] - self.field[lj]) ** 2)
                cent = np.mean(self.field[lj], 0)
                tj = np.argmin(np.sum((cent - self.field[lj]) ** 2, 1))
                seeds[j] = lj[tj]
                inertia += np.sum((cent - self.field[lj]) ** 2)
            if verbose:
                print i, inertia
            if np.absolute(inertia_old - inertia) < eps:
                break
            inertia_old = inertia
        return seeds, label, inertia

    def ward(self, nbcluster):
        """Ward's clustering of self

        Parameters
        ----------
        nbcluster: int,
                   the number of desired clusters

        Returns
        -------
        label: array of shape (self.V)
               the resulting field label
        J (float): the resulting inertia
        """
        from nipy.algorithms.clustering.hierarchical_clustering\
             import ward_segment
        label, J = ward_segment(self, self.field, qmax=nbcluster)

        # compute the resulting inertia
        inertia = 0
        for  j in range(nbcluster):
            lj = np.nonzero(label == j)[0]
            cent = np.mean(self.field[lj], 0)
            inertia += np.sum((cent - self.field[lj]) ** 2)
        return label, inertia

    def copy(self):
        """ copy function
        """
        return Field(self.V, self.edges.copy(),
                     self.weights.copy(), self.field.copy())

    def subfield(self, valid):
        """Returns a subfield of self, with only vertices such that valid > 0

        Parameters
        ----------
        valid: array of shape (self.V),
               nonzero for vertices to be retained

        Returns
        -------
        F: Field instance,
           the desired subfield of self

        Notes
        -----
        The vertices are renumbered as [1..p] where p = sum(valid>0) when
        sum(valid) == 0 then None is returned
        """
        G = self.subgraph(valid)
        if G == None:
            return None
        field = self.field[valid]
        if len(G.edges) == 0:
            edges = np.array([[], []]).T
        else:
            edges = G.edges
        return Field(G.V, edges, G.weights, field)
