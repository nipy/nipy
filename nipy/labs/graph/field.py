# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from _field import closing, custom_watershed, diffusion, dilation, erosion, \
    field_voronoi, get_local_maxima, local_maxima, opening, \
    threshold_bifurcations
from _field import __doc__
import numpy as np
import graph as fg


"""
This module implements the field structure of nipy-neurospin

Author:Bertrand Thirion, 2006--2009

Fixme : add a subfield method, similar to subgraph
"""


def field_from_coo_matrix_and_data(x, data):
    """
    Instantiates a weighted graph from a (sparse) coo_matrix

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
    if x.shape[0] != x.shape[1]:
        raise ValueError("the input coo_matrix is not square")
    if data.shape[0] != x.shape[0]:
        raise ValueError("data and x do not have consistent shapes")
    i, j = x.nonzero()
    edges = np.vstack((i, j)).T
    weights = x.data
    ifield = Field(x.shape[0], edges, weights, data)
    return ifield


class Field(fg.WeightedGraph):
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
        if (edges == None) & (weights == None):
            pass # fixme
        else:
            if edges.shape[0] == np.size(weights):
                E = edges.shape[0]
                # fixme: quick and dirty
                self.V = V
                self.E = E
                self.edges = edges
                self.weights = weights
            else:
                raise ValueError('Incompatible size of the edges \
                                  and weights matrices')
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

    def print_field(self):
        print self.field

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
        """
        Morphological closing of the field data. self.field is changed

        Parameters
        ----------
        nbiter=1 : the number of iterations required
        """
        nbiter = int(nbiter)
        if self.E < 1:
            return
        if nbiter > 0:
            for i in range(self.field.shape[1]):
                self.field[:, i] = closing(self.edges[:, 0], self.edges[:, 1],
                                          self.field[:, i], nbiter)

    def opening(self, nbiter=1):
        """ Morphological opening of the field data.
        self.field is changed inplace

        Parameters
        ----------
        nbiter: int, optional, the number of iterations required
        """
        nbiter = int(nbiter)
        if self.E < 1:
            return
        if nbiter > 0:
            for i in range(self.field.shape[1]):
                self.field[:, i] = opening(self.edges[:, 0], self.edges[:, 1],
                                          self.field[:, i], nbiter)

    def dilation(self, nbiter=1):
        """
        Morphological dimlation of the field data. self.field is changed

        Parameters
        ----------
        nbiter: int, optional, the number of iterations required
        """
        nbiter = int(nbiter)
        if self.E < 1:
            return
        if nbiter > 0:
            for i in range(self.field.shape[1]):
                self.field[:, i] = dilation(self.edges[:, 0], self.edges[:, 1],
                                           self.field[:, i], nbiter)

    def erosion(self, nbiter=1):
        """Morphological openeing of the field

        Parameters
        ----------
        nbiter: int, optional, the number of iterations required
        """
        nbiter = int(nbiter)
        if self.E < 1:
            return
        if nbiter > 0:
            for i in range(self.field.shape[1]):
                self.field[:, i] = erosion(self.edges[:, 0], self.edges[:, 1],
                                          self.field[:, i], nbiter)

    def get_local_maxima(self, refdim=0, th=-1 * np.infty):
        """
        Look for the local maxima of one dimension (refdim) of self.field

        Parameters
        ----------
        refdim (int) the field dimension over which the maxima are looked after
        th = -np.infty (float, optional)
            threshold so that only values above th are considered

        Returns
        -------
        idx: array of shape (nmax)
             indices of the vertices that are local maxima
        depth: array of shape (nmax)
               topological depth of the local maxima :
               depth[idx[i]] = q means that idx[i] is a q-order maximum
        """
        refdim = int(refdim)
        if np.size(self.field) == 0:
            raise ValueError('No field has been defined so far')
        if self.field.shape[1] - 1 < refdim:
            raise ValueError('refdim > field.shape[1]')
        idx = np.arange(np.sum(self.field > th))
        depth = self.V * np.ones(np.sum(self.field > th), np.int)
        if self.E > 0:
            try:
                idx, depth = get_local_maxima(
                    self.edges[:, 0], self.edges[:, 1], self.field[:, refdim],
                    th)
            except:
                idx = []
                depth = []

        return idx, depth

    def local_maxima(self, refdim=0):
        """
        Look for all the local maxima of a field

        Parameters
        ----------
        refdim (int) field dimension over which the maxima are looked after

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
            raise ValueError(refdim > field.shape[1])
        depth = self.V * np.ones(self.V, np.int)
        if self.E > 0:
            depth = local_maxima(self.edges[:, 0], self.edges[:, 1],
                                 self.field[:, refdim])
        return depth

    def diffusion(self, nbiter=1):
        """
        diffusion of a field of data in the weighted graph structure
        Note that this changes self.field

        Parameters
        ----------
        nbiter=1: the number of iterations required
                  (the larger the smoother)

        Note
        ----
        The process is run for all the dimensions of the field
        """
        nbiter = int(nbiter)
        if (self.E > 0) & (nbiter > 0) & (np.size(self.field) > 0):
            self.field = diffusion(self.edges[:, 0], self.edges[:, 1],
                                   self.weights, self.field, nbiter)

    def custom_watershed(self, refdim=0, th=-1 * np.infty):
        """
        watershed analysis of the field.
        Note that bassins are found aound each maximum
        (and not minimum as conventionally)

        Parameters
        ----------
        th is a threshold so that only values above th are considered
        by default, th = -infty (numpy)

        Returns
        -------
        idx: array of shape (nbassins)
             indices of the vertices that are local maxima
        depth: array of shape (nbassins)
               topological the depth of the bassins
               depth[idx[i]] = q means that idx[i] is a q-order maximum
               Note that this is also the diameter of the basins
               associated with local maxima
        major: array of shape (nbassins)
               label of the maximum which dominates each local maximum
               i.e. it describes the hierarchy of the local maxima
        label : array of shape (self.V)
              labelling of the vertices according to their bassin
        """
        if (np.size(self.field) == 0):
            raise ValueError('No field has been defined so far')
        if self.field.shape[1] - 1 < refdim:
            raise ValueError('refdim>field.shape[1]')
        f = self.field[:, refdim]
        idx = np.nonzero(f > th)
        idx = np.reshape(idx, np.size(idx))
        depth = self.V * np.ones(np.sum(f > th), np.int)
        major = np.arange(np.sum(f > th))
        label = np.zeros(self.V, np.int)
        label[idx] = major
        if self.E > 0:
            idx, depth, major, label = custom_watershed(self.edges[:, 0],
                        self.edges[:, 1], f, th)
        return idx, depth, major, label

    def threshold_bifurcations(self, refdim=0, th=-1 * np.infty):
        """
        analysis of the level sets of the field:
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
        if (np.size(self.field) == 0):
            raise ValueError('No field has been defined so far')
        if self.field.shape[1] - 1 < refdim:
            raise ValueError('refdim>field.shape[1]')
        idx = np.nonzero(self.field[:, refdim] > th)
        height = self.V * np.ones(np.sum(self.field > th))
        parents = np.arange(np.sum(self.field > th))
        label = np.zeros(self.V, np.int)
        label[idx] = parents
        if self.E > 0:
            idx, height, parents, label = threshold_bifurcations(\
                self.edges[:, 0], self.edges[:, 1], self.field[:, refdim], th)
        return idx, height, parents, label

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
        label = field_voronoi(self.edges[:, 0], self.edges[:, 1], self.field,
                              seed)
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
        inertia_old = np.infty
        if seeds == None:
            k = label.max() + 1
            if np.size(np.unique(label)) != k:
                raise ValueError('missing values, I cannot proceed')
            seeds = np.zeros(k).astype(np.int)
            for  j in range(k):
                lj = np.nonzero(label == j)[0]
                cent = np.mean(self.field[lj], 0)
                tj = np.argmin(np.sum((cent - self.field[lj]) ** 2, 1))
                seeds[j] = lj[tj]
        else:
            k = np.size(seeds)

        for i in range(maxiter):
            label = field_voronoi(self.edges[:, 0], self.edges[:, 1],
                                  self.field, seeds)
            #update the seeds
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
        from ..clustering.hierarchical_clustering\
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
        return Field(self.V, self.edges, self.weights, self.field)

    def subfield(self, valid):
        """
        Returns a subfield of self,
        with only the vertices such that valid >0

        Parameters
        ----------
        valid: array of shape (self.V),
               nonzero for vertices to be retained

        Returns
        -------
        F: Field instance,
           the desired subfield of self

        Note
        ----
        The vertices are renumbered as [1..p] where p = sum(valid>0)
        when sum(valid==0) then None is returned
        """
        G = self.subgraph(valid)
        if G == None:
            return None
        field = self.field[valid]
        return Field(G.V, G.edges, G.weights, field)
