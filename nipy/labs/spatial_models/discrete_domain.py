"""
This module defines the StructuredDomain class,
that represents a generic neuroimaging kind of domain
This is meant to provide a unified API to deal with n-d imaged and meshes.


Author: Bertrand Thirion, 2010
"""
import numpy as np
import scipy.sparse as sp

from nibabel import load, Nifti1Image, save

from nipy.io.nibcompat import get_header, get_affine
from nipy.algorithms.graph import (WeightedGraph, wgraph_from_coo_matrix,
                                   wgraph_from_3d_grid)

##############################################################
# Ancillary functions
##############################################################


def smatrix_from_3d_array(mask, nn=18):
    """Create a sparse adjacency matrix from an array

    Parameters
    ----------
    mask : 3d array,
           input array, interpreted as a mask
    nn: int, optional
        3d neighboring system to be chosen within {6, 18, 26}

    Returns
    -------
    coo_mat: a sparse coo matrix,
             adjacency of the neighboring system

    """
    ijk = np.array(np.where(mask)).T
    return smatrix_from_3d_idx(ijk, nn)


def smatrix_from_3d_idx(ijk, nn=18):
    """Create a sparse adjacency matrix from 3d index system

    Parameters
    ----------
    idx:array of shape (n_samples, 3), type int
        indexes of certain positions in a 3d space
    nn: int, optional
        3d neighboring system to be chosen within {6, 18, 26}

    Returns
    -------
    coo_mat: a sparse coo matrix,
             adjacency of the neighboring system

    """
    G = wgraph_from_3d_grid(ijk, nn)
    return G.to_coo_matrix()


def smatrix_from_nd_array(mask, nn=0):
    """Create a sparse adjacency matrix from an arbitrary nd array

    Parameters
    ----------
    mask : nd array,
           input array, interpreted as a mask
    nn: int, optional
        nd neighboring system, unsused at the moment

    Returns
    -------
    coo_mat: a sparse coo matrix,
             adjacency of the neighboring system

    """
    idx = np.array(np.where(mask)).T
    return smatrix_from_nd_idx(idx, nn)


def smatrix_from_nd_idx(idx, nn=0):
    """Create a sparse adjacency matrix from nd index system

    Parameters
    ----------
    idx:array of shape (n_samples, dim), type int
        indexes of certain positions in a nd space
    nn: int, optional
        nd neighboring system, unused at the moment

    Returns
    -------
    coo_mat: a sparse coo matrix,
             adjacency of the neighboring system

    """
    n = idx.shape[0]
    dim = idx.shape[1]
    nidx = idx - idx.min(0)
    eA = []
    eB = []

    # compute the edges in each possible direction
    for d in range(dim):
        mi = nidx.max(0) + 2
        a = np.hstack((1, np.cumprod(mi[:dim - 1])))
        v1 = np.dot(nidx, a)
        assert(np.size(v1) == np.size(np.unique(v1)))
        o1 = np.argsort(v1)
        sv1 = v1[o1]
        nz = np.squeeze(np.nonzero(sv1[:n - 1] - sv1[1:] == - 1))
        nz = np.reshape(nz, np.size(nz))
        eA.append(o1[nz])
        eB.append(o1[nz + 1])
        nidx = np.roll(nidx, 1, 1)

    eA = np.concatenate(eA)
    eB = np.concatenate(eB)
    E = 2 * np.size(eA)
    # create a graph structure
    if E == 0:
        return sp.coo_matrix((n, n))

    edges = np.vstack((np.hstack((eA, eB)), np.hstack((eB, eA)))).T
    weights = np.ones(E)
    G = WeightedGraph(n, edges, weights)
    return G.to_coo_matrix()


def array_affine_coord(mask, affine):
    """Compute coordinates from a boolean array and an affine transform

    Parameters
    ----------
    mask: nd array,
           input array, interpreted as a mask
    affine: (n+1, n+1) matrix,
            affine transform that maps the mask points to some embedding space

    Returns
    -------
    coords: array of shape(sum(mask>0), n),
            the computed coordinates

    """
    idx = np.array(np.where(mask)).T
    return idx_affine_coord(idx, affine)


def idx_affine_coord(idx, affine):
    """Compute coordinates from a set of indexes and an affine transform

    Parameters
    ----------
    idx:array of shape (n_samples, dim), type int
        indexes of certain positions in a nd space
    affine: (n+1, n+1) matrix,
            affine transform that maps the mask points to some embedding space

    Returns
    -------
    coords: array of shape(sum(mask>0), n),
            the computed coordinates

    """
    size = idx.shape[0]
    hidx = np.hstack((idx, np.ones((size, 1))))
    coord = np.dot(hidx, affine.T)[:, 0:-1]
    return coord


def reduce_coo_matrix(mat, mask):
    """Reduce a supposedly coo_matrix to the vertices in the mask

    Parameters
    ----------
    mat: sparse coo_matrix,
         input matrix
    mask: boolean array of shape mat.shape[0],
          desired elements

    """
    G = wgraph_from_coo_matrix(mat)
    K = G.subgraph(mask)
    if K is None:
        return None
    return K.to_coo_matrix()


#################################################################
# Functions to instantiate StructuredDomains
#################################################################
def domain_from_binary_array(mask, affine=None, nn=0):
    """Return a StructuredDomain from an n-d array

    Parameters
    ----------
    mask: np.array instance
          a supposedly boolean array that repesents the domain
    affine: np.array, optional
            affine transform that maps the array coordinates
            to some embedding space
            by default, this is np.eye(dim+1, dim+1)
    nn: neighboring system considered
        unsued at the moment

    """
    dim = len(mask.shape)
    if affine is None:
        affine = np.eye(dim + 1)
    mask = mask > 0
    vol = np.absolute(np.linalg.det(affine)) * np.ones(np.sum(mask))
    coord = array_affine_coord(mask, affine)
    topology = smatrix_from_nd_array(mask)
    return StructuredDomain(dim, coord, vol, topology)


def domain_from_image(mim, nn=18):
    """Return a StructuredDomain instance from the input mask image

    Parameters
    ----------
    mim: NiftiIImage instance, or string path toward such an image
         supposedly a mask (where is used to crate the DD)
    nn: int, optional
        neighboring system considered from the image
        can be 6, 18 or 26

    Returns
    -------
    The corresponding StructuredDomain instance

    """
    if isinstance(mim, basestring):
        iim = load(mim)
    else:
        iim = mim
    return domain_from_binary_array(iim.get_data(), get_affine(iim), nn)


def grid_domain_from_binary_array(mask, affine=None, nn=0):
    """Return a NDGridDomain from an n-d array

    Parameters
    ----------
    mask: np.array instance
          a supposedly boolean array that repesents the domain
    affine: np.array, optional
            affine transform that maps the array coordinates
            to some embedding space
            by default, this is np.eye(dim+1, dim+1)
    nn: neighboring system considered
        unsued at the moment

    """
    dim = len(mask.shape)
    shape = mask.shape
    if affine is None:
        affine = np.eye(dim + 1)

    mask = mask > 0
    ijk = np.array(np.where(mask)).T
    vol = np.absolute(np.linalg.det(affine[:3, 0:3])) * np.ones(np.sum(mask))
    topology = smatrix_from_nd_idx(ijk, nn)
    return NDGridDomain(dim, ijk, shape, affine, vol, topology)


def grid_domain_from_image(mim, nn=18):
    """Return a NDGridDomain instance from the input mask image

    Parameters
    ----------
    mim: NiftiIImage instance, or string path toward such an image
         supposedly a mask (where is used to crate the DD)
    nn: int, optional
        neighboring system considered from the image
        can be 6, 18 or 26

    Returns
    -------
    The corresponding NDGridDomain instance

    """
    if isinstance(mim, basestring):
        iim = load(mim)
    else:
        iim = mim
    return grid_domain_from_binary_array(iim.get_data(), get_affine(iim), nn)


def grid_domain_from_shape(shape, affine=None):
    """Return a NDGridDomain from an n-d array

    Parameters
    ----------
    shape: tuple
      the shape of a rectangular domain.
    affine: np.array, optional
      affine transform that maps the array coordinates
      to some embedding space.
      By default, this is np.eye(dim+1, dim+1)

    """
    dim = len(shape)
    if affine is None:
        affine = np.eye(dim + 1)

    rect = np.ones(shape)
    ijk = np.array(np.where(rect)).T
    vol = np.absolute(np.linalg.det(affine[:3, 0:3])) * np.ones(np.sum(rect))
    topology = smatrix_from_nd_idx(ijk, 0)
    return NDGridDomain(dim, ijk, shape, affine, vol, topology)


################################################################
# Domain from mesh
################################################################


class MeshDomain(object):
    """
    temporary class to handle meshes
    """

    def __init__(self, coord, triangles):
        """Initialize mesh domain instance

        Parameters
        ----------
        coord: array of shape (n_vertices, 3),
               the node coordinates
        triangles: array of shape(n_triables, 3),
                   indices of the nodes per triangle

        """
        self.coord = coord
        self.triangles = triangles
        self.V = len(coord)
        # fixme: implement consistency checks

    def area(self):
        """Return array of areas for each node

        Returns
        -------
        area: array of shape self.V,
              area of each node

        """
        E = len(self.triangles)
        narea = np.zeros(self.V)

        def _area(a, b):
            """Area spanned by the vectors(a,b) in 3D
            """
            c = np.array([a[1] * b[2] - a[2] * b[1],
                          - a[0] * b[2] + a[2] * b[0],
                          a[0] * b[1] - a[1] * b[0]])
            return np.sqrt((c ** 2).sum())

        for e in range(E):
            i, j, k = self.triangles[e]
            a = self.coord[i] - self.coord[k]
            b = self.coord[j] - self.coord[k]
            ar = _area(a, b)
            narea[i] += ar
            narea[j] += ar
            narea[k] += ar

        narea /= 6
        # because division by 2 has been 'forgotten' in area computation
        # the area of a triangle is divided into the 3 vertices
        return narea

    def topology(self):
        """Returns a sparse matrix that represents the connectivity in self
        """
        E = len(self.triangles)
        edges = np.zeros((3 * E, 2))
        weights = np.ones(3 * E)

        for i in range(E):
            sa, sb, sc = self.triangles[i]
            edges[3 * i] = np.array([sa, sb])
            edges[3 * i + 1] = np.array([sa, sc])
            edges[3 * i + 2] = np.array([sb, sc])

        G = WeightedGraph(self.V, edges, weights)

        # symmeterize the graph
        G = G.symmeterize()

        # remove redundant edges
        G = G.cut_redundancies()

        # make it a metric graph
        G.set_euclidian(self.coord)
        return G.to_coo_matrix()


def domain_from_mesh(mesh):
    """Instantiate a StructuredDomain from a gifti mesh

    Parameters
    ----------
    mesh: nibabel gifti mesh instance, or path to such a mesh

    """
    if isinstance(mesh, basestring):
        from nibabel.gifti import read
        mesh_ = read(mesh)
    else:
        mesh_ = mesh

    if len(mesh_.darrays) == 2:
        cor, tri = mesh_.darrays
    elif len(mesh_.darrays) == 3:
        cor, nor, tri = mesh_.darrays
    else:
        raise Exception("%d arrays in gifti file (case not handled)" \
                            % len(mesh_.darrays))
    mesh_dom = MeshDomain(cor.data, tri.data)

    vol = mesh_dom.area()
    topology = mesh_dom.topology()
    dim = 2
    return StructuredDomain(dim, mesh_dom.coord, vol, topology)


################################################################
# StructuredDomain class
################################################################


class DiscreteDomain(object):
    """
    Descriptor of a certain domain that consists of discrete elements that
    are characterized by a coordinate system and a topology:
    the coordinate system is specified through a coordinate array
    the topology encodes the neighboring system

    """

    def __init__(self, dim, coord, local_volume, id='', referential=''):
        """Initialize discrete domain instance

        Parameters
        ----------
        dim: int,
             the (physical) dimension of the domain
        coord: array of shape(size, em_dim),
               explicit coordinates of the domain sites
        local_volume: array of shape(size),
                      yields the volume associated with each site
        id: string, optional,
             domain identifier
        referential: string, optional,
                     identifier of the referential of the coordinates system

        Caveat
        ------
        em_dim may be greater than dim e.g. (meshes coordinate in 3D)

        """
        # dimension
        self.dim = dim

        # number of discrete elements
        self.size = coord.shape[0]

        # coordinate system
        if np.size(coord) == coord.shape[0]:
            coord = np.reshape(coord, (np.size(coord), 1))

        if np.size(coord) == 0:
            self.em_dim = dim
        else:
            self.em_dim = coord.shape[1]

        if self.em_dim < dim:
            raise ValueError('Embedding dimension cannot be smaller than dim')
        self.coord = coord

        # volume
        if np.size(local_volume) != self.size:
            raise ValueError("Inconsistent Volume size")
        if (local_volume < 0).any():
            raise ValueError('Volume should be positive')

        self.local_volume = np.ravel(local_volume)
        self.id = id
        self.referential = referential
        self.features = {}

    def copy(self):
        """Returns a copy of self
        """
        new_dom = DiscreteDomain(self.dim, self.coord.copy(),
                                  self.local_volume.copy(), self.id,
                                  self.referential)
        for fid in self.features.keys():
            new_dom.set_feature(fid, self.get_feature(fid).copy())
        return new_dom

    def get_coord(self):
        """Returns self.coord
        """
        return self.coord

    def get_volume(self):
        """Returns self.local_volume
        """
        return self.local_volume

    def connected_components(self):
        """returns a labelling of the domain into connected components
        """
        if self.topology is not None:
            return wgraph_from_coo_matrix(self.topology).cc()
        else:
            return []

    def mask(self, bmask, id=''):
        """Returns an DiscreteDomain instance that has been further masked
        """
        if bmask.size != self.size:
            raise ValueError('Invalid mask size')

        svol = self.local_volume[bmask]
        scoord = self.coord[bmask]
        DD = DiscreteDomain(self.dim, scoord, svol, id, self.referential)

        for fid in self.features.keys():
            f = self.features.pop(fid)
            DD.set_feature(fid, f[bmask])
        return DD

    def set_feature(self, fid, data, override=True):
        """Append a feature 'fid'

        Parameters
        ----------
        fid: string,
             feature identifier
        data: array of shape(self.size, p) or self.size
              the feature data

        """
        if data.shape[0] != self.size:
            raise ValueError('Wrong data size')

        if (fid in self.features) & (override == False):
            return

        self.features.update({fid: data})

    def get_feature(self, fid):
        """Return self.features[fid]
        """
        return self.features[fid]

    def representative_feature(self, fid, method):
        """Compute a statistical representative of the within-Foain feature

        Parameters
        ----------
        fid: string, feature id
        method: string, method used to compute a representative
                to be chosen among 'mean', 'max', 'median', 'min'

        """
        f = self.get_feature(fid)
        if method == "mean":
            return np.mean(f, 0)
        if method == "min":
            return np.min(f, 0)
        if method == "max":
            return np.max(f, 0)
        if method == "median":
            return np.median(f, 0)

    def integrate(self, fid):
        """Integrate  certain feature over the domain and returns the result

        Parameters
        ----------
        fid : string,  feature identifier,
              by default, the 1 function is integrataed, yielding domain volume

        Returns
        -------
        lsum = array of shape (self.feature[fid].shape[1]),
               the result

        """
        if fid == None:
            return np.sum(self.local_volume)
        ffid = self.features[fid]
        if np.size(ffid) == ffid.shape[0]:
            ffid = np.reshape(ffid, (self.size, 1))
        slv = np.reshape(self.local_volume, (self.size, 1))
        return np.sum(ffid * slv, 0)


class StructuredDomain(DiscreteDomain):
    """
    Besides DiscreteDomain attributed, StructuredDomain has a topology,
    which allows many operations (morphology etc.)
    """

    def __init__(self, dim, coord, local_volume, topology, did='',
                 referential=''):
        """Initialize structured domain instance

        Parameters
        ----------
        dim: int,
             the (physical) dimension of the domain
        coord: array of shape(size, em_dim),
               explicit coordinates of the domain sites
        local_volume: array of shape(size),
                      yields the volume associated with each site
        topology: sparse binary coo_matrix of shape (size, size),
                  that yields the neighboring locations in the domain
        did: string, optional,
             domain identifier
        referential: string, optional,
                     identifier of the referential of the coordinates system

        """
        DiscreteDomain.__init__(self, dim, coord, local_volume, id,
                                referential)

        # topology
        if topology is not None:
            if topology.shape != (self.size, self.size):
                raise ValueError('Incompatible shape for topological model')
        self.topology = topology

    def mask(self, bmask, did=''):
        """Returns a StructuredDomain instance that has been further masked
        """
        td = DiscreteDomain.mask(self, bmask)
        stopo = reduce_coo_matrix(self.topology, bmask)
        dd = StructuredDomain(self.dim, td.coord, td.local_volume,
                            stopo, did, self.referential)

        for fid in td.features.keys():
            dd.set_feature(fid, td.features.pop(fid))
        return dd


class NDGridDomain(StructuredDomain):
    """
    Particular instance of StructuredDomain, that receives
    3 additional variables:
    affine: array of shape (dim+1, dim+1),
            affine transform that maps points to a coordinate system
    shape: dim-tuple,
           shape of the domain
    ijk: array of shape(size, dim), int
         grid coordinates of the points

    This is to allow easy conversion to images when dim==3,
    and for compatibility with previous classes

    """

    def __init__(self, dim, ijk, shape, affine, local_volume, topology,
                referential=''):
        """Initialize ndgrid domain instance

        Parameters
        ----------
        dim: int,
             the (physical) dimension of the domain
        ijk: array of shape(size, dim), int
             grid coordinates of the points
        shape: dim-tuple,
           shape of the domain
        affine: array of shape (dim+1, dim+1),
            affine transform that maps points to a coordinate system
        local_volume: array of shape(size),
                      yields the volume associated with each site
        topology: sparse binary coo_matrix of shape (size, size),
                  that yields the neighboring locations in the domain
        referential: string, optional,
                     identifier of the referential of the coordinates system

        Fixme
        -----
        local_volume might be computed on-the-fly as |det(affine)|

        """
        # shape
        if len(shape) != dim:
            raise ValueError('Incompatible shape and dim')
        self.shape = shape

        # affine
        if affine.shape != (dim + 1, dim + 1):
            raise ValueError('Incompatible dim and affine')
        self.affine = affine

        # ijk
        if (np.size(ijk) == ijk.shape[0]) & (np.size(ijk) > 0):
            ijk = np.reshape(ijk, (ijk.size, 1))
            if (ijk.max(0) + 1 > shape).any():
                raise ValueError('Provided indices do not fit within shape')
        self.ijk = ijk

        # coord
        coord = idx_affine_coord(ijk, affine)

        StructuredDomain.__init__(self, dim, coord, local_volume, topology)

    def mask(self, bmask):
        """Returns an instance of self that has been further masked
        """
        if bmask.size != self.size:
            raise ValueError('Invalid mask size')

        svol = self.local_volume[bmask]
        stopo = reduce_coo_matrix(self.topology, bmask)
        sijk = self.ijk[bmask]
        DD = NDGridDomain(self.dim, sijk, self.shape, self.affine, svol,
                          stopo, self.referential)

        for fid in self.features.keys():
            f = self.features.pop(fid)
            DD.set_feature(fid, f[bmask])
        return DD

    def to_image(self, path=None, data=None):
        """Write itself as a binary image, and returns it

        Parameters
        ----------
        path: string, path of the output image, if any
        data: array of shape self.size,
              data to put in the nonzer-region of the image

        """
        if data is None:
            wdata = np.zeros(self.shape, np.int8)
        else:
            wdata = np.zeros(self.shape, data.dtype)
        wdata[self.ijk[:, 0], self.ijk[:, 1], self.ijk[:, 2]] = 1
        if data is not None:
            if data.size != self.size:
                raise ValueError('incorrect data size')
            wdata[wdata > 0] = data

        nim = Nifti1Image(wdata, self.affine)
        get_header(nim)['descrip'] = ('mask image representing domain %s'
                                      % self.id)
        if path is not None:
            save(nim, path)
        return nim

    def make_feature_from_image(self, path, fid=''):
        """Extract the information from an image to make it a domain a feature

        Parameters
        ----------
        path: string or Nifti1Image instance,
              the image from which one wished to extract data
        fid: string, optional
             identifier of the resulting feature.
             if '', the feature is not stored

        Returns
        -------
        the correponding set of values

        """
        if isinstance(path, basestring):
            nim = load(path)
        else:
            nim = path

        if (get_affine(nim) != self.affine).any():
            raise ValueError('nim and self do not have the same referential')

        data = nim.get_data()
        feature = data[self.ijk[:, 0], self.ijk[:, 1], self.ijk[:, 2]]
        if fid is not '':
            self.features[fid] = feature

        return feature
