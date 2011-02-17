"""
This module define the StructuredDomain class,
that represents a generic neuroimaging kind of domain
This is meant to provide a unified API to deal with n-d imaged and meshes.


Author: Bertrand Thirion, 2010
"""
import numpy as np
import scipy.sparse as sp

from nibabel import load, Nifti1Image, save

import nipy.neurospin.graph as fg

##############################################################
# Ancillary functions
##############################################################

def smatrix_from_3d_array(mask, nn=18):
    """
    Create a sparse adjacency matrix from an array

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
    """
    Create a sparse adjacency matrix from 3d index system
    
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
    G = fg.WeightedGraph(ijk.shape[0])
    G.from_3d_grid(ijk, nn)
    return G.to_coo_matrix()

def smatrix_from_nd_array(mask, nn=0):
    """
    Create a sparse adjacency matrix from an arbitrary nd array

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
    """
    Create a sparse adjacency matrix from nd index system

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
    nidx = idx-idx.min(0)
    
    eA = []
    eB = []

    # compute the edges in each possible direction
    for d in range(dim):
        mi = nidx.max(0)+2
        a = np.hstack((1, np.cumprod(mi[:dim-1])))
        v1 = np.dot(nidx, a)
        assert(np.size(v1)==np.size(np.unique(v1)))
        o1 = np.argsort(v1)
        sv1 = v1[o1]
        nz = np.squeeze(np.nonzero(sv1[:n-1]-sv1[1:]==-1))
        nz = np.reshape(nz,np.size(nz))
        eA.append(o1[nz])
        eB.append(o1[nz+1])
        nidx = np.roll(nidx, 1, 1)
        
    eA = np.concatenate(eA)
    eB =  np.concatenate(eB)
    E = 2*np.size(eA)
    # create a graph structure
    if E==0:
        return sp.coo_matrix((n, n))

    edges = np.vstack((np.hstack((eA, eB)), np.hstack((eB, eA)))).T
    weights = np.ones(E)
    G = fg.WeightedGraph(n, edges, weights)
    return G.to_coo_matrix()
    
def array_affine_coord(mask, affine):
    """
    Compute coordinates from a boolean array and an affine transform

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
    """
    Compute coordinates from a set of indexes and an affine transform

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
    coord = np.dot(hidx, affine.T)[:,:-1]
    return coord

def reduce_coo_matrix(mat, mask):
    """
    reduce a supposedly coo_matrix to the vertices in the mask

    Parameters
    ----------
    mat: sparse coo_matrix,
         input matrix
    mask: boolean array of shape mat.shape[0],
          desired elements 
    """
    G = fg.wgraph_from_coo_matrix(mat)
    K = G.subgraph(mask)
    return K.to_coo_matrix()
    

#################################################################
# Functions to instantiate StructuredDomains
#################################################################


def domain_from_image(mim, nn=18):
    """
    return a StructuredDomain instance from the input mask image

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
    return domain_from_array( iim.get_data(), iim.get_affine(), nn)
    
def domain_from_array(mask, affine=None, nn=0):
    """
    return a StructuredDomain from an n-d array

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
        affine =  np.eye(dim + 1)
    mask = mask>0
    vol = np.absolute(np.linalg.det(affine))*np.ones(np.sum(mask))
    coord = array_affine_coord(mask, affine)
    topology = smatrix_from_nd_array(mask) 
    return StructuredDomain(dim, coord, vol, topology)

def grid_domain_from_array(mask, affine=None, nn=0):
    """
    return a NDGridDomain from an n-d array

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
        affine =  np.eye(dim + 1)
        
    mask = mask>0
    ijk = np.array(np.where(mask)).T
    vol = np.absolute(np.linalg.det(affine))*np.ones(np.sum(mask))
    topology = smatrix_from_nd_idx(ijk, nn) 
    return NDGridDomain(dim, ijk, shape, affine, vol, topology)

def grid_domain_from_image(mim, nn=18):
    """
    return a NDGridDomain instance from the input mask image

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
    return  grid_domain_from_array(iim.get_data(), iim.get_affine(), nn)


def domain_from_mesh(mesh):
    """
    Instantiate a StructuredDomain from a gifti mesh
    """
    pass


################################################################
# StructuredDomain class
################################################################

class DiscreteDomain(object):
    """
    Descriptor of a certain domain that consists of discrete elements that
    are characterized by a coordinate system and a topology:
    the coordinate system is specified through a coordinate array
    the topology encodes the neighboring system

    fixme
    -----
    check that local_volume is positive
    """
    
    def __init__(self, dim, coord, local_volume, rid='', referential=''):
        """
        Parameters
        ----------
        dim: int,
             the (physical) dimension of the domain
        coord: array of shape(size, em_dim),
               explicit coordinates of the domain sites
        local_volume: array of shape(size),
                      yields the volume associated with each site
        rid: string, optional,
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
        self.em_dim = coord.shape[1]
        if self.em_dim<dim:
            raise ValueError, 'Embedding dimension cannot be smaller than dim'
        self.coord = coord

        # volume
        if np.size(local_volume)!= self.size:
            raise ValueError, "Inconsistent Volume size"
        self.local_volume = np.ravel(local_volume) 
        
        
        self.rid = rid
        self.referential = referential
        self.features = {}
        
    def get_coord(self):
        """ returns self.coord
        """
        return self.coord

    def get_volume(self):
        """ returns self.local_volume
        """
        return self.local_volume

    def mask(self, bmask, rid=''):
        """
        returns an DiscreteDomain instance that has been further masked
        """
        if bmask.size != self.size:
            raise ValueError, 'Invalid mask size'

        svol = self.local_volume[bmask]
        scoord = self.coord[bmask]
        DD = DiscreteDomain(self.dim, scoord, svol, rid,  self.referential)

        for fid in self.features.keys():
            f = self.features.pop(fid)
            DD.set_feature(fid, f[bmask])
        return DD

    def set_feature(self, fid, data, override=True):
        """
        Append a feature 'fid'
        
        Parameters
        ----------
        fid: string,
             feature identifier
        data: array of shape(self.size, p) or self.size
              the feature data 
        """
        if data.shape[0]!=self.size:
            raise ValueError, 'Wrong data size'
        
        if (self.features.has_key(fid)) & (override==False):
            return
            
        self.features.update({fid:data})
            

    def get_feature(self, fid):
        """return self.features[fid]
        """
        return self.features[fid]

    def representative_feature(self, fid, method):
        """
        Compute a statistical representative of the within-Foain feature

        Parameters
        ----------
        fid: string, feature id
        method: string, method used to compute a representative
                to be chosen among 'mean', 'max', 'median', 'min' 
        """
        f = self.get_feature(fid)
        if method=="mean":
            return np.mean(f, 0)
        if method=="min":
            return np.min(f, 0)
        if method=="max":
            return np.max(f, 0)
        if method=="median":
            return np.median(f, 0)

    def integrate(self, fid):
        """
        Integrate  certain feature over the domain and returns the result

        Parameters
        ----------
        fid : string,  feature identifier,
              by default, the 1 function is integrataed, yielding domain volume

        Returns
        -------
        lsum = array of shape (self.feature[fid].shape[1]),
               the result
        """
        if fid==None:
            return np.sum(self.local_volume)
        ffid = self.features[fid]
        if np.size(ffid)==ffid.shape[0]:
            ffid = np.reshape(ffid, (self.size, 1))
        slv = np.reshape(self.local_volume, (self.size, 1))
        return np.sum(ffid*slv, 0)


class StructuredDomain(DiscreteDomain):
    """
    Besides DiscreteDomain attributed, StructuredDomain has a topology,
    which allows many operations (morphology etc.)
    """
    
    def __init__(self, dim, coord, local_volume, topology, did='',
                 referential=''):
        """
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
                raise ValueError, 'Incompatible shape for topological model'
        self.topology = topology
        
    def mask(self, bmask, did=''):
        """
        returns a StructuredDomain instance that has been further masked
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
        """
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
        if len(shape)!=dim:
            raise ValueError, 'Incompatible shape and dim'
        self.shape = shape

        # affine
        if affine.shape != (dim+1, dim+1):
            raise ValueError, 'Incompatible dim and affine'
        self.affine = affine
        
        # ijk
        if np.size(ijk)==ijk.shape[0]:
            ijk = np.reshape(ijk, (ijk.size,1))
        if (ijk.max(0)+1>shape).any():
            raise ValueError, 'Provided indices do not fit within shape'
        self.ijk = ijk

        # coord
        coord = idx_affine_coord(ijk, affine)

        StructuredDomain.__init__(self, dim, coord, local_volume, topology)

        
    def mask(self, bmask):
        """
        returns an instance of self that has been further masked
        """
        if bmask.size != self.size:
            raise ValueError, 'Invalid mask size'

        svol = self.local_volume[bmask]
        stopo = reduce_coo_matrix(self.topology, bmask)
        sijk = self.ijk[bmask]
        DD = NDGridDomain(self.dim, sijk, self.shape, self.affine, svol,
                          stopo, self.referential)

        for fid in self.features.keys():
            f = self.features.pop(fid)
            DD.set_feature(fid, f[bmask])
        return DD

    def to_image(self, path=None):
        """
        Write itself as an image, and returns it
        """
        data = np.zeros(self.shape).astype(np.int8)
        data[self.ijk[:,0], self.ijk[:,1], self.ijk[:,2]] = 1
        nim = Nifti1Image(data, self.affine)
        nim.get_header()['descrip'] = 'mask image'
        if path is not None:
            save(nim, path)
        return nim
