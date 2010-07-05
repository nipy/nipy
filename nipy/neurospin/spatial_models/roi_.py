# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import numpy as np
from nipy.io.imageformats import load, save, Nifti1Image 
from discrete_domain import *   

###############################################################################
# class DiscreteROI 
###############################################################################


class DiscreteROI(DiscreteDomain):

    def __init__(self, dim, coord, local_volume, topology, referential='',
                 id= ''):
        """
        This is simply a discrete domain, with an identifier (optional)
        """
        self.id = id
        DiscreteDomain.__init__(self, dim, coord, local_volume, topology,
                                referential)

    

class DiscreteGridROI(NDGridDomain):
    """
    fixme : it should depend from DiscreteROI with additional stuff
    in that case, NDGridDomain will probably have to be removed
    """
    
    def __init__(self, dim, ijk, shape, affine, local_volume, topology,
                 referential='', id=''):
        """
        """
        self.id = id
        NDGridDomain.__init__(self, dim, ijk, shape, affine, local_volume,
                              topology, referential)

    def check_header(self, image_path):
        """
        checks that the image is in the header of self

        Parameters
        ----------
        image_path: (string) the path of an image (nifti)

        Returns
        -------
        True if the affine and shape of the given nifti file correpond
        to the ROI's.
        """
        eps = 1.e-15
        nim = load(image_path)
        if (np.absolute(nim.get_affine()-self.affine)).max()>eps:
            return False
        if self.shape is not None:
            return np.all(np.equal(nim.get_shape(), self.shape))
        return True

    def to_nifti1image(self, image_path=None):
        """
        returns and possibly write a binary nifti1image image of self

        Parameters
        -----------
        image_path: string, optional, 
                    the desired image name
        Note
        ----
        Only in 3D at the moment
        """
        if self.dim !=3:
            raise ValueError, "Refusing to create a Nifti1Image \
            with %s dimensions" %self.dim
        data = np.zeros(self.shape)
        data[self.ijk[:, 0], self.ijk[:, 1], self.ijk[:, 2]] = 1
        
        wim = Nifti1Image(data, self.affine)
        wim.get_header()['descrip'] = "ROI %s image"%self.id
        if image_path !=None:
            save(wim, image_path)
        return wim

    def to_array(self):
        """
        returns bool array that represents self
        """
        data = np.zeros(self.shape)
        for i in self.ijk:
            data[i] = 1
        return data
        
    def set_feature_from_image():
        """
        extract some roi-related information from an image

        Parameters
        -----------
        fid: string
            feature id
        image: string
            image path
        """
        if self.dim != 3:
            raise ValueError, "Not sure I can do this, as self.dim is not 3"
        
        self.check_header(image_path)
        nim = load(image_path)  
        data = nim.get_data()
        self.set_feature(fid, data)
        

def discrete_groi_ball( shape, affine, position, radius, id=''):
    """
    """
    pass


def discrete_groi_from_mask_image( mask, id=''):
    """
    """
    pass

def discrete_groi_from_labelled_image( image, label, id=''):
    """
    """
    pass



###############################################################################
# class MultipleROI 
###############################################################################



class MultipleROI(object):
    """
    idem DiscreteDomain (and DiscreteROI), but here coord,
    local_volume and topology are k-length lists of a arrays and
    coo-matrices
    """
    def __init__(self, dim, k, coord, local_volume, topology=None,
                 referential='', id='' ):
        """
        Parameters
        ----------
        dim: int,
             the (physical) dimension of the domain
        k: int,
           the number of ROIs consiedered in the model 
        coord: list of k arrays of shape(size[k], em_dim),
               explicit coordinates of the domain sites
        local_volume: list of k arrays of shape(size[k]),
                      yields the volume associated with each site
        topology: list of k sparse coo_matrices of shape (size[k], size[k]),
                  that yields the neighboring locations in the ROIs
        referential: string, optional,
                     identifier of the referential of the coordinates system
        id: string, optional,
                     identifier of the set of ROIs
                     
        Caveat
        ------
        em_dim may be greater than dim e.g. (meshes coordinate in 3D)

        Fixme:
        ------
        Local_volume and topology should be optional
        """
        # dimension
        self.dim = dim

        #k 
        self.k = int(np.maximum(k, 0))
        if len(coord) != self.k:
            raise ValueError, 'coord should have length %d' %self.k
        
        # number of discrete elements
        self.size = np.zeros(self.k, dtype=np.int)
        for k in range(self.k):
            self.size[k] = coord[k].shape[0]

        # coordinates
        for k in range(self.k):
            if np.size(coord[k]) == coord[k].shape[0]:
                coord[k] = np.reshape(coord[k], (np.size(coord[k]), 1))

        self.em_dim = coord[0].shape[1]
        for k in range(self.k):
            if coord[k].shape[1] != self.em_dim:
                raise ValueError, 'Inconsistant coordinate dimensions'        
        if self.em_dim<dim:
            raise ValueError, 'Embedding dimension cannot be smaller than dim'
        self.coord = []
        for k in range(self.k):
            self.coord.append(coord[k])

        # volume
        if len(local_volume) != self.k:
            raise ValueError, 'local_volume should have a length %d' %self.k
        self.local_volume = []
        for k in range(self.k):
            if np.size(local_volume[k])!= self.size[k]:
                raise ValueError, "Inconsistent Volume size"
            self.local_volume.append(np.ravel(local_volume[k]))
        
        # topology
        if topology is not None:
            self.topology = []
            if len(topology) != self.k:
                raise ValueError, 'Topology should have length %d' %self.k
            for k in range(self.k):
                if self.size[k]==1:
                    continue
                if topology[k].shape != (self.size[k], self.size[k]):
                    raise ValueError, 'Incorrect shape for topological model'
                self.topology.append(topology[k])

        self.referential = referential
        self.features = {}
        
    def get_coord(self, k):
        """ returns self.coord[k]
        """
        if k>self.k-1:
            raise ValueError, 'works only for k<%d'%self.k
        return self.coord[k]

    def get_volume(self):
        """ returns self.local_volume[k]
        """
        if k>self.k-1:
            raise ValueError, 'works only for k<%d'%self.k
        return self.local_volume[k]

    def select(self, valid, id=''):
        """
        returns an instance of multiple_ROI
        with only the subset of ROIs for which valid

        Parameters
        ----------
        valid: array of shape (self.k),
               which ROIs will be included in the output
        id: string, optional,
            identifier of the output instance 
        """
        if valid != self.k:
            raise ValueError, 'Invalid size for valid'

        svol = [self.local_volume[k] for k in range(self.k) if valid[k]]
        stopo = [self.topology[k] for k in range(self.k) if valid[k]]
        scoord = [self.coord[k] for k in range(self.k) if valid[k]]
        DD = Multiple_ROI(self.dim, valid.sum(), scoord, svol, stopo,
                          self.referential, id)

        for fid in self.features.keys():
            f = self.features.pop(fid)
            sf = [f[k] for k in range(self.k) if valid[k]]
            DD.set_feature(fid, sf)
        return DD

    def set_feature(self, fid, data, override=True):
        """
        Append a feature 'fid'
        
        Parameters
        ----------
        fid: string,
             feature identifier
        data: list of self.k arrays of shape(self.size[k], p) or self.size[k]
              the feature data 
        """
        if len(data) != self.k:
            raise ValueError, 'data should have length k'
        for k in range(self.k):
            if data[k].shape[0]!=self.size[k]:
                raise ValueError, 'Wrong data size'
        
        if (self.features.has_key(fid)) & (override==False):
            return
            
        self.features.update({fid:data})
            

    def get_feature(self, fid, k=None):
        """return self.features[fid]
        """
        if k==None:
            return self.features[fid]
        if k>self.k-1:
            raise ValueError, 'works only for k<%d'%self.k
        return self.features[fid][k]

    def representative_feature(self, fid, method='mean'):
        """
        Compute a statistical representative of the within-Foain feature

        Parameters
        ----------
        fid: string, feature id
        method: string, method used to compute a representative
                chosen among 'mean', 'max', 'median', 'min' 
        """
        rf = []
        for k in range(self.k):
            f = self.get_feature(fid, k)
            if method=="mean":
                rf.append(np.mean(f, 0))
            if method=="min":
                rf.append( np.min(f, 0))
            if method=="max":
                rf.append(np.max(f, 0))
            if method=="median":
                rf.append(np.median(f, 0))
        return np.array(rf)
    
    def remove_feature(self):
        """ Remove a certain feature
        """
        return self.features.pop(fid)
        

    def argmax_feature(self, fid):
        """ Return the list  of roi-level argmax of feature called fid
        """
        af = [argmax(self.feature[fid][k]) for k in range(self.k)]
        return np.array(af)


    def integrate(self, fid=None):
        """
        Integrate  certain feature on each ROI and return the k results

        Parameters
        ----------
        fid : string,  feature identifier,
              by default, the 1 function is integrataed, yielding ROI volumes

        Returns
        -------
        lsum = array of shape (self.k, self.feature[fid].shape[1]),
               the results
        """
        if fid==None:
            vol = [np.sum(self.local_volume[k]) for k in range(self.k)] 
            return (np.array(vol))
        lsum = []
        for k in range(self.k):
            slvk = np.reshape(self.local_volume[k], (self.size[k], 1))
            sfk = self.features[fid][k]
            if sfk.size == sfk.shape[0]:
                sfk = np.reshape(sfk,(self.size[k], 1))
            sumk = np.sum(sfk*slvk, 0)
            lsum.append(sumk)
        return np.array(lsum)
        
    def check_features(self):
        """
        """
        pass

    def plot_feature(self):
        """
        """
        pass

def mroi_from_label_image(mim, nn=18):
    """
    return a MultipleROI instance from the input mask image

    Parameters
    ----------
    mim: NiftiIImage instance, or string path toward such an image
         supposedly a label image
    nn: int, optional
        neighboring system considered from the image
        can be 6, 18 or 26
        
    Returns
    -------
    The MultipleROI  instance

    Note
    ----
    Only nonzero labels are considered
    """
    if isinstance(mim, basestring):
        iim = load(mim)
    else :
        iim = mim

    return mroi_from_array(iim.get_data(), iim.get_affine(), nn)
    
def mroi_from_array(labels, affine=None, nn=0):
    """
    return a DiscreteDomain from an n-d array
    
    Parameters
    ----------
    label: np.array instance
          a supposedly boolean array that yields the regions
    affine: np.array, optional
            affine transform that maps the array coordinates
            to some embedding space
            by default, this is np.eye(dim+1, dim+1)
    nn: int, neighboring system considered,
        unsued at the moment

    Note
    ----
    Only nonzero labels are considered
    """
    dim = len(labels.shape)
    shape = labels.shape
    if affine is None:
        affine =  np.eye(dim + 1)
    ul = np.unique(labels[labels!=0])
    k = np.size(ul)
    vol = []
    coord = []
    topology = []
    vvol = np.absolute(np.linalg.det(affine))
    for q in ul:
        vol.append(vvol*np.ones(np.sum(labels==q)))
        coord.append(array_affine_coord(labels==q, affine))
        topology.append(smatrix_from_nd_array(labels==q, nn))
    return MultipleROI(dim, k, coord, vol, topology)
    

def mroi_from_balls(domain, positions, radii):
    """
    Create discrete ROIs as a set of balls within a certain coordinate systems

    Parameters
    ----------
    domain: DiscreteDomain instance,
            the description of a discrete domain
    positions: array of shape(k, dim):
               the positions of the balls
    radii: array of shape(k):
           the sphere radii
    """
    # checks
    if np.size(positions)==positions.shape[0]:
        positions = np.reshape(positions, (positions.size), 1)
    if positions.shape[1] !=  domain.em_dim:
        raise ValueError, 'incompatible dimensions for domain and positions'
    if positions.shape[0] != np.size(radii):
        raise ValueError, 'incompatible positions and radii provided'
    coord = []
    vol = []
    topology = []
    for k in range(radii.size): 
        ik = np.sum((domain.coord-positions[k])**2, 1)<radii[k]**2
        coord.append(domain.coord[ik])
        vol.append(domain.local_volume[ik])
        topology.append(reduce_coo_matrix(domain.topology, ik))
    return MultipleROI(domain.dim, radii.size, coord, vol, topology)


class MultipleGridRoi(MultipleROI):

    def __init__(self, dim, k, idx, shape, affine, topology,
                 referential='', id=''  ):
        pass

    def check_header(self, image_path):
        """
        """
        pass
    
    def to_nifti1image(self, image_path=None):
        """
        """
        pass
    
    def to_array(self):
        """
        """
        pass
        
    def set_feature_from_image():
        """
        """
        pass

    

def mgrid_roi_from_label_image():
    """
    """
    pass

def mgrid_roi_from_balls():
    """
    """
    pass

