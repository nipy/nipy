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
        self.k = np.maximum(k, 0)
        if len(coord) != self.k:
            raise ValueError, 'coord should have length %d' %self.k
        
        # number of discrete elements
        self.size = np.zeros(self.k)
        for k in range(self.k):
            self.size[k] = coord[k].shape[0]

        # coordinates
        for k in range(self.k):
            if np.size(coord[k]) == coord[k].shape[0]:
                coord[k] = np.reshape(coord[k], (np.size(coord[k]), 1))

        self.em_dim = coord[0].shape[1]
        for k in range(self.k):
            if coord[k].shape[1] != em_dim:
                raise ValueError, 'Inconsistant coordinate dimensions'        
        if self.em_dim<dim:
            raise ValueError, 'Embedding dimension cannot be smaller than dim'
        for k in range(self.k):
            self.coord[k] = coord[k]

        # volume
        if len(local_volume) !=k:
            raise ValueError(), 'local_volume should have a length %d' %self.k
        self.local_volume = []
        for k in range(self.k):
            if np.size(local_volume[k])!= self.size[k]:
                raise ValueError, "Inconsistent Volume size"
            self.local_volume.append(np.ravel(local_volume[k]))
        
        # topology
        if topology is not None:
            self.topology = []
            if len(topology) !=k:
                raise ValueError(), 'Topology should have length %d' %self.k
            for k in range(self.k):
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

    def select(self, valid):
        """
        returns an instance of multiple_ROI
        with only the subset of ROIs for which valid

        Parameters
        ----------
        
        """
        if bmask.size != self.size:
            raise ValueError, 'Invalid mask size'

        svol = self.local_volume[bmask]
        stopo = reduce_coo_matrix(self.topology, bmask)
        scoord = self.coord[bmask]
        DD = DiscreteDomain(self.dim, scoord, svol, stopo, self.referential)

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
                chosen among 'mean', 'max', 'median', 'min' 
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




    def set_feature(self):
        """
        """
        pass
    
    def get_features(self):
        """
        """
        pass

    def check_features(self):
        """
        """
        pass

    def representatiive_feature(self):
        """
        """
        pass

    def remove_feature(self):
        """
        """
        pass

    def argmax_feature(self):
        """
        """
        pass

    def plot_feature(self):
        """
        """
        pass

    def clean(self):
        """
        """
        pass

    def integrate(self):
        """
        """
        pass

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

