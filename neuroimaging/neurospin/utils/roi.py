import numpy as np
import nifti

class ROI():
    """
    Temporary ROI class for fff
    Ultimately, it should be merged with the nipy class
    
    ROI definition requires
    - an identifier
    - an header (exactly a nifti header at the moment,
    though not everything is necessary)
    The ROI can be derived from a image or defined
    in the coordinate system implied by header.sform()

    roi.features is a dictionary of informations on the ROI elements.
    It is assumed that the ROI is sampled on a discrete grid, so that
    each feature is in fact a (voxel,feature_dimension) array
    """

    def __init__(self, id="roi", header=None):
        """
        roi = ROI(id='roi', header=None)
        - id (string): roi identifier
        - header (nipy header) : referential-defining information
        """
        self.id = id
        self.header = header
        self.features = dict()

    def check_header(self, image):
        """
        checks that the image is in the header of self
        INPUT:
        - image: (string) the path of an image
        """
        #print "check not implemented yet"
        eps = 1.e-15
        nim = nifti.NiftiImage(image)
        header  = nim.header
        b = True
        if (np.absolute(header['sform']-self.header['sform'])).max()>eps:
            b = False
        for d1,d2 in zip(header['dim'],self.header['dim']):
            if d1!=d2:
                b = False
        return b

    def from_binary_image(self, image):
        """
        Take all the <>0 sites of the image as the ROI
        INPUT:
        - image: (string) the path of an image
        """
        self.check_header(image)
        nim = nifti.NiftiImage(image)
        data = nim.data.T
        self.discrete = np.where(data)
        
    def from_position(self, position, radius):
        """
        a ball in the grid
        requires that the grid and header are defined
        """
        # check that the ref is defined
        if self.header==None:
            raise ValueError, "No defined referntial"

        # define the positions associated with the grid
        sform = self.header['sform']
        dim = self.header['dim'][1:4]
        grid = np.indices(dim)

        nvox = np.prod(dim)
        grid.shape = (3, nvox)
        grid = np.hstack((grid.T, np.ones((nvox, 1))))
        coord = np.dot(grid, sform.T)[:,:3]
        
        # finally derive the mask of the ROI
        dx = coord - position
        sqra = radius**2
        self.discrete = tuple(grid[np.sum(dx**2,1)<sqra,:3].T.astype('i'))
        

    def from_labelled_image(self,image,label):
        """
        All the voxels of the image that have the pre-defined label
        INPUT:
        image: a nifti label (discrete valued) image
        label (int): the desired label
        """
        self.check_header(image)
        nim = nifti.NiftiImage(image)
        data = nim.data.T
        self.discrete = np.where(data==label)
        
        
    def from_position_and_image(self,image,position):
        """
        the label on the image that is closest to the provided position
        INPUT:
        - image:  a nifti label (discrete valued) image
        - position: a position in the common space
        NOTE:
        everything could be performed in the image space
        """
        # check that the header is OK indeed
        self.check_header(image)
        
        # get the coordinates of the regions
        sform = self.header['sform']
        nim = nifti.NiftiImage(image)
        data = nim.data.T.astype('i')
        k = data.max()+1
        cent = np.array([np.mean(np.where(data==i),1) for i in range(k)])
        cent = np.hstack((cent,np.ones((k,1))))
        coord = np.dot(cent,np.transpose(sform))[:,:3]
        
        # find the best match
        dx = coord-position
        k = np.argmin(np.sum(dx**2,1))
        self.discrete = np.where(data==k)
        
    def make_image(self,name):
        """
        write a binary nifty image where the nonzero values are the ROI mask
        INPUT:
        the desired image name
        """
        #if name==None: name = "%s.nii" % self.id
        data = np.zeros(tuple(self.header['dim'][1:4]))
        data[self.discrete]=1
        data = np.reshape(data,tuple(self.header['dim'][1:4])).T
        nim = nifti.NiftiImage(data,self.header)
        nim.description = "ROI image"
        nim.save(name)

    def set_feature(self,fid,data):
        """
        INPUT:
        - fid (string): feature identifier, e.g.
        - data (array of shape (self.VolumeExtent))
        this function creates a reduced feature
        array corresponding to the ROI item
        OUTPUT:
        - ldata: array of shape (roi.nbvox,dim)
        the ROI-based feature
        """
        IVE = np.shape(data)
        VE = tuple(self.header['dim'][1:4])
        if IVE!= VE:
            raise ValueError, "incompatible dimension of provided data"
    
        ldata = data[self.discrete]
        self.features.update({fid:ldata})
        return ldata
        
    def set_feature_from_image(self,fid,image):
        """
        extract some roi-related information from an image
        INPUT:
        - fid: feature id
        - image(string): image name
        """
        nim = nifti.NiftiImage(image)  
        header = nim.header
        data = nim.asarray().T
        self.set_feature(fid,data)

    def get_feature(self,fid):
        """
        return the feature corrsponding to fid, if it exists
        """
        return self.features[fid]

    def set_feature_from_masked_data(self,fid,data,mask):
        """
        idem set_feature but the input data is thought to be masked
        """
        pass
        
    def representative_feature(self,fid,method="mean"):
        """
        Compute a statistical representative of the within-ROI feature
        """
        f = self.get_feature(fid)
        if method=="mean":
            return np.mean(f,0)
        if method=="min":
            return np.min(f,0)
        if method=="max":
            return np.max(f,0)
        if method=="median":
            return np.median(f,0)
        
    def plot_feature(self,fid):
        """
        boxplot the feature within the ROI
        """
        f = self.get_feature(fid)
        import matplotlib.pylab as mp
        mp.figure()
        mp.boxplot(f)
        mp.show()
        

    def __add__(self,other):
        """
        """
        pass

    def __multiply__(self,other):
        """
        """
        pass
        

        

class WeightedROI(ROI):
    """
    ROI where a weighting is defined on the voxels
    """
    
    def __init__(self,id="roi",header=None,grid = None):
        """
        """
        
        pass

class MultipleROI():
    """
    This is  a class to deal with multiple ROIs defined in a given space
    mroi.header is assumed to provide all the referential information
    (this should be changed in the future),
    so that the mroi is basically defined as a multiple sets of 3D coordinates
    finally, there is an associated feature dictionary.
    Typically it is assumed that each feature is an
    (roi,feature_dim) array, i.e. each roi is assumed homogeneous
    wrt the feature
    In the future, it might be possible to complexify the structure
    to model within-ROI variance
    """

    def __init__(self, id="roi", k=0,header=None):
        """
        roi = MultipleROI(id='roi', header=None)
        - id (string): roi identifier
        - k: number of rois that are included in the structure 
        - header (nipy header) : referential-defining information
        """
        self.id = id
        self.k = k
        self.header = header
        self.discrete = []
        self.features = dict()

    def check_features(self):
        """
        check that self.features have the coorect size
        i.e; f.shape[0]=self.k for f in self.features
        """
        fids = self.features.keys()
        for fid in fids:
            if self.features[fid].shape[0]!=self.k:
                print "removing feature %s, which has incorect size" %fid 
                self.features.pop(fid)

    def check_header(self, image):
        """
        checks that the image is in the header of self
        INPUT:
        - image: (string) the path of an image
        """
        #print "check not implemented yet"
        eps = 1.e-15
        nim = nifti.NiftiImage(image)
        header = nim.header
        b = True

        if (np.absolute(header['sform']-self.header['sform'])).max()>eps:
            b = False
        for d1,d2 in zip(header['dim'],self.header['dim']):
            if d1!=d2:
                b = False
        return b

    def from_labelled_image(self,image,labels=None,add=True):
        """
        All the voxels of the image that have non-zero-value
        self.k becomes the number of values of the (discrete) image
        INPUT:
        - image (string): a nifti label (discrete valued) image
        -labels=None : the set of image labels that
        shall be used as ROI definitions
        By default, all the image labels are used
        note that this can be used to append features,
        when rois are already defined
        """
        self.check_header(image)
        nim = nifti.NiftiImage(image)
        data = nim.data.T
        if labels==None:
            udata = np.unique(data[data>0])
            if add: self.k+=np.size(udata)
            for k in range(np.size(udata)):
                self.discrete.append(np.where(data==udata[k]))
        else:
            if add: self.k+=np.size(labels)
            for k in range(np.size(labels)):
                if np.sum(data==labels[k])>0:
                    self.discrete.append(np.where(data==labels[k]))
                else:
                    raise ValueError, "Sorry I don't take empty ROIs"
        self.check_features()

    def as_multiple_balls(self, position, radius):
        """
        self.as_multiple_balls(position, radius)
        Given a set of positions and radii, defines one roi
        at each (position/radius) couple 
        INPUT:
        position: array of shape (k,3): the set of positions
        radius: array of shape (k): the set of radii
        """
        # check that the ref is defined
        if self.header==None:
            raise ValueError, "No defined referential"

        if np.shape(position)[0]!=np.size(radius):
            raise ValueError, "inconsistent position/radius definition"

        if np.shape(position)[1]!=3:
            raise ValueError, "Sorry, I take only 3D regions: provide the positions as a (k,3) array"

        self.k=np.size(radius)
        
        # define the positions associated with the grid
        sform = self.header['sform']
        dim = self.header['dim'][1:4]
        grid = np.indices(dim)

        nvox = np.prod(dim)
        grid.shape = (3, nvox)
        grid = np.hstack((grid.T, np.ones((nvox, 1))))
        coord = np.dot(grid, sform.T)[:,:3]
        
        # finally derive the mask of the ROI
        for k in range(self.k):
            dx = coord - position[k]
            sqra = radius[k]**2
            self.discrete.append(tuple(grid[np.sum(dx**2,1)<sqra,:3].T.astype('i')))
        self.set_roi_feature('position',position)
        self.set_roi_feature('radius',radius)
        self.check_features()


    def append_balls(self,position, radius):
        """
        idem self.as_multiple_balls, but the ROIs are added
        """
        if self.header==None:
            raise ValueError, "No defined referential"

        if np.shape(position)[0]!=np.size(radius):
            raise ValueError, "inconsistent position/radius definition"

        if np.shape(position)[1]!=3:
            raise ValueError, "Sorry, I take only 3D regions: provide the positions as a (k,3) array"

        self.k += np.size(radius)
        
        # define the positions associated with the grid
        sform = self.header['sform']
        dim = self.header['dim'][1:4]
        grid = np.indices(dim)

        nvox = np.prod(dim)
        grid.shape = (3, nvox)
        grid = np.hstack((grid.T, np.ones((nvox, 1))))
        coord = np.dot(grid, sform.T)[:,:3]
        
        # finally derive the mask of the ROI
        for k in range(np.size(radius)):
            dx = coord - position[k]
            sqra = radius[k]**2
            self.discrete.append(tuple(grid[np.sum(dx**2,1)<sqra,:3].T.astype('i')))

        if self.features.has_key('position'):
            self.complete_feature('position',position)
        if self.features.has_key('radius'):
            self.complete_feature('radius',radius)

        self.check_features()

    def complete_feature(self,fid,values):
        """
        completes a feature by appending the values
        """
        f = self.features.pop(fid)
        if values.shape[0]==np.size(values):
            values = np.reshape(values,(np.size(values),1))
        if f.shape[1]!=values.shape[1]:
            raise ValueError, "incompatible dimensions for the features"
        f = np.vstack((f,values))
        self.set_roi_feature(fid,f)
        
        
    def make_image(self,name):
        """
        write a int nifty image where the nonzero values are the ROIs
        INPUT:
        - the desired image name
        NOTE:
        - the background values are set to -1
        - the ROIs values are set as [0..self.k-1]
        """
        data = -np.ones(tuple(self.header['dim'][1:4]),'i')
        for k in range(self.k):
            data[self.discrete[k]]=k
        data = np.reshape(data,tuple(self.header['dim'][1:4])).T
        nim = nifti.NiftiImage(data,self.header)
        nim.description = "ROI image"
        nim.save(name)
  
    def set_roi_feature(self,fid,data):
        """
        INPUT:
        - fid (string): feature identifier, e.g.
        - data: array of shape(self.k,p),with p>0
        this function simply stores data 
        """
        if data.shape[0]!=self.k:
            print data.shape[0],self.k,fid
            raise ValueError, "Incompatible information of the provided data"
        if np.size(data)==self.k:
            data = np.reshape(data,(self.k,1))
        self.features.update({fid:data})

    def set_feature_from_image(self,fid,image,method='average'):
        """
        extract some roi-related information from an image
        INPUT:
        - fid: feature id
        - image(string): image name
        - method='average' (string) : take the roi feature as
        the average feature over the ROI
        """
        self.check_header(image)
        nim = nifti.NiftiImage(image)  
        header = nim.header
        data = nim.asarray().T
        ldata = np.zeros((self.k,1))
        for k in range(self.k):
            ldata[k] = np.mean(data[self.discrete[k]])
        self.set_roi_feature(fid,ldata)

    def plot_feature(self,fid):
        """
        boxplot the feature within the ROI
        Note that this assumes a 1-d feature
        """
        f = self.features[fid]
        if f.shape[1]>1:
            raise ValueError, "cannot plot multi-dimensional feature for the moment"
        import matplotlib.pylab as mp
        mp.figure()
        mp.bar(np.arange(self.k),f)
        mp.show()

    def clean(self,valid):
        """
        remove the regions for which valid<=0
        """
        if np.size(valid)!=self.k:
            raise ValueError, "the valid marker does not have the correct size"

        self.discrete = [self.discrete[k] for k in range(self.k) if valid[k]]
        self.k = np.sum(valid.astype('i'))
        for fid in self.features.keys():
            f = self.features.pop(fid)
            f = f[valid]
            self.set_roi_feature(fid,f)
        self.check_features()

    def get_size(self):
        """
        return the number of voxels per ROI in one array
        """
        size = np.zeros(self.k)
        for k in range(self.k):
            size[k] = np.shape(self.discrete[k],0)
        return size

def test1(verbose = 0):
    nim = nifti.NiftiImage("/tmp/spmT_0024.img")
    header = nim.header
    dat = nim.asarray().T
    roi = ROI("myroi",header)
    roi.from_position(np.array([0,0,0]),5.0)
    roi.make_image("/tmp/myroi.nii")
    roi.set_feature_from_image('activ',"/tmp/spmT_0024.img")
    if verbose: roi.plot_feature('activ')
    return roi

def test2(verbose=0):
    nim = nifti.NiftiImage("/tmp/spmT_0024.img")
    header = nim.header
    roi = ROI(header=header)
    roi.from_labelled_image("/tmp/blob.nii",1)
    roi.make_image("/tmp/roi2.nii")
    
    roi.from_position_and_image("/tmp/blob.nii",np.array([0,0,0]))
    roi.make_image("/tmp/roi3.nii")

    roi.set_feature_from_image('activ',"/tmp/spmT_0024.img")
    mactiv = roi.representative_feature('activ')
    if verbose: roi.plot_feature('activ')
    return roi


def test_mroi1(verbose=0):
    nim =  nifti.NiftiImage("/tmp/blob.nii")
    header = nim.header
    mroi = MultipleROI(header=header)
    mroi.from_labelled_image("/tmp/blob.nii")
    mroi.make_image("/tmp/mroi.nii")
    mroi.set_feature_from_image('activ',"/tmp/spmT_0024.img")
    if verbose: mroi.plot_feature('activ')
    return mroi

def test_mroi2(verbose=0):
    nim =  nifti.NiftiImage("/tmp/blob.nii")
    header = nim.header
    mroi = MultipleROI(header=header)
    pos = 1.0*np.array([[10,10,10],[0,0,0],[20,0,20],[0,0,35]])
    rad = np.array([5.,6.,7.,8.0])
    mroi.as_multiple_balls(pos,rad)
    mroi.append_balls(np.array([[-10.,0.,10.]]),np.array([7.0]))
    mroi.make_image("/tmp/mroi.nii")
    mroi.set_feature_from_image('activ',"/tmp/spmT_0024.img")
    if verbose: mroi.plot_feature('activ')
    return mroi

def test_mroi3(verbose=0):
    nim =  nifti.NiftiImage("/tmp/blob.nii")
    header = nim.header
    mroi = MultipleROI(header=header)
    mroi.as_multiple_balls(np.array([[-10.,0.,10.]]),np.array([7.0]))
    mroi.from_labelled_image("/tmp/blob.nii",np.arange(1,20))
    mroi.from_labelled_image("/tmp/blob.nii",np.arange(31,50))
    mroi.make_image("/tmp/mroi.nii")
    mroi.set_feature_from_image('activ',"/tmp/spmT_0024.img")
    if verbose: mroi.plot_feature('activ')
    valid = np.random.randn(mroi.k)>0.1
    mroi.clean(valid)
    return mroi
