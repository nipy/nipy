# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np
import scipy.stats as sp_stats

from nipy.neurospin.graph.field import Field
from nipy.neurospin.image import apply_affine
from nipy.neurospin.utils import emp_null
from nipy.neurospin.glm import glm
from nipy.neurospin.group.permutation_test import \
     permutation_test_onesample, permutation_test_twosample

# FIXME: rename permutation_test_onesample class
#so that name starts with upper case

# Use the brifti image object
from nibabel import Nifti1Image as Image 


################################################################################
# Cluster statistics 
################################################################################


def bonferroni(p, n):
    return np.minimum(1., p*n)
    
def simulated_pvalue(t, simu_t): 
    return 1 - np.searchsorted(simu_t, t)/float(np.size(simu_t))


def cluster_stats(zimg, mask, height_th, height_control='fpr', 
                  cluster_th=0, nulls={}):
    """
    Return a list of clusters, each cluster being represented by a
    dictionary. Clusters are sorted by descending size order. Within
    each cluster, local maxima are sorted by descending depth order.

    Parameters
    ----------
    zimg: z-score image
    mask: mask image 
    height_th: cluster forming threshold
    height_control: string 
            false positive control meaning of cluster forming 
            threshold: 'fpr'|'fdr'|'bonferroni'|'none'
    cluster_th: cluster size threshold
    null_s : cluster-level calibration method: None|'rft'|array

    Note
    ----
    This works only with three dimsnionsla data
    """
    # Masking
    if len(mask.get_shape())>3:
        xyz = np.where((mask.get_data()>0).squeeze())
        zmap = zimg.get_data().squeeze()[xyz]
    else:
        xyz = np.where(mask.get_data()>0)
        zmap = zimg.get_data()[xyz]

    xyz = np.array(xyz).T
    nvoxels = np.size(xyz, 0)

    # Thresholding 
    if height_control == 'fpr':
        zth = sp_stats.norm.isf(height_th)
    elif height_control == 'fdr':
        zth = emp_null.FDR(zmap).threshold(height_th)
    elif height_control == 'bonferroni':
        zth = sp_stats.norm.isf(height_th/nvoxels)
    else: ## Brute-force thresholding 
        zth = height_th
    pth = sp_stats.norm.sf(zth)
    above_th = zmap>zth
    if len(np.where(above_th)[0]) == 0:
        return None, None ## FIXME
    zmap_th = zmap[above_th]
    xyz_th = xyz[above_th,:]

    # Clustering
    ## Extract local maxima and connex components above some threshold
    ff = Field(np.size(zmap_th), field=zmap_th)
    ff.from_3d_grid(xyz_th, k=18)
    maxima, depth = ff.get_local_maxima(th=zth)
    labels = ff.cc()
    ## Make list of clusters, each cluster being a dictionary 
    clusters = []
    for k in range(labels.max() + 1):
        s = np.sum(labels == k)
        if s >= cluster_th:
            in_cluster = labels[maxima] == k
            m = maxima[in_cluster]
            d = depth[in_cluster]
            sorted = d.argsort()[::-1]
            clusters.append({'size':s, 'maxima':m[sorted], 'depth':d[sorted]}) 

    ## Sort clusters by descending size order
    def smaller(c1, c2):
        return int(np.sign(c2['size']-c1['size']))
    clusters.sort(cmp=smaller)

    # FDR-corrected p-values
    fdr_pvalue = emp_null.FDR(zmap).all_fdr()[above_th]

    # Default "nulls"
    if not nulls.has_key('zmax'):
        nulls['zmax'] = 'bonferroni'
    if not nulls.has_key('smax'):
        nulls['smax'] = None
    if not nulls.has_key('s'):
        nulls['s'] = None

    # Report significance levels in each cluster 
    for c in clusters:
        maxima = c['maxima']
        zscore = zmap_th[maxima]
        pval = sp_stats.norm.sf(zscore)
        # Replace array indices with real coordinates
        c['maxima'] = apply_affine(zimg.get_affine(), xyz_th[maxima]) 
        c['zscore'] = zscore
        c['pvalue'] = pval
        c['fdr_pvalue'] = fdr_pvalue[maxima]

        # Voxel-level corrected p-values
        p = None
        if nulls['zmax'] == 'bonferroni':
            p = bonferroni(pval, nvoxels) 
        elif isinstance(nulls['zmax'], np.ndarray):
            p = simulated_pvalue(zscore, nulls['zmax'])
        c['fwer_pvalue'] = p

        # Cluster-level p-values (corrected)
        p = None
        if isinstance(nulls['smax'], np.ndarray):
            p = simulated_pvalue(c['size'], nulls['smax'])
        c['cluster_fwer_pvalue'] = p

        # Cluster-level p-values (uncorrected)
        p = None
        if isinstance(nulls['s'], np.ndarray):
            p = simulated_pvalue(c['size'], nulls['s'])
        c['cluster_pvalue'] = p

    # General info
    info = {'nvoxels': nvoxels,
            'threshold_z': zth,
            'threshold_p': pth,
            'threshold_pcorr': bonferroni(pth, nvoxels)}

    return clusters, info 

################################################################################
# Peak_extraction
################################################################################


def get_3d_peaks(image, mask=None, threshold=0., nn=18, order_th=0):
    """
    returns all the peaks of image that are with the mask
    and above the provided threshold

    Parameters
    ----------
    image, (3d) test image
    mask=None, (3d) mask image
        By default no masking is performed
    threshold=0., float, threshold value above which peaks are considered
    nn=18, int, number of neighbours of the topological spatial model
    order_th=0, int, threshold on topological order to validate the peaks

    Returns
    -------
    peaks, a list of dictionray, where each dic has the fields:
    vals, map value at the peak
    order, topological order of the peak
    ijk, array of shape (1,3) grid coordinate of the peak
    pos, array of shape (n_maxima,3) mm coordinates (mapped by affine)
        of the peaks
    """
    # Masking
    if mask!=None:
        bmask = mask.get_data().ravel()
        data = image.get_data().ravel()[bmask>0]
        xyz = np.array(np.where(bmask>0)).T
    else:
        shape = image.get_shape()
        data = image.get_data().ravel()
        xyz = np.reshape(np.indices(shape),(3,np.prod(shape))).T
    affine = image.get_affine()

    if not (data>threshold).any():
        return None

    # Extract local maxima and connex components above some threshold
    ff = Field(np.size(data), field=data)
    ff.from_3d_grid(xyz, k=18)
    maxima, order = ff.get_local_maxima(th=threshold)

    # retain only the maxima greater than the specified order
    maxima = maxima[order>order_th]
    order = order[order>order_th]

    n_maxima = len(maxima)
    if n_maxima==0:
        # should not occur ?
        return None
    
    # reorder the maxima to have decreasing peak value
    vals = data[maxima]
    idx = np.argsort(-vals)
    maxima = maxima[idx]
    order = order[idx]
    
    vals = data[maxima]
    ijk = xyz[maxima]
    pos = np.dot(np.hstack((ijk,np.ones((n_maxima,1)))),affine.T)[:,:3]
    
    peaks = [{'val':vals[k], 'order':order[k], 'ijk':ijk[k], 'pos':pos[k]}
             for k in range(n_maxima)]

    
    
    return peaks



################################################################################
# Statistical tests
################################################################################


def mask_intersection(masks):
    """
    Compute mask intersection
    fixme : dirty and already implemented in mask module: should be removed
    """
    mask = masks[0].copy()
    for m in masks[1:]:
        mask = mask * m
    mask[mask != 0] = 1
    return mask 


def prepare_arrays(data_images, vardata_images, mask_images):
    # Compute mask intersection
    mask = mask_intersection([mask.get_data() for mask in mask_images])
    # Compute xyz coordinates from mask 
    xyz = np.array(np.where(mask>0))
    # Prepare data & vardata arrays 
    data = np.array([(d.get_data()[xyz[0], xyz[1], xyz[2]]).squeeze()
                    for d in data_images]).squeeze()
    if vardata_images == None: 
        vardata = None
    else: 
        vardata = np.array([(d.get_data()[xyz[0], xyz[1], xyz[2]]).squeeze()
                            for d in vardata_images]).squeeze()
    return data, vardata, xyz, mask 


def onesample_test(data_images, vardata_images, mask_images, stat_id, 
                   permutations=0, cluster_forming_th=0.01):
    """
    Helper function for permutation-based mass univariate onesample 
    group analysis. 
    """

    # Prepare arrays
    data, vardata, xyz, mask = prepare_arrays(data_images, vardata_images, 
                                              mask_images)

    # Create one-sample permutation test instance
    ptest = permutation_test_onesample(data, xyz, vardata=vardata, 
                                       stat_id=stat_id)

    # Compute z-map image 
    zmap = np.zeros(data_images[0].get_shape()).squeeze()
    zmap[list(xyz)] = ptest.zscore()
    zimg = Image(zmap, data_images[0].get_affine())

    # Compute mask image
    maskimg = Image(mask, data_images[0].get_affine())
    
    # Multiple comparisons
    if permutations <= 0: 
        return zimg, maskimg
    else: 
        # Cluster definition: (threshold, diameter)
        cluster_def = (ptest.height_threshold(cluster_forming_th), None)
  
        # Calibration 
        voxel_res, cluster_res, region_res = \
            ptest.calibrate(nperms=permutations, clusters=[cluster_def])
        nulls = {}
        nulls['zmax'] = ptest.zscore(voxel_res['perm_maxT_values'])
        nulls['s'] = cluster_res[0]['perm_size_values']
        nulls['smax'] = cluster_res[0]['perm_maxsize_values']
        
        # Return z-map image, mask image and dictionary of null distribution 
        # for cluster sizes (s), max cluster size (smax) and max z-score (zmax)
        return zimg, maskimg, nulls


def twosample_test(data_images, vardata_images, mask_images, labels, stat_id, 
                   permutations=0, cluster_forming_th=0.01):
    """
    Helper function for permutation-based mass univariate twosample group 
    analysis. Labels is a binary vector (1-2). Regions more active for group 
    1 than group 2 are inferred.
    """

    # Prepare arrays
    data, vardata, xyz, mask = prepare_arrays(data_images, vardata_images, 
                                              mask_images)

    # Create two-sample permutation test instance
    if vardata_images == None:
        ptest = permutation_test_twosample(data[labels==1], data[labels==2], 
        xyz, stat_id=stat_id)
    else:
        ptest = permutation_test_twosample(data[labels==1], data[labels==2], 
                                           xyz, vardata1=vardata[labels==1], 
                                           vardata2=vardata[labels==2], 
                                           stat_id=stat_id)
    
    # Compute z-map image 
    zmap = np.zeros(data_images[0].get_shape()).squeeze()
    zmap[list(xyz)] = ptest.zscore()
    zimg = Image(zmap, data_images[0].get_affine())

    # Compute mask image
    maskimg = Image(mask, data_images[0].get_affine())
    
    # Multiple comparisons
    if permutations <= 0: 
        return zimg, maskimg
    else: 
        # Cluster definition: (threshold, diameter)
        cluster_def = (ptest.height_threshold(cluster_forming_th), None)
  
        # Calibration 
        voxel_res, cluster_res, region_res = \
            ptest.calibrate(nperms=permutations, clusters=[cluster_def])
        nulls = {}
        nulls['zmax'] = ptest.zscore(voxel_res['perm_maxT_values'])
        nulls['s'] = cluster_res[0]['perm_size_values']
        nulls['smax'] = cluster_res[0]['perm_maxsize_values']
        
        # Return z-map image, mask image and dictionary of null 
        # distribution for cluster sizes (s), max cluster size (smax) 
        # and max z-score (zmax)
        return zimg, maskimg, nulls

################################################################################
# Linear model
################################################################################

def linear_model_fit(data_images, mask_images, design_matrix, vector):
    """
    Helper function for group data analysis using arbitrary design matrix
    """
    
    # Prepare arrays
    data, vardata, xyz, mask = prepare_arrays(data_images, None, mask_images)
    
    # Create glm instance
    G = glm(data, design_matrix)
    
    # Compute requested contrast
    c = G.contrast(vector)
    
    # Compute z-map image 
    zmap = np.zeros(data_images[0].get_shape()).squeeze()
    zmap[list(xyz)] = c.zscore()
    zimg = Image(zmap, data_images[0].get_affine())
    
    return zimg


class LinearModel(object): 

    def_model = 'spherical'
    def_niter = 2

    def __init__(self, data, design_matrix, mask=None, formula=None, 
                 model=def_model, method=None, niter=def_niter):

        # Convert input data and design into sequences
        if not hasattr(data, '__iter__'): 
            data = [data]
        if not hasattr(design_matrix, '__iter__'): 
            design_matrix = [design_matrix]
            
        # configure spatial properties
        # the 'sampling' direction is assumed to be the last
        # TODO: check that all input images have the same shape and
        # that it's consistent with the mask
        nomask = mask == None
        if nomask: 
            self.xyz = None
            self.axis = len(data[0].get_shape())-1
        else: 
            self.xyz = np.where(mask.get_data()>0)
            self.axis = 1
        
        self.spatial_shape = data[0].get_shape()[0:-1]            
        self.affine = data[0].get_affine()

        self.glm = []
        for i in range(len(data)):
            if not isinstance(design_matrix[i], np.ndarray):
                raise ValueError('Invalid design matrix')
            if nomask: 
                Y = data[i].get_data()
            else: 
                Y = data[i].get_data()[self.xyz]
            X = design_matrix[i]
                
            self.glm.append(glm(Y, X, axis=self.axis, 
                                formula=formula, model=model, 
                                method=method, niter=niter))
                

    def dump(self, filename):
        """
        Dump GLM fit as NPZ file.  
        """
        models = len(self.glm) 
        if models==1: 
            self.glm[0].save(filename)
        else: 
            for i in range(models):
                self.glm[i].save(filename+str(i))

    def contrast(self, vector):
        """
        Compute images of contrast and contrast variance.  
        """

        # Compute the overall contrast across models
        c = self.glm[0].contrast(vector)
        for g in self.glm[1:]: 
            c += g.contrast(vector)


        def affect_inmask(dest, src, xyz):
            if xyz == None:
                dest = src
            else:
                dest[xyz] = src
            return dest

        con = np.zeros(self.spatial_shape)
        con_img = Image(affect_inmask(con, c.effect, self.xyz), self.affine)
        vcon = np.zeros(self.spatial_shape)
        vcon_img = Image(affect_inmask(vcon, c.variance, self.xyz), self.affine)
        z = np.zeros(self.spatial_shape)
        z_img = Image(affect_inmask(z, c.zscore(), self.xyz), self.affine)
        
        dof = c.dof
        
        return con_img, vcon_img, z_img, dof




################################################################################
# Hack to have nose skip onesample_test, which is not a unit test
onesample_test.__test__ = False
twosample_test.__test__ = False

