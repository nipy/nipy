import numpy as np
import scipy.stats as sp_stats

from nipy.neurospin.graph.field import Field
from nipy.neurospin.register.transform import apply_affine
from nipy.neurospin.utils import emp_null
from nipy.neurospin.glm import glm
from nipy.neurospin.group.permutation_test import permutation_test_onesample
# FIXME: rename permutation_test_onesample class so that name starts with upper case
### FIXME LATER
from nipy.neurospin import Image


################################################################################
# Cluster statistics 
################################################################################

def z_threshold(height_th, height_control):
    if height_control == 'fpr':
        return sp_stats.norm.isf(height_th)
    elif height_control == 'fdr':
        return emp_null.FDR(zmap).threshold(height_th)
    elif height_control == 'bonferroni':
        return sp_stats.norm.isf(height_th/nvoxels)
    else: ## Brute-force thresholding 
        return height_th

def bonferroni(p, n):
    return np.minimum(1., p*n)
    
def simulated_pvalue(t, simu_t): 
    return 1 - np.searchsorted(simu_t, t)/float(np.size(simu_t))


def cluster_stats(zimg, mask, height_th, height_control='fpr', cluster_th=0, nulls={}):
    """
    clusters =  cluster_stats(zimg, mask, height_th, height_control='fpr', cluster_th=0,
                              null_zmax='bonferroni', null_smax=None, null_s=None)

    Return a list of clusters, each cluster being represented by a
    dictionary. Clusters are sorted by descending size order. Within
    each cluster, local maxima are sorted by descending depth order.

    Parameters
    ----------
      zimg : z-score image
      mask : mask image 
      height_th : cluster forming threshold
      height_control : false positive control meaning of cluster forming threshold: 'fpr'|'fdr'|'fwer'
      cluster_th : cluster size threshold
      null_zmax : voxel-level familywise error correction method: 'bonferroni'|'rft'|array
      null_smax : cluster-level familywise error correction method: None|'rft'|array
      null_s : cluster-level calibration method: None|'rft'|array
    """
    
    # Masking 
    xyz = np.where(mask.get_data()>0)
    zmap = zimg.get_data()[xyz]
    xyz = np.array(xyz).T
    nvoxels = np.size(xyz, 0)

    # Thresholding 
    zth = z_threshold(height_th, height_control)
    pth = sp_stats.norm.sf(zth)
    above_th = zmap>zth
    if np.where(above_th)[0].size == 0:
        return None ## FIXME
    zmap_th = zmap[above_th]
    xyz_th = xyz[above_th]

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
        return np.sign(c2['size']-c1['size'])
    clusters.sort(cmp=smaller)

    # FDR-corrected p-values
    fdr_pvalue = emp_null.FDR(zmap).all_fdr()[above_th]

    # Report significance levels in each cluster 
    for c in clusters:
        maxima = c['maxima']
        zscore = zmap_th[maxima]
        pval = sp_stats.norm.sf(zscore)
        # Replace array indices with real coordinates
        c['maxima'] = apply_affine(zimg.get_affine(), xyz_th[maxima].T).T 
        c['zscore'] = zscore
        c['pvalue'] = pval
        c['fdr_pvalue'] = fdr_pvalue[maxima]

        # Default "nulls"
        if not nulls.has_key('zmax'):
            nulls['zmax'] = 'bonferroni'
        if not nulls.has_key('smax'):
            nulls['smax'] = None
        if not nulls.has_key('s'):
            nulls['s'] = None

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
# Statistical tests
################################################################################


def mask_intersection(masks):
    """
    Compute mask intersection
    """
    mask = masks[0]
    for m in masks[1:]:
        mask = mask * m
    return mask 


def prepare_arrays(data_images, vardata_images, mask_images):

    # Compute mask intersection
    mask = mask_intersection([mask.get_data() for mask in mask_images])
    
    # Compute xyz coordinates from mask 
    xyz = np.array(np.where(mask>0))
    
    # Prepare data & vardata arrays 
    data = np.array([d.get_data()[xyz[0],xyz[1],xyz[2]] for d in data_images])
    if vardata_images == None: 
        vardata = None
    else: 
        vardata = np.array([d.get_data()[xyz[0],xyz[1],xyz[2]] for d in vardata_images])
    
    return data, vardata, xyz, mask 


def onesample_test(data_images, vardata_images, mask_images, stat_id, 
                   permutations=0, cluster_forming_th=0.01):
    """
    Helper function for permutation-based mass univariate onesample group analysis. 
    """

    # Prepare arrays
    data, vardata, xyz, mask = prepare_arrays(data_images, vardata_images, mask_images)

    # Create one-sample permutation test instance
    ptest = permutation_test_onesample(data, xyz, vardata=vardata, stat_id=stat_id)

    # Compute z-map image 
    zmap = np.zeros(data_images[0].get_shape())
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
        
        # Return z-map image, list of cluster dictionaries and info dictionary 
        return zimg, maskimg, nulls


################################################################################
# Linear model
################################################################################

def affect_inmask(dest, src, xyz):
    if xyz == None:
        dest = src
    else:
        dest[xyz[0,:], xyz[1,:], xyz[2,:]] = src
    return dest



class LinearModel: 

    def __init__(self, data=None, design_matrix=None, mask=None, formula=None, 
                 model='spherical', method=None, niter=2):

        if data == None:
            self.data = None
            self.xyz = None
            self.glm = None

        else:
            if not isinstance(design_matrix, np.ndarray):
                raise ValueError('Invalid design matrix')
            
            self.data = data
            if mask == None:
                self.xyz = None
                Y = data.get_data()
                axis = 3
            else:
                self.xyz = np.where(mask.get_data()>0)
                Y = data.get_data()[self.xyz]
                axis = 1
                
            self.glm = glm(Y, design_matrix, formula=formula, axis=axis, model=model, 
                           method=method, niter=niter)


    def dump(self, filename):
        """
        Dump GLM fit as NPZ file.  
        """
        self.glm.save(filename)


    def contrast(self, vector):
        """
        Compute images of contrast and contrast variance.  
        """
        c = self.glm.contrast(vector)
        
        con = np.zeros(self.data.get_shape()[1:4])
        con_img = Image(affect_inmask(con, c.effect, self.xyz), self.data.get_affine())

        vcon = np.zeros(self.data.get_shape()[1:4])
        vcon_img = Image(affect_inmask(vcon, c.variance, self.xyz), self.data.get_affine())

        dof = c.dof
        
        return con_img, vcon_img, dof













################################################################################
# Hack to have nose skip onesample_test, which is not a unit test
onesample_test.__test__ = False

#import sys
#if 'nose' in sys.modules:
#    def onesample_test():
#        import nose
#        raise nose.SkipTest('Not a test')
