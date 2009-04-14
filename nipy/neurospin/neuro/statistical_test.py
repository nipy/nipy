import nipy.neurospin as fff2

import nipy.neurospin.graph.field as field
import nipy.neurospin.registration.transform_affine as affine
import nipy.neurospin.group.permutation_test as permutt

from nipy.neurospin.utils import emp_null

import numpy as np
import scipy.stats as sps


################################################################################
# Cluster statistics 
################################################################################

def z_threshold(height_th, height_control):
    if height_control == 'fpr':
        return sps.norm.isf(height_th)
    elif height_control == 'fdr':
        return emp_null.FDR(zmap).threshold(height_th)
    elif height_control == 'bonferroni':
        return sps.norm.isf(height_th/nvoxels)
    else: ## Brute-force thresholding 
        return height_th

def bonferroni(p, n):
    return np.minimum(1., p*n)
    
def simulated_pvalue(t, simu_t): 
    return 1 - np.searchsorted(simu_t, t)/float(np.size(simu_t))


def cluster_stats(zimg, mask, height_th, height_control='fpr', cluster_th=0,
                  null_zmax='bonferroni', null_smax=None, null_s=None):
    """
    clusters =  cluster_stats(zimg, mask, height_th, height_control='fpr', cluster_th=0,
                              null_zmax='bonferroni', null_smax=None, null_s=None)

    Return a list of clusters, each cluster being represented by a
    dictionary. Clusters are sorted by descending size order. Within
    each cluster, local maxima are sorted by descending depth order.

    Input consist of the following: 
      zimg -- z-score image
      mask -- mask image 
      height_th -- cluster forming threshold
      height_control -- false positive control meaning of cluster forming threshold: 'fpr'|'fdr'|'fwer'
      size_th -- cluster size threshold
      null_zmax -- voxel-level familywise error correction method: 'bonferroni'|'rft'|array
      null_smax -- cluster-level familywise error correction method: None|'rft'|array
      null_s -- cluster-level calibration method: None|'rft'|array
    """
    if not isinstance(zimg, fff2.neuro.image) or not isinstance(mask, fff2.neuro.image): 
        raise ValueError, 'Invalid input images.' 
    
    # Masking 
    xyz = np.where(mask.array>0)
    zmap = zimg.array[xyz]
    xyz = np.array(xyz).T
    nvoxels = np.size(xyz, 0)

    # Thresholding 
    zth = z_threshold(height_th, height_control)
    pth = sps.norm.sf(zth)
    above_th = zmap>zth
    if np.where(above_th)[0].size == 0:
        return None ## FIXME
    zmap_th = zmap[above_th]
    xyz_th = xyz[above_th]

    # Clustering
    ## Extract local maxima and connex components above some threshold
    ff = field.Field(np.size(zmap_th), field=zmap_th)
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
        pval = sps.norm.sf(zscore)
        # Replace array indices with real coordinates
        c['maxima'] = affine.transform(xyz_th[maxima].T, zimg.transform).T 
        c['zscore'] = zscore
        c['pvalue'] = pval
        c['fdr_pvalue'] = fdr_pvalue[maxima]

        # Voxel-level corrected p-values
        p = None
        if null_zmax == 'bonferroni':
            p = bonferroni(pval, nvoxels) 
        elif isinstance(null_zmax, np.ndarray):
            p = simulated_pvalue(zscore, null_zmax)
        c['fwer_pvalue'] = p

        # Cluster-level p-values (corrected)
        p = None
        if isinstance(null_smax, np.ndarray):
            p = simulated_pvalue(c['size'], null_smax)
        c['cluster_fwer_pvalue'] = p

        # Cluster-level p-values (uncorrected)
        p = None
        if isinstance(null_s, np.ndarray):
            p = simulated_pvalue(c['size'], null_s)
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
    mask = mask_intersection([mask.array for mask in mask_images])
    
    # Compute xyz coordinates from mask 
    xyz = np.array(np.where(mask>0))
    
    # Prepare data & vardata arrays 
    data = np.array([d.array[xyz[0],xyz[1],xyz[2]] for d in data_images])
    vardata = np.array([d.array[xyz[0],xyz[1],xyz[2]] for d in vardata_images])
    
    return data, vardata, xyz, mask 


def onesample_test(data_images, vardata_images, mask_images, stat_id,
                   comparisons=False, cluster_forming_th=0.01, cluster_th=0):
    """
    zimg, clusters, info = onesample_test(data_images, vardata_images, mask_images, stat_id,
                                          comparisons=False, cluster_forming_th=0.01, cluster_th=0)

    """

    # Prepare arrays
    data, vardata, xyz, mask = prepare_arrays(data_images, 
                                              vardata_images, mask_images)

    # Create one-sample permutation test instance
    ptest = permutt.permutation_test_onesample(data, xyz, 
                                            vardata=vardata, stat_id=stat_id)

    # Compute z-map image 
    zmap = np.zeros(data_images[0].array.shape)
    zmap[xyz[0,:],xyz[1,:],xyz[2,:]] = ptest.zscore()
    zimg = fff2.neuro.image(data_images[0])
    zimg.set_array(zmap)

    # Compute mask image
    maskimg = fff2.neuro.image(data_images[0])
    maskimg.set_array(mask)
    
    # Multiple comparisons
    if not comparisons:
        return zimg, maskimg
    else: 
        # Cluster definition: (threshold, diameter)
        cluster_def = (ptest.height_threshold(cluster_forming_th), None)
        # Calibration 
        voxel_res, cluster_res, region_res = \
            ptest.calibrate(nperms=self.permutations, clusters=[cluster_def])
        null_zmax = ptest.zscore(voxel_res['perm_maxT_values'])
        null_s = cluster_res[0]['perm_size_values']
        null_smax = cluster_res[0]['perm_maxsize_values']
        
        # Return z-map image, list of cluster dictionaries and info dictionary 
        return zimg, maskimg, null_zmax, null_smax, null_s

################################################################################
# Hack to have nose skip onesample_test, which is not a unit test
onesample_test.__test__ = False

#import sys
#if 'nose' in sys.modules:
#    def onesample_test():
#        import nose
#        raise nose.SkipTest('Not a test')
