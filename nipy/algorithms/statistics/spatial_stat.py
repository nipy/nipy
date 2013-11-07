# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np
import scipy.stats as sp_stats
from nibabel.affines import apply_affine

from ..graph.field import field_from_graph_and_data
from ..graph.graph import wgraph_from_3d_grid
from .empirical_pvalue import (gaussian_fdr,
                               gaussian_fdr_threshold)


###############################################################################
# Cluster statistics
###############################################################################


def bonferroni(p, n):
    return np.minimum(1., p * n)


def simulated_pvalue(t, simu_t):
    return 1 - np.searchsorted(simu_t, t) / float(np.size(simu_t))


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

    Notes
    -----
    This works only with three dimensional data
    """
    # Masking
    if len(mask.shape) > 3:
        xyz = np.where((mask.get_data() > 0).squeeze())
        zmap = zimg.get_data().squeeze()[xyz]
    else:
        xyz = np.where(mask.get_data() > 0)
        zmap = zimg.get_data()[xyz]

    xyz = np.array(xyz).T
    nvoxels = np.size(xyz, 0)

    # Thresholding
    if height_control == 'fpr':
        zth = sp_stats.norm.isf(height_th)
    elif height_control == 'fdr':
        zth = gaussian_fdr_threshold(zmap, height_th)
    elif height_control == 'bonferroni':
        zth = sp_stats.norm.isf(height_th / nvoxels)
    else: ## Brute-force thresholding
        zth = height_th
    pth = sp_stats.norm.sf(zth)
    above_th = zmap > zth
    if len(np.where(above_th)[0]) == 0:
        return None, None ## FIXME
    zmap_th = zmap[above_th]
    xyz_th = xyz[above_th]

    # Clustering
    ## Extract local maxima and connex components above some threshold
    ff = field_from_graph_and_data(wgraph_from_3d_grid(xyz_th, k=18), zmap_th)
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
            clusters.append({'size': s,
                             'maxima': m[sorted],
                             'depth': d[sorted]})

    ## Sort clusters by descending size order
    clusters.sort(key=lambda c : c['size'], reverse=True)

    # FDR-corrected p-values
    fdr_pvalue = gaussian_fdr(zmap)[above_th]

    # Default "nulls"
    if not 'zmax' in nulls:
        nulls['zmax'] = 'bonferroni'
    if not 'smax' in nulls:
        nulls['smax'] = None
    if not 's' in nulls:
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

###############################################################################
# Peak_extraction
###############################################################################


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
    peaks, a list of dictionaries, where each dict has the fields:
    vals, map value at the peak
    order, topological order of the peak
    ijk, array of shape (1,3) grid coordinate of the peak
    pos, array of shape (n_maxima,3) mm coordinates (mapped by affine)
        of the peaks
    """
    # Masking
    if mask is not None:
        bmask = mask.get_data().ravel()
        data = image.get_data().ravel()[bmask > 0]
        xyz = np.array(np.where(bmask > 0)).T
    else:
        shape = image.shape
        data = image.get_data().ravel()
        xyz = np.reshape(np.indices(shape), (3, np.prod(shape))).T
    affine = image.get_affine()

    if not (data > threshold).any():
        return None

    # Extract local maxima and connex components above some threshold
    ff = field_from_graph_and_data(wgraph_from_3d_grid(xyz, k=18), data)
    maxima, order = ff.get_local_maxima(th=threshold)

    # retain only the maxima greater than the specified order
    maxima = maxima[order > order_th]
    order = order[order > order_th]

    n_maxima = len(maxima)
    if n_maxima == 0:
        # should not occur ?
        return None

    # reorder the maxima to have decreasing peak value
    vals = data[maxima]
    idx = np.argsort(- vals)
    maxima = maxima[idx]
    order = order[idx]

    vals = data[maxima]
    ijk = xyz[maxima]
    pos = np.dot(np.hstack((ijk, np.ones((n_maxima, 1)))), affine.T)[:, :3]
    peaks = [{'val': vals[k], 'order': order[k], 'ijk': ijk[k], 'pos': pos[k]}
             for k in range(n_maxima)]

    return peaks
