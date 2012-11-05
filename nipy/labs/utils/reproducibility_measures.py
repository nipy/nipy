# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
These are several functions for computing reproducibility measures.
A use script should be appended soon on the repository.

In general thuis proceeds as follows:
The dataset is subject to jacknife subampling ('splitting'),
each subsample being analysed independently.
A reproducibility measure is then derived;

All is used to produce the work described in
Analysis of a large fMRI cohort:
Statistical and methodological issues for group analyses.
Thirion B, Pinel P, Meriaux S, Roche A, Dehaene S, Poline JB.
Neuroimage. 2007 Mar;35(1):105-20.

Bertrand Thirion, 2009-2010
"""

import numpy as np
from nipy.labs.spatial_models.discrete_domain import \
    grid_domain_from_binary_array

# ---------------------------------------------------------
# ----- cluster handling functions ------------------------
# ---------------------------------------------------------


def histo_repro(h):
    """ Given the histogram h, compute a standardized reproducibility measure

    Parameters
    ----------
    h array of shape(xmax+1), the histogram values

    Returns
    -------
    hr, float: the measure
    """
    k = np.size(h) - 1
    if k == 1:
        return 0.
    nf = np.dot(h, np.arange(k + 1)) / k
    if nf == 0:
        return 0.
    n1k = np.arange(1, k + 1)
    res = 1.0 * np.dot(h[1:], n1k * (n1k - 1)) / (k * (k - 1))
    return res / nf


def cluster_threshold(stat_map, domain, th, csize):
    """Perform a thresholding of a map at the cluster-level

    Parameters
    ----------
    stat_map: array of shape(nbvox)
         the input data
    domain: Nifti1Image instance,
          referential- and domain-defining image
    th (float): cluster-forming threshold
    cisze (int>0): cluster size threshold

    Returns
    -------
    binary array of shape (nvox): the binarized thresholded map

    Notes
    -----
    Should be replaced by a more standard function in the future
    """
    if stat_map.shape[0] != domain.size:
        raise ValueError('incompatible dimensions')

    # first build a domain of supra_threshold regions
    thresholded_domain = domain.mask(stat_map > th)

    # get the connected components
    label = thresholded_domain.connected_components()

    binary = - np.ones(domain.size)
    binary[stat_map > th] = label
    nbcc = len(np.unique(label))
    for i in range(nbcc):
        if np.sum(label == i) < csize:
            binary[binary == i] = - 1

    binary = (binary > -1)
    return binary


def get_cluster_position_from_thresholded_map(stat_map, domain, thr=3.0,
                                              csize=10):
    """
    the clusters above thr of size greater than csize in
    18-connectivity are computed

    Parameters
    ----------
    stat_map : array of shape (nbvox),
           map to threshold
    mask: Nifti1Image instance,
          referential- and domain-defining image
    thr: float, optional,
         cluster-forming threshold
    cisze=10: int
              cluster size threshold

    Returns
    -------
    positions array of shape(k,anat_dim):
              the cluster positions in physical coordinates
              where k= number of clusters
              if no such cluster exists, None is returned
    """
    # if no supra-threshold voxel, return
    if (stat_map <= thr).all():
        return None

    # first build a domain of supra_threshold regions
    thresholded_domain = domain.mask(stat_map > thr)

    # get the connected components
    label = thresholded_domain.connected_components()

    # get the coordinates
    coord = thresholded_domain.get_coord()

    # get the barycenters
    baryc = []
    for i in range(label.max() + 1):
        if np.sum(label == i) >= csize:
            baryc.append(np.mean(coord[label == i], 0))

    if len(baryc) == 0:
        return None

    baryc = np.vstack(baryc)
    return baryc


def get_peak_position_from_thresholded_map(stat_map, domain, threshold):
    """The peaks above thr in 18-connectivity are computed

    Parameters
    ----------
    stat_map: array of shape (nbvox): map to threshold
    deomain: referential- and domain-defining image
    thr, float: cluster-forming threshold

    Returns
    -------
    positions array of shape(k,anat_dim):
              the cluster positions in physical coordinates
              where k= number of clusters
              if no such cluster exists, None is returned
    """
    from ..statistical_mapping import get_3d_peaks

    # create an image to represent stat_map
    simage = domain.to_image(data=stat_map)
    
    # extract the peaks
    peaks = get_3d_peaks(simage, threshold=threshold, order_th=2)
    if peaks == None:
        return None

    pos = np.array([p['pos'] for p in peaks])
    return pos


# ---------------------------------------------------------
# ----- data splitting functions ------------------------
# ---------------------------------------------------------


def bootstrap_group(nsubj, ngroups):
    """Split the proposed group into redundant subgroups by bootstrap

    Parameters
    ----------
    nsubj (int) the number of subjects in the population
    ngroups(int) Number of subbgroups to be drawn

    Returns
    -------
    samples: a list of ngroups arrays containing
             the indexes of the subjects in each subgroup
    """
    groupsize = nsubj
    samples = [(groupsize * np.random.rand(groupsize)).astype(np.int)
               for i in range(ngroups)]
    return samples


def split_group(nsubj, ngroups):
    """Split the proposed group into random disjoint subgroups

    Parameters
    ----------
    nsubj (int) the number of subjects to be split
    ngroups(int) Number of subbgroups to be drawn

    Returns
    -------
    samples: a list of ngroups arrays containing
             the indexes of the subjects in each subgroup
    """
    groupsize = int(np.floor(nsubj / ngroups))
    rperm = np.argsort(np.random.rand(nsubj))
    samples = [rperm[i * groupsize: (i + 1) * groupsize]
              for i in range(ngroups)]
    return samples


# ---------------------------------------------------------
# ----- statistic computation -----------------------------
# ---------------------------------------------------------


def conjunction(x, vx, k):
    """Returns a conjunction statistic as the sum of the k lowest t-values

    Parameters
    ----------
    x: array of shape(nrows, ncols),
       effect matrix
    vx: array of shape(nrows, ncols),
        variance matrix
    k: int,
       number of subjects in the conjunction

    Returns
    -------
    t array of shape(nrows): conjunction statistic
    """
    t = np.sort(x / np.sqrt(np.maximum(vx, 1.e-15)))
    cjt = np.sum(t[:, :k], 1)
    return cjt


def ttest(x):
    """Returns the t-test for each row of the data x
    """
    from ..group.onesample import stat
    t = stat(x.T, id='student', axis=0)
    return np.squeeze(t)


def fttest(x, vx):
    """Assuming that x and vx represent a effect and variance estimates,
    returns a cumulated ('fixed effects') t-test of the data over each row

    Parameters
    ----------
    x: array of shape(nrows, ncols): effect matrix
    vx: array of shape(nrows, ncols): variance matrix

    Returns
    -------
    t array of shape(nrows): fixed effect statistics array
    """
    if np.shape(x) != np.shape(vx):
        raise ValueError("incompatible dimensions for x and vx")
    n = x.shape[1]
    t = x / np.sqrt(np.maximum(vx, 1.e-15))
    t = t.mean(1) * np.sqrt(n)
    return t


def mfx_ttest(x, vx):
    """Idem fttest, but returns a mixed-effects statistic

    Parameters
    ----------
    x: array of shape(nrows, ncols): effect matrix
    vx: array of shape(nrows, ncols): variance matrix

    Returns
    -------
    t array of shape(nrows): mixed effect statistics array
    """
    from ..group.onesample import stat_mfx
    t = stat_mfx(x.T, vx.T, id='student_mfx', axis=0)
    return np.squeeze(t)


def voxel_thresholded_ttest(x, threshold):
    """Returns a binary map of the ttest>threshold
    """
    t = ttest(x)
    return t > threshold


def statistics_from_position(target, data, sigma=1.0):
    """ Return a number characterizing how close data is from
    target using a kernel-based statistic

    Parameters
    ----------
    target: array of shape(nt,anat_dim) or None
            the target positions
    data: array of shape(nd,anat_dim) or None
          the data position
    sigma=1.0 (float), kernel parameter
              or  a distance that say how good good is

    Returns
    -------
    sensitivity (float): how well the targets are fitted
                by the data  in [0,1] interval
                1 is good
                0 is bad
    """
    from ...algorithms.utils.fast_distance import euclidean_distance as ed
    if data == None:
        if target == None:
            return 0.# could be 1.0 ?
        else:
            return 0.
    if target == None:
        return 0.

    dmatrix = ed(data, target) / sigma
    sensitivity = dmatrix.min(0)
    sensitivity = np.exp( - 0.5 * sensitivity ** 2)
    sensitivity = np.mean(sensitivity)
    return sensitivity


# -------------------------------------------------------
# ---------- The main functions -----------------------------
# -------------------------------------------------------


def voxel_reproducibility(data, vardata, domain, ngroups, method='crfx',
                          swap=False, verbose=0, **kwargs):
    """ return a measure of voxel-level reproducibility of activation patterns

    Parameters
    ----------
    data: array of shape (nvox,nsubj)
          the input data from which everything is computed
    vardata: array of shape (nvox,nsubj)
             the corresponding variance information
             ngroups (int):
             Number of subbgroups to be drawn
    domain: referential- and domain-defining image
    ngourps: int,
             number of groups to be used in the resampling procedure
    method: string, to be chosen among 'crfx', 'cmfx', 'cffx'
            inference method under study
    verbose: bool, verbosity mode

    Returns
    -------
    kappa (float): the desired  reproducibility index
    """
    rmap = map_reproducibility(data, vardata, domain, ngroups, method,
                                     swap, verbose, **kwargs)

    h = np.array([np.sum(rmap == i) for i in range(ngroups + 1)])
    hr = histo_repro(h)
    return hr


def draw_samples(nsubj, ngroups, split_method='default'):
    """ Draw randomly ngroups sets of samples from [0..nsubj-1]

    Parameters
    ----------
    nsubj, int, the total number of items
    ngroups, int, the number of desired groups
    split_method: string, optional,
                  to be chosen among 'default', 'bootstrap', 'jacknife'
                  if 'bootstrap', then each group will be nsubj
                     drawn with repetitions among nsubj
                  if 'jacknife' the population is divided into
                      ngroups disjoint equally-sized subgroups
                  if 'default', 'bootstrap' is used when nsubj < 10 * ngroups
                     otherwise jacknife is used

    Returns
    -------
    samples, a list of ngroups array that represent the subsets.

    fixme : this should allow variable bootstrap,
    i.e. draw ngroups of groupsize among nsubj
    """
    if split_method == 'default':
        if nsubj > 10 * ngroups:
            samples = split_group(nsubj, ngroups)
        else:
            samples = bootstrap_group(nsubj, ngroups)
    elif split_method == 'bootstrap':
        samples = bootstrap_group(nsubj, ngroups)
    elif split_method == '':
        samples = split_group(nsubj, ngroups)
    else:
        raise ValueError('unknown splitting method')

    return samples


def map_reproducibility(data, vardata, domain, ngroups, method='crfx',
                        swap=False, verbose=0, **kwargs):
    """ Return a reproducibility map for the given method

    Parameters
    ----------
    data: array of shape (nvox,nsubj)
          the input data from which everything is computed
    vardata: array of the same size
             the corresponding variance information
    domain: referential- and domain-defining image
    ngroups (int): the size of each subrgoup to be studied
    threshold (float): binarization threshold
              (makes sense only if method==rfx)
    method='crfx', string to be chosen among 'crfx', 'cmfx', 'cffx'
           inference method under study
    verbose=0 : verbosity mode

    Returns
    -------
    rmap: array of shape(nvox)
          the reproducibility map
    """
    nsubj = data.shape[1]
    nvox = data.shape[0]
    samples = draw_samples(nsubj, ngroups)
    rmap = np.zeros(nvox)

    for i in range(ngroups):
        x = data[:, samples[i]]

        if swap:
            # randomly swap the sign of x
            x *= (2 * (np.random.rand(len(samples[i])) > 0.5) - 1)

        if method is not 'crfx':
            vx = vardata[:, samples[i]]
        csize = kwargs['csize']
        threshold = kwargs['threshold']

        # compute the statistical maps according to the method you like
        if method == 'crfx':
            stat_map = ttest(x)
        elif method == 'cffx':
            stat_map = fttest(x, vx)
        elif method == 'cmfx':
            stat_map = mfx_ttest(x, vx)
        elif method == 'cjt':
            # if kwargs.has_key('k'):
            if 'k' in kwargs:
                k = kwargs['k']
            else:
                k = nsubj / 2
            stat_map = conjunction(x, vx, k)
        else:
            raise ValueError('unknown method')

        # add the binarized map to a reproducibility map
        rmap += cluster_threshold(stat_map, domain, threshold, csize) > 0

    return rmap


def peak_reproducibility(data, vardata, domain, ngroups, sigma, method='crfx',
                         swap=False, verbose=0, **kwargs):
    """ Return a measure of cluster-level reproducibility
    of activation patterns
    (i.e. how far clusters are from each other)

    Parameters
    ----------
    data: array of shape (nvox,nsubj)
          the input data from which everything is computed
    vardata: array of shape (nvox,nsubj)
             the variance of the data that is also available
    domain: refenrtial- and domain-defining image
    ngroups (int),
             Number of subbgroups to be drawn
    sigma: float, parameter that encodes how far far is
    threshold: float, binarization threshold
    method: string to be chosen among 'crfx', 'cmfx' or 'cffx',
           inference method under study
    swap = False: if True, a random sign swap of the data is performed
         This is used to simulate a null hypothesis on the data.
    verbose=0 : verbosity mode

    Returns
    -------
    score (float): the desired  cluster-level reproducibility index
    """
    tiny = 1.e-15
    nsubj = data.shape[1]
    samples = draw_samples(nsubj, ngroups)
    all_pos = []

    # compute the positions in the different subgroups
    for i in range(ngroups):
        x = data[:, samples[i]]

        if swap:
            # apply a random sign swap to x
            x *= (2 * (np.random.rand(len(samples[i])) > 0.5) - 1)

        if method is not 'crfx':
            vx = vardata[:, samples[i]]
        if method is not 'bsa':
            threshold = kwargs['threshold']

            if method == 'crfx':
                stat_map = ttest(x)
            elif method == 'cmfx':
                stat_map = mfx_ttest(x, vx)
            elif method == 'cffx':
                stat_map = fttest(x, vx)
            elif method == 'cjt':
                if 'k' in kwargs:
                    k = kwargs['k']
                else:
                    k = nsubj / 2
                stat_map = conjunction(x, vx, k)

            pos = get_peak_position_from_thresholded_map(
                stat_map, domain, threshold)
            all_pos.append(pos)
        else:
            # method='bsa' is a special case
            tx = x / (tiny + np.sqrt(vx))
            afname = kwargs['afname']
            theta = kwargs['theta']
            dmax = kwargs['dmax']
            ths = kwargs['ths']
            thq = kwargs['thq']
            smin = kwargs['smin']
            niter = kwargs['niter']
            afname = afname + '_%02d_%04d.pic' % (niter, i)
            pos = coord_bsa(domain, tx, theta, dmax, ths, thq, smin, afname)
            all_pos.append(pos)

    # derive a kernel-based goodness measure from the pairwise comparison
    # of sets of positions
    score = 0
    for i in range(ngroups):
        for j in range(i):
            score += statistics_from_position(all_pos[i], all_pos[j], sigma)
            score += statistics_from_position(all_pos[j], all_pos[i], sigma)
    score /= (ngroups * (ngroups - 1))
    return score


def cluster_reproducibility(data, vardata, domain, ngroups, sigma,
                            method='crfx', swap=False, verbose=0,
                            **kwargs):
    """Returns a measure of cluster-level reproducibility
    of activation patterns
    (i.e. how far clusters are from each other)

    Parameters
    ----------
    data: array of shape (nvox,nsubj)
          the input data from which everything is computed
    vardata: array of shape (nvox,nsubj)
             the variance of the data that is also available
    domain: referential- and domain- defining image instance
    ngroups (int),
             Number of subbgroups to be drawn
    sigma (float): parameter that encodes how far far is
    threshold (float):
              binarization threshold
    method='crfx', string to be chosen among 'crfx', 'cmfx' or 'cffx'
           inference method under study
    swap = False: if True, a random sign swap of the data is performed
         This is used to simulate a null hypothesis on the data.
    verbose=0 : verbosity mode

    Returns
    -------
    score (float): the desired  cluster-level reproducibility index
    """
    tiny = 1.e-15
    nsubj = data.shape[1]
    samples = draw_samples(nsubj, ngroups)
    all_pos = []

    # compute the positions in the different subgroups
    for i in range(ngroups):
        x = data[:, samples[i]]

        if swap:
            # apply a random sign swap to x
            x *= (2 * (np.random.rand(len(samples[i])) > 0.5) - 1)

        if method is not 'crfx':
            vx = vardata[:, samples[i]]
        if method is not 'bsa':
            csize = kwargs['csize']
            threshold = kwargs['threshold']
            if method == 'crfx':
                stat_map = ttest(x)
            elif method == 'cmfx':
                stat_map = mfx_ttest(x, vx)
            elif method == 'cffx':
                stat_map = fttest(x, vx)
            elif method == 'cjt':
                if  'k' in kwargs:
                    k = kwargs['k']
                else:
                    k = nsubj / 2
                stat_map = conjunction(x, vx, k)
            pos = get_cluster_position_from_thresholded_map(stat_map, domain,
                                                            threshold, csize)
            all_pos.append(pos)
        else:
            # method='bsa' is a special case
            tx = x / (tiny + np.sqrt(vx))
            afname = kwargs['afname']
            theta = kwargs['theta']
            dmax = kwargs['dmax']
            ths = kwargs['ths']
            thq = kwargs['thq']
            smin = kwargs['smin']
            niter = kwargs['niter']
            afname = afname + '_%02d_%04d.pic' % (niter, i)
            pos = coord_bsa(domain, tx, theta, dmax, ths, thq, smin, afname)
        all_pos.append(pos)

    # derive a kernel-based goodness measure from the pairwise comparison
    # of sets of positions
    score = 0
    for i in range(ngroups):
        for j in range(i):
            score += statistics_from_position(all_pos[i], all_pos[j], sigma)
            score += statistics_from_position(all_pos[j], all_pos[i], sigma)

    score /= (ngroups * (ngroups - 1))
    return score


def group_reproducibility_metrics(
    mask_images, contrast_images, variance_images, thresholds, ngroups,
    method, cluster_threshold=10, number_of_samples=10, sigma=6.,
    do_clusters=True, do_voxels=True, do_peaks=True, swap=False):
    """
    Main function to perform reproducibility analysis, including nifti1 io

    Parameters
    ----------
    threshold: list or 1-d array,
               the thresholds to be tested

    Returns
    -------
    cluster_rep_results: dictionary,
                         results of cluster-level reproducibility analysi
    voxel_rep_results: dictionary,
                       results of voxel-level reproducibility analysis
    peak_rep_results: dictionary,
                      results of peak-level reproducibility analysis
    """
    from nibabel import load
    from ..mask import intersect_masks

    if ((len(variance_images) == 0) & (method is not 'crfx')):
        raise ValueError('Variance images are necessary')

    nsubj = len(contrast_images)

    # compute the group mask
    affine = load(mask_images[0]).get_affine()
    mask = intersect_masks(mask_images, threshold=0) > 0
    domain = grid_domain_from_binary_array(mask, affine)

    # read the data
    group_con = []
    group_var = []
    for s in range(nsubj):
        group_con.append(load(contrast_images[s]).get_data()[mask])
        if len(variance_images) > 0:
            group_var.append(load(variance_images[s]).get_data()[mask])

    group_con = np.squeeze(np.array(group_con)).T
    group_con[np.isnan(group_con)] = 0
    if len(variance_images) > 0:
        group_var = np.squeeze(np.array(group_var)).T
        group_var[np.isnan(group_var)] = 0
        group_var = np.maximum(group_var, 1.e-15)

    # perform the analysis
    voxel_rep_results = {}
    cluster_rep_results = {}
    peak_rep_results = {}

    for ng in ngroups:
        if do_voxels:
            voxel_rep_results.update({ng: {}})
        if do_clusters:
            cluster_rep_results.update({ng: {}})
        if do_peaks:
            peak_rep_results.update({ng: {}})
        for th in thresholds:
            kappa = []
            cls = []
            pk = []
            kwargs = {'threshold': th, 'csize': cluster_threshold}

            for i in range(number_of_samples):
                if do_voxels:
                    kappa.append(voxel_reproducibility(
                            group_con, group_var, domain, ng, method, swap,
                            **kwargs))
                if do_clusters:
                    cls.append(cluster_reproducibility(
                            group_con, group_var, domain, ng, sigma, method,
                            swap, **kwargs))
                if do_peaks:
                    pk.append(peak_reproducibility(
                            group_con, group_var, domain, ng, sigma, method,
                            swap, **kwargs))

            if do_voxels:
                voxel_rep_results[ng].update({th: np.array(kappa)})
            if do_clusters:
                cluster_rep_results[ng].update({th: np.array(cls)})
            if do_peaks:
                peak_rep_results[ng].update({th: np.array(cls)})

    return voxel_rep_results, cluster_rep_results, peak_rep_results


# -------------------------------------------------------
# ---------- BSA stuff ----------------------------------
# -------------------------------------------------------


def coord_bsa(domain, betas, theta=3., dmax=5., ths=0, thq=0.5, smin=0,
              afname=None):
    """ main function for  performing bsa on a dataset
    where bsa =  nipy.labs.spatial_models.bayesian_structural_analysis

    Parameters
    ----------
    domain: image instance,
          referential- and domain-defining image
    betas: array of shape (nbnodes, subjects),
           the multi-subject statistical maps
    theta: float, optional
           first level threshold
    dmax: float>0, optional
          expected cluster std in the common space in units of coord
    ths: int, >=0), optional
         representatitivity threshold
    thq: float, optional,
         posterior significance threshold should be in [0,1]
    smin: int, optional,
          minimal size of the regions to validate them
    afname: string, optional
            path where intermediate resullts cam be pickelized

    Returns
    -------
    afcoord array of shape(number_of_regions,3):
            coordinate of the found landmark regions
    """
    from ..spatial_models.bayesian_structural_analysis import compute_BSA_quick

    crmap, AF, BF, p = compute_BSA_quick(
        domain, betas, dmax, thq, smin, ths, theta, verbose=0)
    if AF == None:
        return None
    if afname is not None:
        import pickle
        pickle.dump(AF, afname)
    afcoord = AF.discrete_to_roi_features('position')
    return afcoord
