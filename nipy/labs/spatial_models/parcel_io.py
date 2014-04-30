# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Utility functions for mutli-subjectParcellation:
this basically uses nipy io lib to perform IO opermation
in parcel definition processes
"""

import numpy as np
import os.path

from nibabel import load, save, Nifti1Image

from nipy.io.nibcompat import get_header, get_affine
from nipy.algorithms.clustering.utils import kmeans
from .discrete_domain import grid_domain_from_image
from .mroi import SubDomains
from ..mask import intersect_masks

from warnings import warn

warn('Module nipy.labs.spatial_models.parcel_io' + 
     'deprecated, will be removed',
     FutureWarning,
     stacklevel=2)


def mask_parcellation(mask_images, nb_parcel, threshold=0, output_image=None):
    """ Performs the parcellation of a certain mask

    Parameters
    ----------
    mask_images: string or Nifti1Image or list of strings/Nifti1Images,
                 paths of mask image(s) that define(s) the common space.
    nb_parcel: int,
               number of desired parcels
    threshold: float, optional,
               level of intersection of the masks
    output_image: string, optional
                  path of the output image

    Returns
    -------
    wim: Nifti1Imagine instance,  representing the resulting parcellation
    """
    if isinstance(mask_images, basestring):
        mask = mask_images
    elif isinstance(mask_images, Nifti1Image):
        mask = mask_images
    else:
        # mask_images should be a list
        mask_data = intersect_masks(mask_images, threshold=0) > 0
        mask = Nifti1Image(mask_data.astype('u8'),
                           get_affine(load(mask_images[0])))

    domain = grid_domain_from_image(mask)
    cent, labels, J = kmeans(domain.coord, nb_parcel)
    sub_dom = SubDomains(domain, labels)
    # get id (or labels) image
    wim = sub_dom.to_image(fid='id', roi=True)
    return wim


def parcel_input(mask_images, learning_images, ths=.5, fdim=None):
    """Instantiating a Parcel structure from a give set of input

    Parameters
    ----------
    mask_images: string or Nifti1Image or list of strings/Nifti1Images,
                 paths of mask image(s) that define(s) the common space.
    learning_images: (nb_subject-) list of  (nb_feature-) list of strings,
                     paths of feature images used as input to the
                     parcellation procedure
    ths=.5: threshold to select the regions that are common across subjects.
            if ths = .5, thethreshold is half the number of subjects
    fdim: int, optional
          if nb_feature (the dimension of the data) used in subsequent analyses
          if greater than fdim,
          a PCA is perfomed to reduce the information in the data
          Byd efault, no reduction is performed

    Returns
    -------
    domain : discrete_domain.DiscreteDomain instance
        that stores the spatial information on the parcelled domain
    feature: (nb_subect-) list of arrays of shape (domain.size, fdim)
        feature information available to parcellate the data
    """
    nb_subj = len(learning_images)

    # get a group-level mask
    if isinstance(mask_images, basestring):
        mask = mask_images
    elif isinstance(mask_images, Nifti1Image):
        mask = mask_images
    else:
        # mask_images should be a list
        grp_mask = intersect_masks(mask_images, threshold=ths) > 0
        mask = Nifti1Image(grp_mask.astype('u8'),
                           get_affine(load(mask_images[0])))

    # build the domain
    domain = grid_domain_from_image(mask, nn=6)
    #nn = 6 for speed up and stability

    # load the functional data
    feature = []
    nbeta = len(learning_images[0])
    for s in range(nb_subj):
        if len(learning_images[s]) != nbeta:
            raise ValueError('Inconsistent number of dimensions')
        feature.append(np.array([domain.make_feature_from_image(b)
                                 for b in learning_images[s]]).T)

    # Possibly reduce the dimension of the functional data
    if (len(feature[0].shape) == 1) or (fdim is None):
        return domain, feature
    if fdim < feature[0].shape[1]:
        import numpy.linalg as nl
        subj = np.concatenate([s * np.ones(feature[s].shape[0]) \
                                   for s in range(nb_subj)])
        cfeature = np.concatenate(feature)
        cfeature -= np.mean(cfeature, 0)
        m1, m2, m3 = nl.svd(cfeature, 0)
        cfeature = np.dot(m1, np.diag(m2))
        cfeature = cfeature[:, 0:fdim]
        feature = [cfeature[subj == s] for s in range(nb_subj)]

    return domain, feature


def write_parcellation_images(Pa, template_path=None, indiv_path=None,
                              subject_id=None, swd=None):
    """ Write images that describe the spatial structure of the parcellation

    Parameters
    ----------
    Pa : MultiSubjectParcellation instance,
         the description of the parcellation
    template_path: string, optional,
                   path of the group-level parcellation image
    indiv_path: list of strings, optional
                paths of the individual parcellation images
    subject_id: list of strings of length Pa.nb_subj
                subject identifiers, used to infer the paths when not available
    swd: string, optional
         output directory used to infer the paths when these are not available
    """
    # argument check
    if swd == None:
        from tempfile import mkdtemp
        swd = mkdtemp()

    if subject_id == None:
        subject_id = ['subj_%04d' % s for s in range(Pa.nb_subj)]

    if len(subject_id) != Pa.nb_subj:
        raise ValueError('subject_id does not match parcellation')

    # If necessary, generate the paths
    if template_path is None:
        template_path = os.path.join(swd, "template_parcel.nii")
    if indiv_path is None:
        indiv_path = [os.path.join(swd, "parcel%s.nii" % subject_id[s])
                        for s in range(Pa.nb_subj)]

    # write the template image
    tlabs = Pa.template_labels.astype(np.int16)
    template = SubDomains(Pa.domain, tlabs)
    template_img = template.to_image(
        fid='id', roi=True, descrip='Intra-subject parcellation template')
    save(template_img, template_path)

    # write subject-related stuff
    for s in range(Pa.nb_subj):
        # write the individual label images
        labs = Pa.individual_labels[:, s]
        parcellation = SubDomains(Pa.domain, labs)
        parcellation_img = parcellation.to_image(
            fid='id', roi=True, descrip='Intra-subject parcellation')
        save(parcellation_img, indiv_path[s])


def parcellation_based_analysis(Pa, test_images, test_id='one_sample',
                                rfx_path=None, condition_id='', swd=None):
    """ This function computes parcel averages and RFX at the parcel-level

    Parameters
    ----------
    Pa: MultiSubjectParcellation instance
        the description of the parcellation
    test_images: (Pa.nb_subj-) list of paths
                 paths of images used in the inference procedure
    test_id: string, optional,
          if test_id=='one_sample', the one_sample statstic is computed
          otherwise, the parcel-based signal averages are returned
    rfx_path: string optional,
              path of the resulting one-sample test image, if applicable
    swd: string, optional
         output directory used to compute output path if rfx_path is not given
    condition_id: string, optional,
                  contrast/condition id  used to compute output path

    Returns
    -------
    test_data: array of shape(Pa.nb_parcel, Pa.nb_subj)
               the parcel-level signal average if test is not 'one_sample'
    prfx: array of shape(Pa.nb_parcel),
          the one-sample t-value if test_id is 'one_sample'
    """
    nb_subj = Pa.nb_subj

    # 1. read the test data
    if len(test_images) != nb_subj:
        raise ValueError('Inconsistent number of test images')

    test = np.array([Pa.domain.make_feature_from_image(ti)
                     for ti in test_images]).T
    test_data = Pa.make_feature('', np.array(test))

    if test_id is not 'one_sample':
        return test_data

    # 2. perform one-sample test
    # computation
    from ..utils.reproducibility_measures import ttest
    prfx = ttest(test_data)

    # Write the stuff
    template = SubDomains(Pa.domain, Pa.template_labels)
    template.set_roi_feature('prfx', prfx)
    wim = template.to_image('prfx', roi=True)
    hdr = get_header(wim)
    hdr['descrip'] = 'parcel-based random effects image (in t-variate)'
    if rfx_path is not None:
        save(wim, rfx_path)

    return prfx


def fixed_parcellation(mask_image, betas, nbparcel, nn=6, method='ward',
                          write_dir=None, mu=10., verbose=0, fullpath=None):
    """ Fixed parcellation of a given dataset

    Parameters
    ----------
    domain/mask_image
    betas: list of paths to activation images from the subject
    nbparcel, int : number fo desired parcels
    nn=6: number of nearest neighbors  to define the image topology
          (6, 18 or 26)
    method='ward': clustering method used, to be chosen among
                   'ward', 'gkm', 'ward_and-gkm'
                   'ward': Ward's clustering algorithm
                   'gkm': Geodesic k-means algorithm, random initialization
                   'gkm_and_ward': idem, initialized by Ward's clustering
    write_di: string, topional, write directory.
                    If fullpath is None too, then no file output.
    mu = 10., float: the relative weight of anatomical information
    verbose=0: verbosity mode
    fullpath=None, string,
                   path of the output image
                   If write_dir and fullpath are None then no file output.
                   If only fullpath is None then it is the write dir + a name
                   depending on the method.

    Notes
    -----
    Ward's method takes time (about 6 minutes for a 60K voxels dataset)

    Geodesic k-means is 'quick and dirty'

    Ward's + GKM is expensive but quite good

    To reduce CPU time, rather use nn=6 (especially with Ward)
    """
    from nipy.algorithms.graph.field import field_from_coo_matrix_and_data

    if method not in ['ward', 'gkm', 'ward_and_gkm', 'kmeans']:
        raise ValueError('unknown method')
    if nn not in [6, 18, 26]:
        raise ValueError('nn should be 6,18 or 26')

    # step 1: load the data ----------------------------
    # 1.1 the domain
    domain = grid_domain_from_image(mask_image, nn)

    if method is not 'kmeans':
        # 1.2 get the main cc of the graph
        # to remove the small connected components
        pass

    coord = domain.coord

    # 1.3 read the functional data
    beta = np.array([domain.make_feature_from_image(b) for b in betas])

    if len(beta.shape) > 2:
        beta = np.squeeze(beta)

    if beta.shape[0] != domain.size:
        beta = beta.T

    feature = np.hstack((beta, mu * coord / np.std(coord)))

    #step 2: parcellate the data ---------------------------

    if method is not 'kmeans':
        g = field_from_coo_matrix_and_data(domain.topology, feature)

    if method == 'kmeans':
        _, u, _ = kmeans(feature, nbparcel)

    if method == 'ward':
        u, _ = g.ward(nbparcel)

    if method == 'gkm':
        seeds = np.argsort(np.random.rand(g.V))[:nbparcel]
        _, u, _ = g.geodesic_kmeans(seeds)

    if method == 'ward_and_gkm':
        w, _ = g.ward(nbparcel)
        _, u, _ = g.geodesic_kmeans(label=w)

    lpa = SubDomains(domain, u)

    if verbose:
        var_beta = np.array(
            [np.var(beta[lpa.label == k], 0).sum() for k in range(lpa.k)])
        var_coord = np.array(
            [np.var(coord[lpa.label == k], 0).sum() for k in range(lpa.k)])
        size = lpa.get_size()
        vf = np.dot(var_beta, size) / size.sum()
        va = np.dot(var_coord, size) / size.sum()
        print nbparcel, "functional variance", vf, "anatomical variance", va

    # step3:  write the resulting label image
    if fullpath is not None:
        label_image = fullpath
    elif write_dir is not None:
        label_image = os.path.join(write_dir, "parcel_%s.nii" % method)
    else:
        label_image = None

    if label_image is not None:
        lpa_img = lpa.to_image(
            fid='id', roi=True, descrip='Intra-subject parcellation image')
        save(lpa_img, label_image)
        if verbose:
            print "Wrote the parcellation images as %s" % label_image

    return lpa
