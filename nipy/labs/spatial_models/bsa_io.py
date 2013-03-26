# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This module is the interface to the bayesian_structural_analysis (bsa) module
It handles the images provided as input and produces result images.
"""

import numpy as np
import os.path as op
from nibabel import load, save, Nifti1Image

from ..mask import intersect_masks
from .bayesian_structural_analysis import compute_landmarks
from .discrete_domain import domain_from_image


def make_bsa_image(
    mask_images, stat_images, threshold=3., sigma=5., prevalence_threshold=0,
    prevalence_pval=0.5, smin=0, write_dir=None, method='simple', 
    subjects_id=None, contrast_id='default', dens_path=None, cr_path=None,
    verbose=0):
    """ Main function for  performing bsa on a set of images.
    It creates the some output images in the given directory

    Parameters
    ----------
    mask_images: A list of image paths that yield binary images,
                 one for each subject
                 the number os subjects, n_subjects, is taken as len(mask_images)
    stat_images: A list of image paths that yields the activation images,
           one for each subject
    threshold=3., threshold used to ignore all the image data that si below
    sigma=5., prior width of the spatial model
             corresponds to multi-subject uncertainty
    prevalence_threshold: float, optional
                          threshold on the representativity measure
    prevalence_pval: float, optional 
                     p-value of the representativity test:
             test = p(representativity>prevalence_threshold) > prevalence_pval
    smin: float, optional
          minimal size (in voxels) of the extracted blobs
          smaller blobs are merged into larger ones
    write_dir: string, optional
               if not None, output directory
    method: string, one of ['simple', 'quick'], optional, 
            applied region detection method; to be chose among
    subjects_id=None: list of strings, identifiers of the subjects.
                  by default it is range(n_subjects)
    contrast_id='default', string, identifier of the contrast
    dens_path=None, string, path of the output density image
                   if False, no image is written
                   if None, the path is join(write_dir, density_contrast_id.nii)
    cr_path=None,  string, path of the (4D) output label image
                  if False, no image is written
                  if None, many images are written,
                  with paths computed from write_dir, subjects_id and contrast_id

    Returns
    -------
    landmarks: an nipy.labs.spatial_models.structural_bfls.landmark_regions
        instance that describes the structures found at the group level
         None is returned if nothing has been found significant
         at the group level
    hrois : a list of nipy.labs.spatial_models.hroi.Nroi instances
       (one per subject) that describe the individual coounterpart of landmarks

    fixme
    =====
    unique mask should be allowed
    """
    # Sanity check
    if len(mask_images) != len(stat_images):
        raise ValueError("the number of masks and activation images" \
                             "should be the same")
    n_subjects = len(mask_images)
    if subjects_id == None:
        subjects_id = [str(subject) for subject in range(n_subjects)]

    # Read the referential information
    nim = load(mask_images[0])
    ref_dim = nim.shape[:3]
    affine = nim.get_affine()

    # Read the masks and compute the "intersection"
    mask = np.reshape(intersect_masks(mask_images), ref_dim).astype('u8')

    # encode it as a domain
    domain = domain_from_image(Nifti1Image(mask, affine), nn=18)
    nvox = domain.size

    # read the functional images
    lbeta = []
    for subject in range(n_subjects):
        rbeta = load(stat_images[subject])
        beta = np.reshape(rbeta.get_data(), ref_dim)
        lbeta.append(beta[mask > 0])
    lbeta = np.array(lbeta).T

    # launch the method
    crmap = np.zeros(nvox)
    density = np.zeros(nvox)
    landmarks = None
    hrois = [None for s in range(n_subjects)]

    if method == 'simple':
        crmap, landmarks, hrois, density = compute_landmarks(
            domain, lbeta, sigma, prevalence_pval, prevalence_threshold,
            threshold, smin, algorithm='standard', verbose=verbose)

    if method == 'quick':
        crmap, landmarks, hrois, co_clust = compute_landmarks(
            domain, lbeta, sigma, prevalence_pval, prevalence_threshold,
            threshold, smin, algorithm='quick', verbose=verbose)

        density = np.zeros(nvox)
        crmap = landmarks.map_label(domain.coord, 0.95, sigma)

    # Write the results as images
    # the spatial density image
    if dens_path is not False:
        density_map = np.zeros(ref_dim)
        density_map[mask > 0] = density
        wim = Nifti1Image(density_map, affine)
        wim.get_header()['descrip'] = 'group-level spatial density \
                                       of active regions'
        if dens_path == None:
            dens_path = op.join(write_dir, "density_%s.nii" % contrast_id)
        save(wim, dens_path)

    if cr_path == False:
        return landmarks, hrois

    default_idx = landmarks.k + 2

    if cr_path == None and write_dir == None:
        return landmarks, hrois

    if cr_path == None:
        # write a 3D image for group-level labels
        cr_path = op.join(write_dir, "CR_%s.nii" % contrast_id)
        labels = - 2 * np.ones(ref_dim)
        labels[mask > 0] = crmap
        wim = Nifti1Image(labels.astype('int16'), affine)
        wim.get_header()['descrip'] = 'group Level labels from bsa procedure'
        save(wim, cr_path)

        # write a prevalence image
        cr_path = op.join(write_dir, "prevalence_%s.nii" % contrast_id)
        prevalence_map = np.zeros(ref_dim)
        prevalence_map[mask > 0] = landmarks.prevalence_density()
        wim = Nifti1Image(prevalence_map, affine)
        wim.get_header()['descrip'] = 'Weighted prevalence image'
        save(wim, cr_path)

        # write 3d images for the subjects
        for subject in range(n_subjects):
            label_image = op.join(write_dir, "AR_s%s_%s.nii" % (
                    subjects_id[subject], contrast_id))
            labels = - 2 * np.ones(ref_dim)
            labels[mask > 0] = -1
            if hrois[subject] is not None:
                nls = hrois[subject].get_roi_feature('label').astype(np.int)
                nls[nls == - 1] = default_idx
                lab = hrois[subject].label
                lab[lab > - 1] = nls[lab[lab > - 1]]
                labels[mask > 0] = lab

            wim = Nifti1Image(labels.astype('int16'), affine)
            wim.get_header()['descrip'] = \
                'Individual label image from bsa procedure'
            save(wim, label_image)
    else:
        # write everything in a single 4D image
        wdim = (ref_dim[0], ref_dim[1], ref_dim[2], n_subjects + 1)
        labels = - 2 * np.ones(wdim, 'int16')
        labels[mask > 0, 0] = crmap
        for subject in range(n_subjects):
            labels[mask > 0, subject + 1] = - 1
            if hrois[s] is not None:
                nls = hrois[subject].get_roi_feature('label')
                nls[nls == - 1] = default_idx
                lab = hrois[subject].label
                lab[lab > - 1] = nls[lab[lab > - 1]]
                labels[mask > 0, subject + 1] = lab
        wim = Nifti1Image(labels, affine)
        wim.get_header()['descrip'] = 'group Level and individual labels\
            from bsa procedure'
        save(wim, cr_path)

    return landmarks, hrois
