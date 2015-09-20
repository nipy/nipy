# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This module is the interface to the bayesian_structural_analysis (bsa) module
It handles the images provided as input and produces result images.
"""
from __future__ import absolute_import

import numpy as np
import os.path as op
from nibabel import load, save, Nifti1Image

from ..mask import intersect_masks
from .bayesian_structural_analysis import compute_landmarks
from .discrete_domain import domain_from_image
from ...io.nibcompat import get_header, get_affine

from ...externals.six import string_types

def make_bsa_image(
    mask_images, stat_images, threshold=3., smin=0, sigma=5.,
    prevalence_threshold=0, prevalence_pval=0.5, write_dir=None,
    algorithm='density', contrast_id='default'):
    """ Main function for  performing bsa on a set of images.
    It creates the some output images in the given directory

    Parameters
    ----------
    mask_images: list of str,
                 image paths that yield mask images, one for each subject
    stat_images: list of str,
                 image paths of the activation images, one for each subject
    threshold: float, optional,
               threshold used to ignore all the image data that is below
    smin: float, optional,
          minimal size (in voxels) of the extracted blobs
          smaller blobs are merged into larger ones
    sigma: float, optional,
           variance of the spatial model, i.e. cross-subject uncertainty
    prevalence_threshold: float, optional
                          threshold on the representativity measure
    prevalence_pval: float, optional,
                     p-value of the representativity test:
             test = p(representativity>prevalence_threshold) > prevalence_pval
    write_dir: string, optional,
               if not None, output directory
    method: {'density', 'co-occurrence'}, optional,
            Inference method used in the landmark definition
    contrast_id: string, optional,
                 identifier of the contrast

    Returns
    -------
    landmarks: nipy.labs.spatial_models.structural_bfls.landmark_regions
         instance that describes the structures found at the group level
         None is returned if nothing has been found significant
         at the group level
    hrois : list of nipy.labs.spatial_models.hroi.Nroi instances,
       (one per subject), describe the individual counterpart of landmarks
    """
    n_subjects = len(stat_images)

    # Read the referential information
    nim = load(stat_images[0])
    ref_dim = nim.shape[:3]
    affine = get_affine(nim)

    # Read the masks and compute the "intersection"
    # mask = np.reshape(intersect_masks(mask_images), ref_dim).astype('u8')
    if isinstance(mask_images, string_types):
        mask = load(mask_images).get_data()
    elif isinstance(mask_images, Nifti1Image):
        mask = mask_images.get_data()
    else:
        # mask_images should be a list of strings or images
        mask = intersect_masks(mask_images).astype('u8')

    # encode it as a domain
    domain = domain_from_image(Nifti1Image(mask, affine), nn=18)
    n_voxels = domain.size

    # read the functional images
    stats = []
    for stat_image in stat_images:
        beta = np.reshape(load(stat_image).get_data(), ref_dim)
        stats.append(beta[mask > 0])
    stats = np.array(stats).T

    # launch the method
    crmap = - np.ones(n_voxels).astype(np.int16)
    density = np.zeros(n_voxels)
    landmarks = None
    hrois = [None for _ in range(n_subjects)]
    default_idx = 0
    prevalence = np.array([])

    landmarks, hrois = compute_landmarks(
        domain, stats, sigma, prevalence_pval, prevalence_threshold,
        threshold, smin, algorithm=algorithm)

    if landmarks is not None:
        crmap = landmarks.map_label(domain.coord, 0.95, sigma).astype(np.int16)
        density = landmarks.kernel_density(
            k=None, coord=domain.coord, sigma=sigma)
        default_idx = landmarks.k + 2
        prevalence = landmarks.roi_prevalence()

    if write_dir == False:
        return landmarks, hrois

    # Write the results as images

    # the spatial density image
    density_map = np.zeros(ref_dim)
    density_map[mask > 0] = density
    wim = Nifti1Image(density_map, affine)
    get_header(wim)['descrip'] = ('group-level spatial density '
                                  'of active regions')
    dens_path = op.join(write_dir, "density_%s.nii" % contrast_id)
    save(wim, dens_path)

    # write a 3D image for group-level labels
    labels = - 2 * np.ones(ref_dim)
    labels[mask > 0] = crmap
    wim = Nifti1Image(labels.astype('int16'), affine)
    get_header(wim)['descrip'] = 'group Level labels from bsa procedure'
    save(wim, op.join(write_dir, "CR_%s.nii" % contrast_id))

    # write a prevalence image
    prev_ = np.zeros(crmap.size).astype(np.float)
    prev_[crmap > -1] = prevalence[crmap[crmap > -1]]
    prevalence_map = - np.ones(ref_dim)
    prevalence_map[mask > 0] = prev_
    wim = Nifti1Image(prevalence_map, affine)
    get_header(wim)['descrip'] = 'Weighted prevalence image'
    save(wim, op.join(write_dir, "prevalence_%s.nii" % contrast_id))

    # write a 4d images with all subjects results
    wdim = (ref_dim[0], ref_dim[1], ref_dim[2], n_subjects)
    labels = - 2 * np.ones(wdim, 'int16')
    for subject in range(n_subjects):
        labels[mask > 0, subject] = - 1
        if hrois[subject].k > 0:
            nls = hrois[subject].get_roi_feature('label')
            nls[nls == - 1] = default_idx
            lab = hrois[subject].label
            lab[lab > - 1] = nls[lab[lab > - 1]]
            labels[mask > 0, subject] = lab
    wim = Nifti1Image(labels, affine)
    get_header(wim)['descrip'] = 'Individual labels from bsa procedure'
    save(wim, op.join(write_dir, "AR_%s.nii" % contrast_id))
    return landmarks, hrois
