# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This module is the interface to the bayesian_structural_analysis (bsa) module
It handles the images provided as input and produces result images.

"""

import numpy as np
import os.path as op
from nibabel import load, save, Nifti1Image

from nipy.neurospin.mask import intersect_masks
import nipy.neurospin.spatial_models.bayesian_structural_analysis as bsa
from discrete_domain import domain_from_image 


def make_bsa_image(
    mask_images, betas, theta=3., dmax= 5., ths=0, thq=0.5, smin=0, swd=None,
    method='simple', subj_id=None, nbeta='default', densPath=None,
    crPath=None, verbose=0, reshuffle=False):
    """
    main function for  performing bsa on a set of images.
    It creates the some output images in the given directory

    Parameters
    ------------
    mask_images: A list of image paths that yield binary images,
                 one for each subject
                 the number os subjects, nsubj, is taken as len(mask_images)
    betas: A list of image paths that yields the activation images,
           one for each subject
    theta=3., threshold used to ignore all the image data that si below
    dmax=5., prior width of the spatial model;
             corresponds to multi-subject uncertainty 
    ths=0: threshold on the representativity measure of the obtained
           regions
    thq=0.5: p-value of the representativity test:
             test = p(representativity>ths)>thq
    smin=0: minimal size (in voxels) of the extracted blobs
            smaller blobs are merged into larger ones
    swd: string, optional
        if not None, output directory
    method='simple': applied region detection method; to be chose among
                     'simple', 'ipmi'
    subj_id=None: list of strings, identifiers of the subjects.
                  by default it is range(nsubj)
    nbeta='default', string, identifier of the contrast
    densPath=None, string, path of the output density image
                   if False, no image is written
                   if None, the path is computed from swd, nbeta
    crPath=None,  string, path of the (4D) output label image
                  if False, no ime is written
                  if None, many images are written, 
                  with paths computed from swd, subj_id and nbeta
    reshuffle: bool, optional
               if true, randomly swap the sign of the data
    
    Returns
    -------
    AF: an nipy.neurospin.spatial_models.structural_bfls.landmark_regions
        instance that describes the structures found at the group level
         None is returned if nothing has been found significant 
         at the group level
    BF : a list of nipy.neurospin.spatial_models.hroi.Nroi instances
       (one per subject) that describe the individual coounterpart of AF

    if method=='loo', the output is different:
        mll, float, the average likelihood of the data under the model
        after cross validation
        ll0, float the log-likelihood of the data under the global null
  
    fixme: unique mask should be allowed
    """
    # Sanity check
    if len(mask_images)!=len(betas):
        raise ValueError,"the number of masks and activation images\
        should be the same"
    nsubj = len(mask_images)
    if subj_id==None:
        subj_id = [str(i) for i in range(nsubj)]
    
    # Read the referential information
    nim = load(mask_images[0])
    ref_dim = nim.get_shape()[:3]
    affine = nim.get_affine()
    
    # Read the masks and compute the "intersection"
    mask = np.reshape(intersect_masks(mask_images), ref_dim)
 
    # encode it as a domain
    dom = domain_from_image(Nifti1Image(mask, affine), nn=18)
    nvox = dom.size
    
    # read the functional images
    lbeta = []
    for s in range(nsubj):
        rbeta = load(betas[s])
        beta = np.reshape(rbeta.get_data(), ref_dim)
        lbeta.append(beta[mask])
    lbeta = np.array(lbeta).T

    if reshuffle:
        rswap = 2*(np.random.randn(nsubj)>0.5)-1
        lbeta = np.dot(lbeta, np.diag(rswap))
        
    # launch the method
    crmap = np.zeros(nvox)
    p = np.zeros(nvox)
    AF = None
    BF = [None for s in range(nsubj)]

    if method=='ipmi':
        crmap,AF,BF,p = bsa.compute_BSA_ipmi(dom, lbeta, dmax, thq, smin,
                                             ths, theta, verbose=verbose)
    if method=='simple':
        crmap,AF,BF,p = bsa.compute_BSA_simple(
            dom, lbeta, dmax, thq, smin, ths, theta, verbose=verbose)
        
    if method=='quick':
        crmap, AF, BF, co_clust = bsa.compute_BSA_quick(
            dom, lbeta, dmax, thq, smin, ths, theta, verbose=verbose)
            
        density = np.zeros(nvox)
        crmap = AF.map_label(dom.coord, 0.95, dmax)

    if method=='loo':
         mll, ll0 = bsa.compute_BSA_loo (
             dom, lbeta, dmax, thq, smin, ths, theta, verbose=verbose)
         
         return mll, ll0
    
                    
    # Write the results as images
    # the spatial density image
    if densPath != False:
        density = np.zeros(ref_dim)
        density[mask] = p
        wim = Nifti1Image (density, affine)
        wim.get_header()['descrip'] = 'group-level spatial density \
                                       of active regions'
        if densPath==None:
            densPath = op.join(swd,"density_%s.nii"%nbeta)
        save(wim, densPath)
    
    if crPath==False:
        return AF, BF

    
    default_idx = AF.k+2

    if crPath==None and swd==None:
        return AF, BF

    if crPath==None:
        # write a 3D image for group-level labels
        crPath = op.join(swd, "CR_%s.nii"%nbeta)
        Label = -2*np.ones(ref_dim, 'int16')
        Label[mask] = crmap
        wim = Nifti1Image (Label, affine)
        wim.get_header()['descrip'] = 'group Level labels from bsa procedure'
        save(wim, crPath)

        # write a prevalence image
        crPath = op.join(swd, "prevalence_%s.nii"%nbeta)
        # prev = AF.prevalence_density()
        prev = np.zeros(ref_dim)
        prev[mask] = AF.prevalence_density()
        wim = Nifti1Image (prev, affine)
        wim.get_header()['descrip'] = 'Weighted prevalence image'
        save(wim, crPath)
            
        #write 3d images for the subjects
        for s in range(nsubj):
            LabelImage = op.join(swd,"AR_s%s_%s.nii"%(subj_id[s],nbeta))
            Label = -2*np.ones(ref_dim, 'int16')
            Label[mask] = -1
            if BF[s] != None:
                nls = BF[s].get_roi_feature('label')
                nls[nls==-1] = default_idx
                lab = BF[s].label
                lab[lab>-1] = nls[lab[lab>-1]]
                Label[mask] = lab
               
        
            wim = Nifti1Image (Label, affine)
            wim.get_header()['descrip'] = \
                'Individual label image from bsa procedure'
            save(wim, LabelImage)
    else:
        # write everything in a single 4D image
        wdim = (ref_dim[0], ref_dim[1], ref_dim[2], nsubj+1)
        Label = -2*np.ones(wdim,'int16')
        Label[mask, 0] = crmap
        for s in range(nsubj):
            Label[mask,s+1]=-1
            if BF[s]!=None:
                nls = BF[s].get_roi_feature('label')
                nls[nls==-1] = default_idx
                lab = BF[s].label
                lab[lab>-1] = nls[lab[lab>-1]]
                Label[mask, s+1] = lab
                #for k in range(BF[s].k):
                #    Label[mask, s+1][BF[s].label==k] =  nls[k]
        wim = Nifti1Image (Label, affine)
        wim.get_header()['descrip'] = 'group Level and individual labels\
            from bsa procedure'
        save(wim, crPath)
        
    return AF,BF

