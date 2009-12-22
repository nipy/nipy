"""
This module is the interface to the bayesian_structural_analysis (bsa) module
It handles the images provided as input and produces result images.

"""

import numpy as np
import os.path as op
from nipy.io.imageformats import load, save, Nifti1Image
from nipy.neurospin.utils.mask import intersect_masks
import nipy.neurospin.spatial_models.bayesian_structural_analysis as bsa
import nipy.neurospin.graph.field as ff



def make_bsa_image(mask_images, betas, theta=3., dmax= 5., ths=0, thq=0.5,
                   smin=0, swd="/tmp/", method='simple', subj_id=None,
                   nbeta='default', densPath=None, crPath=None, verbose=0):
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
    swd='/tmp': writedir
    method='simple': applied region detection method; to be chose among
                     'simple', 'dev','ipmi'
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
    Returns
    -------
    AF: an nipy.neurospin.spatial_models.structural_bfls.landmark_regions
        instance that describes the structures found at the group level
         None is returned if nothing has been found significant 
         at the group level
    BF : a list of nipy.neurospin.spatial_models.hroi.Nroi instances
       (one per subject) that describe the individual coounterpart of AF

    if method=='loo', the output is different:
        mll, float, the average likelihood of the data under H1 after cross validation
        ll0, float the log-likelihood of the data under the global null
  
    fixme: unique mask should be allowed
    """
    # Sanity check
    if len(mask_images)!=len(betas):
        raise ValueError,"the number of masks and activation images\
        should be the same"
    nsubj = len(mask_images)
    if subj_id==None:
        subj_id = [str[i] for i in range(nsubj)]
    
    # Read the referential information
    nim = load(mask_images[0])
    ref_dim = nim.get_shape()
    affine = nim.get_affine()
    
    # Read the masks and compute the "intersection"
    mask = intersect_masks(mask_images)
    xyz = np.array(np.where(mask)).T
    nvox = xyz.shape[0]

    # create the field strcture that encodes image topology
    Fbeta = ff.Field(nvox)
    Fbeta.from_3d_grid(xyz.astype(np.int),18)

    # Get  coordinates in mm
    xyz = np.hstack((xyz,np.ones((nvox,1))))
    coord = np.dot(xyz,affine.T)[:,:3]
    xyz = xyz.astype(np.int)
    
    # read the functional images
    lbeta = []
    for s in range(nsubj):
        rbeta = load(betas[s])
        beta = rbeta.get_data()
        beta = beta[mask]
        lbeta.append(beta)
    lbeta = np.array(lbeta).T

    # launch the method
    g0 = 1.0/(np.absolute(np.linalg.det(affine))*nvox)
    bdensity = 1
    crmap = np.zeros(nvox)
    p = np.zeros(nvox)
    AF = None
    BF = [None for s in range(nsubj)]

    if method=='ipmi':
        crmap,AF,BF,p = bsa.compute_BSA_ipmi(Fbeta, lbeta, coord, dmax, 
                        xyz[:,:3], affine, ref_dim, thq, smin, ths,
                        theta, g0, bdensity, verbose=verbose)
    if method=='dev':
        crmap,AF,BF,p = bsa.compute_BSA_dev  (Fbeta, lbeta, coord, 
                        dmax, xyz[:,:3], affine, ref_dim, 
                        thq, smin,ths, theta, g0, bdensity, verbose=verbose)
    if method=='simple':
        crmap,AF,BF,p = bsa.compute_BSA_simple (Fbeta, lbeta, coord, dmax, 
                        xyz[:,:3], affine, ref_dim, 
                        thq, smin, ths, theta, g0, verbose=verbose)
        
    if method=='simple_quick':
        crmap,AF,BF,co_clust = bsa.compute_BSA_simple_quick(Fbeta, lbeta, coord, dmax, 
                        xyz[:,:3], affine, ref_dim, 
                        thq, smin, ths, theta, g0, verbose=verbose)
        density = np.zeros(nvox)
        crmap = AF.map_label(coord,0.95,dmax)

    if method=='loo':
         mll, ll0 = bsa.compute_BSA_loo (Fbeta, lbeta, coord, dmax, 
                                xyz[:,:3], affine, ref_dim, 
                                thq, smin,ths, theta, g0, verbose=verbose)
         return mll, ll0
    
                    
    # Write the results as images
    # the spatial density image
    if densPath != False:
        density = np.zeros(ref_dim)
        density[mask] = p
        wim = Nifti1Image (density, affine)
        wim.get_header()['descrip'] = 'group-level spatial density of active regions'
        if densPath==None:
            densPath = op.join(swd,"density_%s.nii"%nbeta)
        save(wim, densPath)
    
    if crPath==False:
        return AF, BF

    if AF==None:
        default_idx = 0
    else:
        default_idx = AF.k+2
    
    if crPath==None:
        # write a 3D image for group-level labels
        crPath = op.join(swd,"CR_%s.nii"%nbeta)
        Label = -2*np.ones(ref_dim,'int16')
        Label[mask] = crmap
        wim = Nifti1Image (Label, affine)
        wim.get_header()['descrip'] = 'group Level labels from bsa procedure'
        save(wim, crPath)

        #write 3d images for the subjects
        for s in range(nsubj):
            LabelImage = op.join(swd,"AR_s%s_%s.nii"%(subj_id[s],nbeta))
            Label = -2*np.ones(ref_dim,'int16')
            Label[mask]=-1
            if BF[s]!=None:
                nls = BF[s].get_roi_feature('label')
                nls[nls==-1] = default_idx
                for k in range(BF[s].k):
                    xyzk = BF[s].xyz[k].T 
                    Label[xyzk[0],xyzk[1],xyzk[2]] =  nls[k]
        
            wim = Nifti1Image (Label, affine)
            wim.get_header()['descrip'] = 'Individual label image from bsa procedure'
            save(wim, LabelImage)
    else:
        # write everything in a single 4D image
        wdim = (ref_dim[0], ref_dim[1], ref_dim[2], nsubj+1)
        Label = -2*np.ones(wdim,'int16')
        Label[mask,0] = crmap
        for s in range(nsubj):
            Label[mask,s+1]=-1
            if BF[s]!=None:
                nls = BF[s].get_roi_feature('label')
                nls[nls==-1] = default_idx
                for k in range(BF[s].k):
                    xyzk = BF[s].xyz[k].T 
                    Label[xyzk[0],xyzk[1],xyzk[2],s+1] =  nls[k]
        wim = Nifti1Image (Label, affine)
        wim.get_header()['descrip'] = 'group Level and individual labels\
            from bsa procedure'
        save(wim, crPath)
        
    return AF,BF

def make_bsa_image_with_output_paths(mask_images, betas, denspath, crpath,
                                     theta=3., dmax= 5., ths=0, thq=0.5, smin=0,
                                     method='simple'):
    """
    Deprecated : will be removed soon

    idem make_bsa_image but paths of the output are set explictly.
    Moreover the segmented regions are written in one single image 
    """
    # Sanity check
    if len(mask_images)!=len(betas):
        print len(mask_images),len(betas)        
        raise ValueError,"the number of masks and activation images\
        should be the same"
    nsubj = len(mask_images)
        
    # Read the referential information
    nim = load(mask_images[0])
    ref_dim = nim.get_shape()
    affine = nim.get_affine()
    
    # Read the masks and compute the "intersection"
    mask = intersect_masks(mask_images)
    xyz = np.array(np.where(mask)).T
    nvox = xyz.shape[0]

    # create the field strcture that encodes image topology
    Fbeta = ff.Field(nvox)
    Fbeta.from_3d_grid(xyz.astype(np.int),18)

    # Get  coordinates in mm
    xyz = np.hstack((xyz,np.ones((nvox,1))))
    coord = np.dot(xyz,affine.T)[:,:3]
    xyz = xyz.astype(np.int)
    
    # read the functional images
    lbeta = []
    for s in range(nsubj):
        rbeta = load(betas[s])
        beta = rbeta.get_data()
        beta = beta[mask]
        lbeta.append(beta)
    lbeta = np.array(lbeta).T
    lbeta = np.reshape(lbeta,(nvox,nsubj))

    # launch the method
    g0 = 1.0/(np.absolute(np.linalg.det(affine))*nvox)
    bdensity = 1
    crmap = np.zeros(nvox)
    p = np.zeros(nvox)
    AF = None
    BF = [None for s in range(nsubj)]

    if method=='ipmi':
        crmap,AF,BF,p = bsa.compute_BSA_ipmi(Fbeta, lbeta, coord, dmax, 
                        xyz[:,:3], affine, ref_dim, thq, smin, ths,
                        theta, g0, bdensity)
    if method=='dev':
        crmap,AF,BF,p = bsa.compute_BSA_dev  (Fbeta, lbeta, coord, 
                        dmax, xyz[:,:3], affine, ref_dim, 
                        thq, smin,ths, theta, g0, bdensity,verbose=1)
    if method=='simple':
        crmap,AF,BF,p = bsa.compute_BSA_simple (Fbeta, lbeta, coord, dmax, 
                        xyz[:,:3], affine, ref_dim, 
                        thq, smin, ths, theta, g0, verbose=0)
        
    if method=='simple_quick':
        crmap,AF,BF,co_clust = bsa.compute_BSA_simple_quick (Fbeta, lbeta, coord, dmax, 
                        xyz[:,:3], affine, ref_dim, 
                        thq, smin, ths, theta, g0, verbose=0)
        density = np.zeros(nvox)
        crmap = AF.map_label(coord,0.95,dmax)

    if method=='loo':
        crmap,AF,BF,p = bsa.compute_BSA_loo (Fbeta, lbeta, coord, dmax, 
                        xyz[:,:3], affine, ref_dim, 
                        thq, smin,ths, theta, g0, verbose=0)
    
                    
    # Write the results
    Label = -2*np.ones(ref_dim,'int16')
    Label[mask] = crmap.astype('i')
    wim = Nifti1Image (Label, affine)
    wim.get_header()['descrip'] = 'group Level labels from bsa procedure'
    save(wim, crpath)

    density = np.zeros(ref_dim)
    density[mask] = p
    wim = Nifti1Image (density, affine)
    wim.get_header()['descrip'] = 'group-level spatial density of active regions'
    save(wim, denspath)

    if AF==None:
        default_idx = 0
    else:
        default_idx = AF.k+2
    
    # write everything in one image
    wdim = (ref_dim[0], ref_dim[1], ref_dim[2], nsubj+1)
    Label = -2*np.ones(wdim,'int16')
    Label[mask,0] = crmap.astype(np.int)
    for s in range(nsubj):
        Label[mask,s+1]=-1
        if BF[s]!=None:
            nls = BF[s].get_roi_feature('label')
            nls[nls==-1] = default_idx
            for k in range(BF[s].k):
                xyzk = BF[s].xyz[k].T 
                Label[xyzk[0],xyzk[1],xyzk[2],s+1] =  nls[k]
    wim = Nifti1Image (Label, affine)
    wim.get_header()['descrip'] = 'group Level and individual labels\
        from bsa procedure'
    save(wim, crpath)

    return AF,BF, maxc
    
