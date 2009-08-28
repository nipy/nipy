"""
Utility functions for mutli-subjectParcellation:
this basically uses nipy io lib to perform IO opermation 
in parcel definition processes
"""


import numpy as np
import os.path
from nipy.io.imageformats import load, save, Nifti1Image 
from parcellation import Parcellation


def parcel_input(mask_images, nbeta, learning_images,
                ths = .5, fdim=3, affine=None):   
    """
    Instantiating a Parcel structure from a give set of input

    Parameters
    ----------
    mask_images: list of the paths of the mask images that are used
                 to define the common space. These can be cortex segmentations
                 (resampled at the same resolution as the remainder of the data)
                 Note that nsubj = len(mask_images)
    nbeta: list of integers, 
           ids of the contrast of under study
    learning_images: path of functional images used as input to the
      parcellation procedure. normally these are statistics(student/normal) images.
    ths=.5: threshold to select the regions that are common across subjects.
            if ths = .5, thethreshold is half the number of subjects
    affine=None provides the transformation to Talairach space.
                if affine==None, this is taken from the image header

    Results
    -------
    pa Parcellation instance  that stores the 
      individual masks and grid coordinates
    istats: nsubject-length list of arrays of shape
      (number of within-mask voxels of each subjet,fdim)
      which contains the amount of functional information
      available to parcel the data
    Talairach: array of size (nvoxels,3): MNI coordinates of the
      points corresponding to MXYZ
    """
    nsubj = len(mask_images)
    
    # Read the referential information
    nim = load(mask_images[0])
    ref_dim = nim.get_shape()
    grid_size = np.prod(ref_dim)
    if affine==None:
        affine = nim.get_affine()
    
    # take the individual masks
    mask = []
    for s in range(nsubj):
        nim = load(mask_images[s])
        temp = nim.get_data()
        rbeta = load(learning_images[s][0])
        maskb = rbeta.get_data()
        temp = np.minimum(temp,1-(maskb==0))        
        mask.append(temp)
        # fixme : check that all images are co-registered
        
    mask = np.squeeze(np.array(mask))

    # "intersect" the masks
    # fixme : this is nasty
    if ths ==.5:
        ths = nsubj/2
    else:
        ths = np.minimum(np.maximum(ths,0),nsubj-1)

    mask = mask>0
    smask = np.sum(mask,0)>ths
    mxyz = np.array(np.where(smask)).T
    nvox = mxyz.shape[0]
    mask = mask[:,smask>0].T    

    # Compute the position of each voxel in the common space    
    coord = np.dot(np.hstack((mxyz,np.ones((nvox,1)))),affine.T)[:,:3]
        
    # Load the functional data
    istats = []
    for s in range(nsubj): 
        stat = []
        lxyz = np.array(mxyz[mask[:,s],:])
        
        for b in range(nbeta):
            # the stats (noise-normalized contrasts) images
            rbeta = load(learning_images[s][b])
            temp = rbeta.get_data()
            temp = temp[lxyz[:,0],lxyz[:,1],lxyz[:,2]]
            temp = np.reshape(temp, np.size(temp))
            stat.append(temp)

        stat = np.array(stat)
        istats.append(stat.T)
    
    # Possibly reduce the dimension of the  functional data
    if fdim<istats[0].shape[1]:
        rstats = np.concatenate(istats)
        rstats = np.reshape(rstats,(rstats.shape[0],nbeta))
        rstats = rstats-np.mean(rstats)
        import numpy.linalg as nl
        m1,m2,m3 = nl.svd(rstats,0)
        rstats = np.dot(m1,np.diag(m2))
        rstats = rstats[:,:fdim]
        subj = np.concatenate([s*np.ones(istats[s].shape[0]) \
                               for s in range(nsubj)])
        istats = [rstats[subj==s] for s in range (nsubj)]

    pa = Parcellation(1,mxyz,mask-1)  
    
    return pa,istats,coord

def Parcellation_output(Pa, mask_images, learning_images, coord, nbru, 
                        verbose=1,swd = "/tmp"):
    """
    Function that produces images that describe the spatial structure
    of the parcellation.  It mainly produces label images at the group
    and subject level
    
    Parameters
    ----------
    Pa : Parcellation instance that describes the parcellation
    mask_images: list of images paths that define the mask
    learning_images: list of float images containing the input data
    coord: array of shape (nvox,3) that contains(approximated)
           MNI-coordinates of the brain mask voxels considered in the
           parcellation process
    nbru: list of subject ids
    verbose=1 : verbosity level
    swd = '/tmp': write directory
    
    Results
    -------
    Pa: the updated Parcellation instance
    """
    nsubj = Pa.nb_subj
    mxyz = Pa.ijk
    Pa.set_subjects(nbru)
    
    # write the template image
    tlabs = Pa.group_labels
    LabelImage = os.path.join(swd,"template_parcel.nii") 
    rmask = load(mask_images[0])
    ref_dim = rmask.get_shape()
    grid_size = np.prod(ref_dim)
    affine = rmask.get_affine()
    
    Label = np.zeros(ref_dim)
    Label[Pa.ijk[:,0],Pa.ijk[:,1],Pa.ijk[:,2]]=tlabs+1
    
    wim = Nifti1Image (Label, affine)
    hdr = wim.get_header()
    hdr['descrip'] = 'group_level Label image obtained from a \
                     parcellation procedure'
    save(wim, LabelImage)
    
    # write subject-related stuff
    Jac = []
    if Pa.isfield('jacobian'):
        Jac = Pa.get_feature('jacobian')
        Jac = np.reshape(Jac,(Pa.k,nsubj))
        
    for s in range(nsubj):
        # write the images
        labs = Pa.label[:,s]
        LabelImage = os.path.join(swd,"parcel%s.nii" % nbru[s])
        JacobImage = os.path.join(swd,"jacob%s.nii" % nbru[s])      

        Label = np.zeros(ref_dim).astype(np.int)
        Label[Pa.ijk[:,0],Pa.ijk[:,1],Pa.ijk[:,2]]=labs+1
        wim = Nifti1Image (Label, affine)
        hdr = wim.get_header()
        hdr['descrip'] = 'individual Label image obtained \
                         from a parcellation procedure'
        save(wim, LabelImage)

        if ((verbose)&(np.size(Jac)>0)):
            Label = np.zeros(ref_dim)
            Label[Pa.ijk[:,0],Pa.ijk[:,1],Pa.ijk[:,2]]=Jac[labs,s]
            wim = Nifti1Image (Label, affine)
            hdr = wim.get_header()
            hdr['descrip'] = 'image of the jacobian of the deformation \
                              associated with the parcellation'
            save(wim, LabelImage)       

    return Pa

def Parcellation_based_analysis(Pa, test_images, numbeta, swd="/tmp", 
                                    DMtx=None, verbose=1, method_id=0):
    """
    This function computes parcel averages and RFX at the parcel-level

    Parameters
    ----------
    Pa Parcellation instance that is updated in this function
    test_images: double list of paths of functional images used 
                 as input to for inference. 
                 Normally these are contrast images.
                 double list is 
                 [number of subjects [number of contrasts]]
    numbeta: list of int of the associated ids
    swd='/tmp': write directory
    DMtx=None: array od shape (nsubj,ncon) 
               a design matrix for second-level analyses 
              (not implemented yet)
    verbose=1: verbosity level
    method_id = 0: an id of the method used.
              This is useful to compare the outcome of different 
              Parcellation+RFX  procedures

    Results
    -------
    Pa: the updated Parcellation instance
    """
    nsubj = Pa.nb_subj
    mxyz = Pa.ijk.T
    mask = Pa.label>-1
    nbeta = len(numbeta)
    
    # 1. read the test data
    # fixme: Check that everybody is in the same referential
    Test = []
    for s in range(nsubj):
        beta = []
        lxyz = mxyz[:,mask[:,s]]
        lxyz = np.array(lxyz)

        for b in range(nbeta):
            # the raw contrast images   
            rbeta = load(test_images[s][b])
            temp = rbeta.get_data()
            temp = temp[lxyz[0,:],lxyz[1,:],lxyz[2,:]]
            temp = np.reshape(temp, np.size(temp))
            beta.append(temp)
            temp[np.isnan(temp)]=0 ##

        beta = np.array(beta)
        Test.append(beta.T) 

    # 2. compute the parcel-based stuff
    # and make inference inference (RFX,...)

    prfx = np.zeros((Pa.k,nbeta))
    vinter = np.zeros(nbeta)
    for b in range(nbeta):
        unitest = [np.reshape(Test[s][:,b],(np.size(Test[s][:,b]),1)) \
                  for s in range(nsubj)]
        cname = 'contrast_%04d'%(numbeta[b])
        Pa.make_feature(unitest, cname)
        prfx[:,b] =  np.reshape(Pa.PRFX(cname,1),Pa.k)
        vinter[b] = Pa.variance_inter(cname)

    vintra = Pa.variance_intra(Test)

    if verbose:
        print 'average intra-parcel variance', vintra
        print 'average intersubject variance', vinter.mean()
            
    # 3. Write the stuff
    # write RFX images
    ref_dim = rbeta.get_shape()
    affine = rbeta.get_affine()
    grid_size = np.prod(ref_dim)
    tlabs = Pa.group_labels

    # write the prfx images
    for b in range(len(numbeta)):
        RfxImage = os.path.join(swd,"prfx_%s_%d.nii" % (numbeta[b],method_id))
        if ((verbose)&(np.size(prfx)>0)):
            rfx_map = np.zeros(ref_dim)
            rfx_map[Pa.ijk[:,0],Pa.ijk[:,1],Pa.ijk[:,2]] = prfx[tlabs,b]
            wim = Nifti1Image (rfx_map, affine)
            hdr = wim.get_header()
            hdr['descrip'] = 'parcel-based eandom effects image (in z-variate)'
            save(wim, RfxImage)     
        
    return Pa


