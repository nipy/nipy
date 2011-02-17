# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Utility functions for mutli-subjectParcellation:
this basically uses nipy io lib to perform IO opermation 
in parcel definition processes
"""

from nipy.neurospin.clustering.clustering import kmeans
import numpy as np
import os.path
from nibabel import load, save, Nifti1Image 
from parcellation import Parcellation

def mask_parcellation(mask_images, nb_parcel, output_image=None):
    """
    Performs the parcellation of a certain mask

    Parameters
    ----------
    mask_images: list of strings,
                 paths of the mask images that define the common space.
    nb_parcel: int,
               number of desired parcels
    output_image: string, optional
                   path of the output image
                   
    Returns
    -------
    wim: Nifti1Imagine instance,  the resulting parcellation
    """
    from ..mask import intersect_masks

    # compute the group mask
    affine = load(mask_images[0]).get_affine()
    shape = load(mask_images[0]).get_shape()
    mask = intersect_masks(mask_images, threshold=0)>0
    ijk = np.where(mask)
    ijk = np.array(ijk).T
    nvox = ijk.shape[0]

    # Get and cluster  coordinates 
    ijk = np.hstack((ijk,np.ones((nvox,1))))
    coord = np.dot(ijk, affine.T)[:,:3]
    cent, tlabs, J = kmeans(coord, nb_parcel)
        
    # Write the results
    label = -np.ones(shape)
    label[mask]= tlabs
    wim = Nifti1Image(label, affine)
    wim.get_header()['descrip'] = 'Label image in %d parcels'%nb_parcel    
    if output_image is not None:
        save(wim, output_image)
    return wim
    

def parcel_input(mask_images, nbeta, learning_images,
                ths = .5, fdim=3, affine=None):   
    """
    Instantiating a Parcel structure from a give set of input

    Parameters
    ----------
    mask_images: list of strings,
                 paths of the mask images that  define the common space.
                 These can be cortex segmentations
                 (at the same resolution as the remainder of the data)
                 Note that nsubj = len(mask_images)
    nbeta: list of integers, 
           ids of the contrast of under study
    learning_images: path of functional images used as input to the
                     parcellation procedure. normally these are statistics
                     (student/normal) images.
    ths=.5: threshold to select the regions that are common across subjects.
            if ths = .5, thethreshold is half the number of subjects
    fdim=3, int
            dimension of the data used in subsequent analyses 
            if smaller than len(nbeta), 
            a PCA is perfomed to reduce the information in the data
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
    if affine==None:
        affine = nim.get_affine()

    # take the individual masks
    mask = []
    for s in range(nsubj):
        nim = load(mask_images[s])
        temp = np.squeeze(nim.get_data())
        rbeta = load(learning_images[s][0])
        maskb = np.squeeze(rbeta.get_data())
        temp = np.minimum(temp, 1-(maskb==0))        
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
    mask = mask[:, smask>0].T    

    # Compute the position of each voxel in the common space    
    coord = np.dot(np.hstack((mxyz, np.ones((nvox, 1)))), affine.T)[:, :3]
        
    # Load the functional data
    istats = []
    for s in range(nsubj): 
        stat = []
        lxyz = np.array(mxyz[mask[:,s], :])
        
        for b in range(nbeta):
            # the stats (noise-normalized contrasts) images
            rbeta = load(learning_images[s][b])
            temp = rbeta.get_data()
            temp = temp[lxyz[:,0], lxyz[:,1], lxyz[:,2]]
            temp = np.reshape(temp, np.size(temp))
            stat.append(temp)

        stat = np.array(stat)
        istats.append(stat.T)
    
    # Possibly reduce the dimension of the  functional data
    if fdim<istats[0].shape[1]:
        rstats = np.concatenate(istats)
        rstats = np.reshape(rstats,(rstats.shape[0], nbeta))
        rstats = rstats-np.mean(rstats)
        import numpy.linalg as nl
        m1,m2,m3 = nl.svd(rstats, 0)
        rstats = np.dot(m1, np.diag(m2))
        rstats = rstats[:, :fdim]
        subj = np.concatenate([s*np.ones(istats[s].shape[0]) \
                               for s in range(nsubj)])
        istats = [rstats[subj==s] for s in range (nsubj)]

    pa = Parcellation(1,mxyz,mask-1)  
    
    return pa, istats, coord

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
    Pa.set_subjects(nbru)
    
    # write the template image
    tlabs = Pa.group_labels
    LabelImage = os.path.join(swd,"template_parcel.nii") 
    rmask = load(mask_images[0])
    ref_dim = rmask.get_shape()
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
            save(wim, JacobImage)       

    return Pa

def parcellation_output_with_paths(Pa, mask_images, group_path, indiv_path):
    """
    Function that produces images that describe the spatial structure
    of the parcellation.  It mainly produces label images at the group
    and subject level
    
    Parameters
    ----------
    Pa : Parcellation instance that describes the parcellation
    mask_images: list of images paths that define the mask
    coord: array of shape (nvox,3) that contains(approximated)
           MNI-coordinates of the brain mask voxels considered in the
           parcellation process
    group_path, string, path of the group-level parcellation image
    indiv_path, list of strings, paths of the individual parcellation images   
    
    fixme
    -----
    the referential-defining information should be part of the Pa instance
    """
    nsubj = Pa.nb_subj
    
    # write the template image
    tlabs = Pa.group_labels
    rmask = load(mask_images[0])
    ref_dim = rmask.get_shape()
    affine = rmask.get_affine()
    
    Label = np.zeros(ref_dim)
    Label[Pa.ijk[:,0],Pa.ijk[:,1],Pa.ijk[:,2]]=tlabs+1
    
    wim = Nifti1Image (Label, affine)
    hdr = wim.get_header()
    hdr['descrip'] = 'group_level Label image obtained from a \
                     parcellation procedure'
    save(wim, group_path)
    
    # write subject-related stuff
    for s in range(nsubj):
        # write the images
        labs = Pa.label[:,s]
        Label = np.zeros(ref_dim).astype(np.int)
        Label[Pa.ijk[:,0],Pa.ijk[:,1],Pa.ijk[:,2]]=labs+1
        wim = Nifti1Image (Label, affine)
        hdr = wim.get_header()
        hdr['descrip'] = 'individual Label image obtained \
                         from a parcellation procedure'
        save(wim, indiv_path[s])
    


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


def one_subj_parcellation(MaskImage, betas, nbparcel, nn=6, method='ward', 
                          write_dir=None, mu=10., verbose=0, fullpath=None):
    """
    Parcellation of a one-subject dataset
    Return: a tuple (Parcellation instance, parcellation labels)
    
    Parameters
    ----------
    MaskImage: path to the mask-defining_image of the subject
    betas: list of paths to activation images from the subject
    nbparcel, int : number fo desired parcels
    nn=6: number of nearest neighbors  to define the image topology 
          (6, 18 or 26)
    method='ward': clustering method used, to be chosen among
                   'ward', 'gkm', 'ward_and-gkm'
                   'ward': Ward's clustering algorithm
                   'gkm': Geodesic k-means algorithm, random initialization
                   'gkm_and_ward': idem, initialized by Ward's clustering
    write_dir=None: write directory. If fullpath is None too, then no file output.
    mu = 10., float: the relative weight of anatomical information
    verbose=0: verbosity mode
    fullpath=None, string,
                   path of the output image
                   If write_dir and fullpath are None then no file output.
                   If only fullpath is None then it is the write dir + a name 
                   depending on the method.
    Note
    ----
    Ward's method takes time (about 6 minutes for a 60K voxels dataset)
    Geodesic k-means is 'quick and dirty'
    Ward's + GKM is expensive but quite good
    To reduce CPU time, rather use nn=6 (especially with Ward)    
    """
    import nipy.neurospin.graph as fg
    import nipy.neurospin.graph.field as ff
    
    if method not in ['ward','gkm','ward_and_gkm','kmeans']:
        raise ValueError, 'unknown method'
    if nn not in [6,18,26]:
        raise ValueError, 'nn should be 6,18 or 26'
    nbeta = len(betas)
    
    # step 1: load the data ----------------------------
    #1.1 the mask image
    nim = load(MaskImage)
    ref_dim =  nim.get_shape()
    affine = nim.get_affine()
    mask = nim.get_data()
    xyz = np.array(np.where(mask>0)).T
    nvox = xyz.shape[0]

    if method is not 'kmeans':
        # 1.2 get the main cc of the graph 
        # to remove the small connected components
        g = fg.WeightedGraph(nvox)
        g.from_3d_grid(xyz.astype(np.int),nn)
        
        aux = np.zeros(g.V).astype('bool')
        imc = g.main_cc()
        aux[imc]= True
        if np.sum(aux)==0:
            raise ValueError, "empty mask. Cannot proceed"
        g = g.subgraph(aux)
        lmask = np.zeros(ref_dim)
        lmask[xyz[:,0],xyz[:,1],xyz[:,2]]=aux
        xyz = xyz[aux,:]
        nvox = xyz.shape[0]
    else:
        lmask = mask

    # 1.3 from vox to mm
    xyz2 = np.hstack((xyz,np.ones((nvox,1))))
    coord = np.dot(xyz2, affine.T)[:,:3]

    # 1.4 read the functional data
    beta = []
    for b in range(nbeta):
        rbeta = load(betas[b])
        lbeta = rbeta.get_data()
        lbeta = lbeta[lmask>0]
        beta.append(lbeta)

    beta = np.array(beta)
    if len(beta.shape)>2:
        beta = np.squeeze(beta)
        
    if beta.shape[0]!=nvox:
        beta = beta.T


    #step 2: parcel the data ---------------------------
    feature = np.hstack((beta, mu*coord/np.std(coord)))
    if method is not 'kmeans':
        g = ff.Field(nvox, g.edges, g.weights, feature)

    if method=='kmeans':
        cent, u, J = kmeans(feature, nbparcel)

    if method=='ward':
        u, J0 = g.ward(nbparcel)

    if method=='gkm':
        seeds = np.argsort(np.random.rand(g.V))[:nbparcel]
        seeds, u, J1 = g.geodesic_kmeans(seeds)

    if method=='ward_and_gkm':
        w,J0 = g.ward(nbparcel)
        seeds, u, J1 = g.geodesic_kmeans(label=w)

    lpa = Parcellation(nbparcel, xyz, np.reshape(u,(nvox,1)))
    if verbose:
        pi = np.reshape(lpa.population(), nbparcel)
        vi = np.sum(lpa.var_feature_intra([beta])[0], 1)
        vf = np.dot(pi,vi)/nvox
        va =  np.dot(pi,np.sum(lpa.var_feature_intra([coord])[0],1))/nvox
        print nbparcel, "functional variance", vf, "anatomical variance",va


    # step3:  write the resulting label image
    Label = -np.ones(ref_dim,'int16')
    Label[lmask>0] = u

    if fullpath is not None:
        LabelImage = fullpath
    elif write_dir is not None:
        if method=='kmeans':
            LabelImage = os.path.join(write_dir,"parcel_kmeans.nii")
        if method=='ward':
            LabelImage = os.path.join(write_dir,"parcel_wards.nii")
        elif method=='gkm':
            LabelImage = os.path.join(write_dir,"parcel_gkmeans.nii")
        elif method=='ward_and_gkm':
            LabelImage = os.path.join(write_dir,"parcel_wgkmeans.nii")
    else:
        LabelImage = None
    
    if LabelImage is not None:
        wim = Nifti1Image(Label, affine)
        hdr = wim.get_header()
        hdr['descrip'] = 'Intra-subject parcellation image'
        save(wim, LabelImage)
        print "Wrote the parcellation images as %s" %LabelImage

    return lpa, Label
