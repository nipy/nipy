"""
This scipt generates a noisy activation image image
and applies the bayesian structural analysis on it

Author : Bertrand Thirion, 2009
"""
#autoindent

import numpy as np
import scipy.stats as st
import matplotlib.pylab as mp
import nipy.neurospin.graph.field as ff
import nipy.neurospin.utils.simul_2d_multisubject_fmri_dataset as simul
import nipy.neurospin.spatial_models.bayesian_structural_analysis as bsa
import profile


def make_bsa_2d(betas, theta=3., dmax=5., ths=0, thq=0.5, smin=0, 
                        nbeta=[0],method='simple',verbose = 0):
    """
    Function for performing bayesian structural analysis
    on a set of images.
    """
    ref_dim = np.shape(betas[0])
    nsubj = betas.shape[0]
    xyz = np.array(np.where(betas[:1])).T.astype(np.int)
    nvox = np.size(xyz, 0)
    
    # create the field strcture that encodes image topology
    Fbeta = ff.Field(nvox)
    Fbeta.from_3d_grid(xyz, 18)

    # Get  coordinates in mm
    coord = xyz.astype(np.float)

    # get the functional information
    lbeta = np.array([np.ravel(betas[k]) for k in range(nsubj)]).T

    # the voxel volume is 1.0
    g0 = 1.0/(1.0*nvox)*1./np.sqrt(2*np.pi*dmax**2)
    affine = np.eye(4)
    shape = (1, ref_dim[0], ref_dim[1])
    lmax=0
    bdensity = 1
    if method=='ipmi':
        group_map, AF, BF, likelihood = \
                   bsa.compute_BSA_ipmi(Fbeta, lbeta, coord, dmax, xyz,
                                        affine, shape, thq,
                                        smin, ths, theta, g0, bdensity)
    if method=='simple':
        group_map, AF, BF, likelihood = \
                   bsa.compute_BSA_simple(Fbeta, lbeta, coord, dmax,xyz,
                                          affine, shape, thq, smin, ths,
                                          theta, g0)
    if method=='loo':
         group_map, AF, BF, likelihood = \
                   bsa.compute_BSA_loo(Fbeta, lbeta, coord, dmax,xyz,
                                          affine, shape, thq, smin, ths,
                                          theta, g0)
    if method=='dev':
        group_map, AF, BF, likelihood = \
                   bsa.compute_BSA_dev(Fbeta, lbeta, coord, dmax,xyz,
                                       affine, shape, thq,
                                      smin, ths, theta, g0, bdensity)
    if method=='simple2':
        likelihood = np.zeros(ref_dim)
        group_map, AF, BF, coclustering = \
                   bsa.compute_BSA_simple2(Fbeta, lbeta, coord, dmax,xyz,
                                          affine, shape, thq, smin, ths,
                                          theta, g0)
        
    if method not in['loo', 'dev','simple','ipmi','simple2']:
        raise ValueError,'method is not ocrreactly defined'
    
    if verbose==0:
        return AF,BF
    
    if AF != None:
        lmax = AF.k+2
        AF.show()

    group_map.shape = ref_dim
    mp.figure()
    mp.subplot(1,3,1)
    mp.imshow(group_map, interpolation='nearest', vmin=-1, vmax=lmax)
    mp.title('Blob separation map')
    mp.colorbar()

    if AF != None:
        group_map = AF.map_label(coord,0.95,dmax)
        group_map.shape = ref_dim
    
    mp.subplot(1,3,2)
    mp.imshow(group_map, interpolation='nearest', vmin=-1, vmax=lmax)
    mp.title('group-level position 95% \n confidence regions')
    mp.colorbar()

    mp.subplot(1,3,3)
    likelihood.shape = ref_dim
    mp.imshow(likelihood, interpolation='nearest')
    mp.title('Spatial density under h1')
    mp.colorbar()

    
    mp.figure()
    if nsubj==10:
        for s in range(nsubj):
            mp.subplot(2, 5, s+1)
            lw = -np.ones(ref_dim)
            if BF[s]!=None:
                nls = BF[s].get_roi_feature('label')
                nls[nls==-1] = np.size(AF)+2
                for k in range(BF[s].k):
                    xyzk = BF[s].xyz[k].T 
                    lw[xyzk[1],xyzk[2]] =  nls[k]

            mp.imshow(lw, interpolation='nearest', vmin=-1, vmax=lmax)
            mp.axis('off')

    mp.figure()
    if nsubj==10:
        for s in range(nsubj):
            mp.subplot(2,5,s+1)
            mp.imshow(betas[s],interpolation='nearest',vmin=betas.min(),
                      vmax=betas.max())
            mp.axis('off')

    return AF, BF


# ------------------------------------------------------
#------- Main script -----------------------------------
#-------------------------------------------------------

# generate the data
nsubj = 10
dimx = 60
dimy = 60
pos = 2*np.array([[ 6,  7],
                  [10, 10],
                  [15, 10]])
ampli = np.array([5, 7, 6])
sjitter = 1.0
dataset = simul.make_surrogate_array(nbsubj=nsubj, dimx=dimx, dimy=dimy, 
                                     pos=pos, ampli=ampli, width=5.0)
betas = np.reshape(dataset, (nsubj, dimx, dimy))

# set various parameters
theta = float(st.t.isf(0.01, 100))
dmax = 5./1.5
ths = 1#nsubj/2
thq = 0.9
verbose = 1
smin = 5
method = 'simple2'#'loo'#'dev'#'ipmi'#

# run the algo
AF, BF = make_bsa_2d(betas, theta, dmax, ths, thq, smin,method=method,verbose=verbose)
AF, BF = make_bsa_2d(betas, theta, dmax, ths, thq, smin,method='simple',verbose=verbose)


mp.show()
