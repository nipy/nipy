"""
This scipt generates a noisy activation image image
and applies the bayesian structural analysis on it

Author : Bertrand Thirion, 2009
"""
#autoindent

import numpy as np
import scipy.stats as st
import nipy.neurospin.graph.field as ff
import nipy.neurospin.utils.simul_2d_multisubject_fmri_dataset as simul
import nipy.neurospin.spatial_models.bayesian_structural_analysis as bsa
import nipy.neurospin.spatial_models.structural_bfls as sbf

def make_bsa_2d(betas, theta=3., dmax=5., ths=0, thq=0.5, smin=0, 
                        nbeta=[0],method='simple'):
    """ Function for performing bayesian structural analysis on a set of images.
    """
    ref_dim = np.shape(betas[0])
    nbsubj = betas.shape[0]
    xyz = np.array(np.where(betas[:1]))
    nbvox = np.size(xyz, 1)
    
    # create the field strcture that encodes image topology
    Fbeta = ff.Field(nbvox)
    Fbeta.from_3d_grid(xyz.astype(np.int).T, 18)

    # Get  coordinates in mm
    xyz = np.transpose(xyz)
    tal = xyz.astype(np.float)

    # get the functional information
    lbeta = np.array([np.ravel(betas[k]) for k in range(nbsubj)]).T

    # the voxel volume is 1.0
    g0 = 1.0/(1.0*nbvox)
    bdensity = 1

    if method=='ipmi':
        group_map, AF, BF, likelihood = \
                   bsa.compute_BSA_ipmi(Fbeta, lbeta, tal, dmax,xyz, None, thq,
                                        smin, ths, theta, g0, bdensity)
    if method=='simple':
        group_map, AF, BF, likelihood = \
                   bsa.compute_BSA_simple(Fbeta, lbeta, tal, dmax,xyz,
                                          None, thq, smin, ths, theta, g0)
    if method=='dev':
        group_map, AF, BF, likelihood = \
                   bsa.compute_BSA_dev(Fbeta, lbeta, tal, dmax,xyz, None, thq,
                                       smin, ths, theta, g0, bdensity)
    if method=='sbf':
        pval = 0.2
        group_map, AF, BF = sbf.Compute_Amers (Fbeta,lbeta,xyz,None,
                                               tal,dmax, theta, ths ,pval)
    return AF, BF


    
def test_bsa_null_simple():
    # generate the data
    nbsubj=10
    
    dimx=60
    dimy=60
    pos = 2*np.array([[ 6,  7],
                      [10, 10],
                      [15, 10]])
    ampli = np.array([0, 0, 0])
    sjitter = 1.0
    dataset = simul.make_surrogate_array(nbsubj=nbsubj, dimx=dimx, dimy=dimy, 
                                         pos=pos, ampli=ampli, width=5.0)
    betas = np.reshape(dataset, (nbsubj, dimx, dimy))
    
    # set various parameters
    theta = float(st.t.isf(0.01, 100))
    dmax = 5./1.5
    ths = nbsubj/2
    thq = 0.9
    verbose = 1
    smin = 5

    # run the algo
    AF, BF = make_bsa_2d(betas, theta, dmax, ths, thq, smin,method='simple')

    #make sure that nothing is detected
    assert(AF==None)

def test_bsa_null_dev():
    # generate the data
    nbsubj=10
    
    dimx=60
    dimy=60
    pos = 2*np.array([[ 6,  7],
                      [10, 10],
                      [15, 10]])
    ampli = np.array([0, 0, 0])
    sjitter = 1.0
    dataset = simul.make_surrogate_array(nbsubj=nbsubj, dimx=dimx, dimy=dimy, 
                                         pos=pos, ampli=ampli, width=5.0)
    betas = np.reshape(dataset, (nbsubj, dimx, dimy))
    
    # set various parameters
    theta = float(st.t.isf(0.01, 100))
    dmax = 5./1.5
    ths = nbsubj/2
    thq = 0.9
    verbose = 1
    smin = 5

    # run the algo
    AF, BF = make_bsa_2d(betas, theta, dmax, ths, thq, smin,method='dev')

    #make sure that nothing is detected
    assert(AF==None)

def test_bsa_null_ipmi():
    # generate a null dataset
    nbsubj=10
    
    dimx=60
    dimy=60
    pos = 2*np.array([[ 6,  7],
                      [10, 10],
                      [15, 10]])
    ampli = np.array([0, 0, 0])
    sjitter = 1.0
    dataset = simul.make_surrogate_array(nbsubj=nbsubj, dimx=dimx, dimy=dimy, 
                                         pos=pos, ampli=ampli, width=5.0)
    betas = np.reshape(dataset, (nbsubj, dimx, dimy))
    
    # set various parameters
    theta = float(st.t.isf(0.01, 100))
    dmax = 5./1.5
    ths = nbsubj/2
    thq = 0.9
    verbose = 1
    smin = 5

    # run the algo
    AF, BF = make_bsa_2d(betas, theta, dmax, ths, thq, smin,method='ipmi')

    #make sure that nothing is detected
    assert(AF==None)

def test_bsa_simple():
    # generate the data
    nbsubj=10
    
    dimx=60
    dimy=60
    pos = 2*np.array([[ 6,  7],
                      [10, 10],
                      [15, 10]])
    ampli = np.array([5, 7, 6])
    sjitter = 1.0
    dataset = simul.make_surrogate_array(nbsubj=nbsubj, dimx=dimx, dimy=dimy, 
                                         pos=pos, ampli=ampli, width=5.0,
                                         seed=1)
    betas = np.reshape(dataset, (nbsubj, dimx, dimy))
    
    # set various parameters
    theta = float(st.t.isf(0.01, 100))
    dmax = 5./1.5
    ths = 1
    thq = 0.9
    verbose = 1
    smin = 5

    # run the algo
    AF, BF = make_bsa_2d(betas, theta, dmax, ths, thq, smin)

    #make sure that at least 1 spot is detected
    assert(AF.k>1)

def test_bsa_dev():
    # generate the data
    nbsubj=10
    
    dimx=60
    dimy=60
    pos = 2*np.array([[ 6,  7],
                      [10, 10],
                      [15, 10]])
    ampli = np.array([5, 7, 6])
    sjitter = 1.0
    dataset = simul.make_surrogate_array(nbsubj=nbsubj, dimx=dimx, dimy=dimy, 
                                         pos=pos, ampli=ampli, width=5.0,
                                         seed=1)
    betas = np.reshape(dataset, (nbsubj, dimx, dimy))
    
    # set various parameters
    theta = float(st.t.isf(0.01, 100))
    dmax = 5./1.5
    ths = 1
    thq = 0.9
    verbose = 1
    smin = 5

    # run the algo
    AF, BF = make_bsa_2d(betas, theta, dmax, ths, thq, smin,method='dev')

    #make sure that at least 1 spot is detected
    assert(AF.k>1)

def test_bsa_ipmi():
    # generate the data
    nbsubj=10
    
    dimx=60
    dimy=60
    pos = 2*np.array([[ 6,  7],
                      [10, 10],
                      [15, 10]])
    ampli = np.array([5, 7, 6])
    sjitter = 1.0
    dataset = simul.make_surrogate_array(nbsubj=nbsubj, dimx=dimx, dimy=dimy, 
                                         pos=pos, ampli=ampli, width=5.0,
                                         seed=1)
    betas = np.reshape(dataset, (nbsubj, dimx, dimy))
    
    # set various parameters
    theta = float(st.t.isf(0.01, 100))
    dmax = 5./1.5
    ths = 1
    thq = 0.9
    verbose = 1
    smin = 5

    # run the algo
    AF, BF = make_bsa_2d(betas, theta, dmax, ths, thq, smin,method='ipmi')

    #make sure that  at least 1 spot is detected
    assert(AF.k>1)



if __name__ == '__main__':
    import nose
    nose.run(argv=['', __file__])
