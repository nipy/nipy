# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import numpy as np
import nipy.neurospin.spatial_models.hierarchical_parcellation as hp
import nipy.neurospin.utils.simul_multisubject_fmri_dataset as simul
import nipy.neurospin.spatial_models.parcellation as fp
import nipy.neurospin.graph.graph as fg
import nipy.neurospin.graph.field as ff
import nipy.neurospin.clustering.clustering as fc

#####################################################################
# New part
#####################################################################

import nipy.neurospin.spatial_models.discrete_domain as dom
from nipy.neurospin.graph.field import field_from_coo_matrix_and_data

def test_parcel_interface():
    """ Simply test parcellation interface
    """
    # prepare some data
    shape = (10, 10, 10)
    nb_parcel = 10
    data =  np.random.randn(np.prod(shape))
    domain =  dom.grid_domain_from_array(np.ones(shape))
    g = field_from_coo_matrix_and_data(domain.topology, data)
    u, J0 = g.ward(nb_parcel)
    tmp = np.array([np.sum(u == k) for k in range(nb_parcel)])

    #instantiate a parcellation
    msp = fp.MultiSubjectParcellation(domain, u, u) 
    assert msp.nb_parcel == nb_parcel
    assert msp.nb_subj == 1
    assert (msp.population().ravel() == tmp).all()

def test_parcel_interface_multi_subj():
    """ test parcellation interface, with multiple subjects
    """
    # prepare some data
    shape = (10, 10, 10)
    nb_parcel = 10
    nb_subj = 5
    v = []
    for s in range(nb_subj):
        data =  np.random.randn(np.prod(shape))
        domain =  dom.grid_domain_from_array(np.ones(shape))
        g = field_from_coo_matrix_and_data(domain.topology, data)
        u, J0 = g.ward(nb_parcel)
        v.append(u)

    v = np.array(v).T    
    tmp = np.array([np.sum(v == k, 0) for k in range(nb_parcel)])

    #instantiate a parcellation
    msp = fp.MultiSubjectParcellation(domain, u, v) 
    assert msp.nb_parcel == nb_parcel
    assert msp.nb_subj == nb_subj
    assert (msp.population() == tmp).all()

def test_parcel_feature():
    """ Simply test parcellation feature interface
    """
    # prepare some data
    shape = (10, 10, 10)
    nb_parcel = 10
    data =  np.random.randn(np.prod(shape), 1)
    domain =  dom.grid_domain_from_array(np.ones(shape))
    g = field_from_coo_matrix_and_data(domain.topology, data)
    u, J0 = g.ward(nb_parcel)

    #instantiate a parcellation
    msp = fp.MultiSubjectParcellation(domain, u, u) 
    msp.make_feature('data', data)
    assert msp.get_feature('data').shape== (nb_parcel, 1)
    
    # test with a copy
    msp2 = msp.copy()
    assert (msp2.get_feature('data') == msp2.get_feature('data')).all()

    # test a multi_dimensional feature
    dim = 4
    msp.make_feature('new', np.random.randn(np.prod(shape), 1, dim))
    assert msp.get_feature('new').shape== (nb_parcel, 1, dim)

def test_parcel_feature_multi_subj():
    """ Test parcellation feature interface with multiple subjects
    """
    # prepare some data
    shape = (10, 10, 10)
    nb_parcel = 10
    nb_subj = 5
    v = []
    for s in range(nb_subj):
        data =  np.random.randn(np.prod(shape))
        domain =  dom.grid_domain_from_array(np.ones(shape))
        g = field_from_coo_matrix_and_data(domain.topology, data)
        u, J0 = g.ward(nb_parcel)
        v.append(u)

    v = np.array(v).T    
    msp = fp.MultiSubjectParcellation(domain, u, v) 

    # test a multi_dimensional feature
    # dimension 1
    msp.make_feature('data', np.random.randn(np.prod(shape), nb_subj))
    assert msp.get_feature('data').shape== (nb_parcel, nb_subj)

    #dimension>1
    dim = 4    
    msp.make_feature('data', np.random.randn(np.prod(shape), nb_subj, dim))
    assert msp.get_feature('data').shape== (nb_parcel, nb_subj, dim)
    
    # msp.features['data'] has been overriden
    assert msp.features.keys() == ['data']

#####################################################################
# Deprecated part
#####################################################################

def make_data_field():
    nsubj = 1
    dimx = 60
    dimy = 60
    pos = 3*np.array([[ 6,  7],
                      [10, 10],
                      [15, 10]])
    ampli = np.array([5, 7, 6])
    sjitter = 6.0
    dataset = simul.surrogate_2d_dataset(nbsubj=nsubj, dimx=dimx,
                                         dimy=dimy, 
                                         pos=pos, ampli=ampli, width=10.0)

    # step 2 : prepare all the information for the parcellation
    nbparcel = 10
    ref_dim = (dimx,dimy)
    xy = np.array(np.where(dataset[0])).T
    nvox = np.size(xy,0)
    xyz = np.hstack((xy,np.zeros((nvox,1))))
	
    ldata = np.reshape(dataset,(dimx*dimy,1))
    anat_coord = xy
    mu = 10.
    nn = 18
    feature = np.hstack((ldata/np.std(ldata),
                         mu*anat_coord/np.std(anat_coord)))
    g = fg.WeightedGraph(nvox)
    g.from_3d_grid(xyz.astype(np.int),nn)
    g = ff.Field(nvox, g.edges, g.weights, feature)
    return g

def test_parcel_one_subj_1():
    nbparcel = 10
    g = make_data_field()
    u, J0 = g.ward(nbparcel)
    assert((np.unique(u) == np.arange(nbparcel)).all())
    

def test_parcel_one_subj_2():
    nbparcel = 10
    g = make_data_field()
    seeds = np.argsort(np.random.rand(g.V))[:nbparcel]
    seeds, u, J1 = g.geodesic_kmeans(seeds)
    assert((np.unique(u) == np.arange(nbparcel)).all())


def test_parcel_one_subj_3():
    nbparcel = 10
    g = make_data_field()
    w, J0 = g.ward(nbparcel)
    seeds, u, J1 = g.geodesic_kmeans(label=w)
    assert((np.unique(u) == np.arange(nbparcel)).all())

def test_parcel_one_subj_4():
    nbparcel = 10
    g = make_data_field()
    _, u, _ = fc.kmeans(g.field, nbparcel)
    assert((np.unique(u) == np.arange(nbparcel)).all())


def test_parcel_multi_subj():
    """
    """
    # step 1:  generate some synthetic data
    nsubj = 10
    dimx = 60
    dimy = 60
    pos = 3*np.array([[ 6,  7],
                      [10, 10],
                      [15, 10]])
    ampli = np.array([5, 7, 6])
    sjitter = 6.0
    dataset = simul.surrogate_2d_dataset(nbsubj=nsubj, dimx=dimx,
                                         dimy=dimy, 
                                         pos=pos, ampli=ampli, width=10.0)

    # step 2 : prepare all the information for the parcellation
    nbparcel = 10
    ref_dim = (dimx,dimy)
    xy = np.array(np.where(dataset[0])).T
    nvox = np.size(xy,0)
    xyz = np.hstack((xy,np.zeros((nvox,1))))
	
    ldata = np.reshape(dataset,(nsubj,dimx*dimy,1))
    anat_coord = xy
    mask = np.ones((nvox,nsubj)).astype('bool')
    Pa = fp.Parcellation(nbparcel,xyz,mask-1)

    # step 3 : run the algorithm
    Pa =  hp.hparcel(Pa, ldata, anat_coord, mu = 10.0)
    	
    # step 4:  look at the results
    Label =  np.array([np.reshape(Pa.label[:,s],(dimx,dimy))
                       for s in range(nsubj)])
    control = True
    for s in range(nsubj):
        control *= (np.unique(Label)==np.arange(nbparcel)).all()
    assert(control)

if __name__ == '__main__':
    import nose
    nose.run(argv=['', __file__])
