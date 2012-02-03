# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import numpy as np

from nipy.algorithms.graph.field import field_from_coo_matrix_and_data
from ..hierarchical_parcellation import hparcel
from ...utils.simul_multisubject_fmri_dataset import surrogate_2d_dataset
from ..parcellation import MultiSubjectParcellation
from ..discrete_domain import grid_domain_from_binary_array


def test_parcel_interface():
    """ Simply test parcellation interface
    """
    # prepare some data
    shape = (5, 5, 5)
    nb_parcel = 10
    data = np.random.randn(np.prod(shape))
    domain = grid_domain_from_binary_array(np.ones(shape))
    g = field_from_coo_matrix_and_data(domain.topology, data)
    u, J0 = g.ward(nb_parcel)
    tmp = np.array([np.sum(u == k) for k in range(nb_parcel)])

    #instantiate a parcellation
    msp = MultiSubjectParcellation(domain, u, u)
    assert msp.nb_parcel == nb_parcel
    assert msp.nb_subj == 1
    assert (msp.population().ravel() == tmp).all()


def test_parcel_interface_multi_subj():
    """ test parcellation interface, with multiple subjects
    """
    # prepare some data
    shape = (5, 5, 5)
    nb_parcel = 10
    nb_subj = 5
    v = []
    for s in range(nb_subj):
        data = np.random.randn(np.prod(shape))
        domain = grid_domain_from_binary_array(np.ones(shape))
        g = field_from_coo_matrix_and_data(domain.topology, data)
        u, J0 = g.ward(nb_parcel)
        v.append(u)

    v = np.array(v).T
    tmp = np.array([np.sum(v == k, 0) for k in range(nb_parcel)])

    #instantiate a parcellation
    msp = MultiSubjectParcellation(domain, u, v)
    assert msp.nb_parcel == nb_parcel
    assert msp.nb_subj == nb_subj
    assert (msp.population() == tmp).all()


def test_parcel_feature():
    """ Simply test parcellation feature interface
    """
    # prepare some data
    shape = (5, 5, 5)
    nb_parcel = 10
    data = np.random.randn(np.prod(shape), 1)
    domain = grid_domain_from_binary_array(np.ones(shape))
    g = field_from_coo_matrix_and_data(domain.topology, data)
    u, J0 = g.ward(nb_parcel)

    #instantiate a parcellation
    msp = MultiSubjectParcellation(domain, u, u)
    msp.make_feature('data', data)
    assert msp.get_feature('data').shape == (nb_parcel, 1)

    # test with a copy
    msp2 = msp.copy()
    assert (msp2.get_feature('data') == msp2.get_feature('data')).all()

    # test a multi_dimensional feature
    dim = 4
    msp.make_feature('new', np.random.randn(np.prod(shape), 1, dim))
    assert msp.get_feature('new').shape == (nb_parcel, 1, dim)


def test_parcel_feature_multi_subj():
    """ Test parcellation feature interface with multiple subjects
    """
    # prepare some data
    shape = (5, 5, 5)
    nb_parcel = 10
    nb_subj = 5
    v = []
    for s in range(nb_subj):
        data = np.random.randn(np.prod(shape))
        domain = grid_domain_from_binary_array(np.ones(shape))
        g = field_from_coo_matrix_and_data(domain.topology, data)
        u, J0 = g.ward(nb_parcel)
        v.append(u)

    v = np.array(v).T
    msp = MultiSubjectParcellation(domain, u, v)

    # test a multi_dimensional feature
    # dimension 1
    msp.make_feature('data', np.random.randn(np.prod(shape), nb_subj))
    assert msp.get_feature('data').shape == (nb_parcel, nb_subj)

    #dimension>1
    dim = 4
    msp.make_feature('data', np.random.randn(np.prod(shape), nb_subj, dim))
    assert msp.get_feature('data').shape == (nb_parcel, nb_subj, dim)

    # msp.features['data'] has been overriden
    assert msp.features.keys() == ['data']


def test_parcel_hierarchical():
    """Test the algorithm for hierrachical parcellation
    """
    # step 1:  generate some synthetic data
    n_subj = 10
    shape = (30, 30)
    dataset = surrogate_2d_dataset(n_subj=n_subj, shape=shape)

    # step 2 : prepare all the information for the parcellation
    nb_parcel = 10
    domain = grid_domain_from_binary_array(dataset[0] ** 2, np.eye(3))
    ldata = np.reshape(dataset, (n_subj, np.prod(shape), 1))

    # step 3 : run the algorithm
    Pa = hparcel(domain, ldata, nb_parcel)

    # step 4:  look at the results
    Label = Pa.individual_labels
    control = True
    for s in range(n_subj):
        control *= (np.unique(Label[:, s]) == np.arange(nb_parcel)).all()

    assert(control)


def test_prfx():
    """Test the ability to construct parcel features and random effects models
    """
    # step 1:  generate some synthetic data
    n_subj = 10
    shape = (30, 30)
    dataset = surrogate_2d_dataset(n_subj=n_subj, shape=shape)

    # step 2 : prepare all the information for the parcellation
    nb_parcel = 10
    domain = grid_domain_from_binary_array(dataset[0] ** 2, np.eye(3))
    ldata = np.reshape(dataset, (n_subj, np.prod(shape), 1))

    # step 3 : run the algorithm
    Pa = hparcel(domain, ldata, nb_parcel)
    pdata = Pa.make_feature('functional',
                            np.rollaxis(np.array(ldata), 1, 0))
    one_sample = np.squeeze(pdata.mean(0) / pdata.std(0))
    assert np.shape(one_sample) == tuple([nb_parcel])
    assert one_sample.mean() < 1
    assert one_sample.mean() > -1

if __name__ == '__main__':
    import nose
    nose.run(argv=['', __file__])
