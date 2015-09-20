# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

from nibabel import Nifti1Image
from ..utils.simul_multisubject_fmri_dataset import \
    surrogate_2d_dataset
from ..statistical_mapping import cluster_stats

def make_surrogate_data():
    """ Return a single deterministic 3D image 
    """
    shape = (40, 40)
    pos   = np.array([[ 2, 10],
                      [10,  4],
                      [20, 30],
                      [30, 20]])
    ampli = np.array([5, 5, 5, 5])
    data = surrogate_2d_dataset(n_subj=1, pos=pos, shape=shape, noise_level=0,
                                ampli=ampli, spatial_jitter=0,
                                signal_jitter=0).squeeze()
    data = np.reshape(data, (shape[0], shape[1], 1))
    return Nifti1Image(data, np.eye(4))
    
def test1():
    img = make_surrogate_data()
    clusters, info = cluster_stats(img, img, height_th=3., height_control='None', cluster_th=0, nulls={})
    assert len(clusters)==4
    
def test2():
    img = make_surrogate_data()
    clusters, info = cluster_stats(img, img, height_th=3., height_control='None', cluster_th=5, nulls={})
    assert len(clusters)==4

def test3():
    img = make_surrogate_data()
    clusters, info = cluster_stats(img, img, height_th=3., height_control='None', cluster_th=10, nulls={})
    assert len(clusters)==0


def test_4():
    img = make_surrogate_data()
    clusters, info = cluster_stats(img, img, height_th=.001, height_control='fpr', cluster_th=0, nulls={})
    assert len(clusters)==4
    
def test_5():
    img = make_surrogate_data()
    clusters, info = cluster_stats(img, img, height_th=.05, height_control='bonferroni', cluster_th=0, nulls={})
    assert len(clusters)==4

def test_6():
    img = make_surrogate_data()
    clusters, info = cluster_stats(img, img, height_th=.05, 
                                   height_control='fdr', cluster_th=0, nulls={})
    print(len(clusters), sum([c['size'] for c in clusters]))
    assert len(clusters)==4

def test7():
    img = make_surrogate_data()
    clusters, info = cluster_stats(img, img, height_th=3., height_control='None', cluster_th=0, nulls={})
    nstv = sum([c['size'] for c in clusters])
    assert nstv==36

def test_8():
    img = make_surrogate_data()
    clusters, info = cluster_stats(img, img, height_th=.001, height_control='fpr', cluster_th=0, nulls={})
    nstv = sum([c['size'] for c in clusters])
    assert nstv==36
 
if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])

