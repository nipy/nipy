import numpy as np

from nipy.neurospin.utils.simul_2d_multisubject_fmri_dataset import \
    make_surrogate_array
from nipy.neurospin.utils.threshold import threshold_array, threshold_z_array

def make_surrogate_data():
    """ 
    """
    dimx = 40
    dimy = 40
    pos   = np.array([[ 2, 10],
                      [10,  4],
                      [20, 30],
                      [30, 20]])
    ampli = np.array([5,5,5,5])
    data = make_surrogate_array(nbsubj=1, pos=pos, dimx=40, noise_level=0,
                                dimy=40, ampli=ampli, spatial_jitter=0,
                                signal_jitter=0).squeeze()
    return data
    
def test1():
    x = make_surrogate_data()
    x = np.reshape(x,(x.shape[0],x.shape[1],1))
    thx = threshold_array(x,  th=3., smin=0, nn=18)
    assert np.sum(thx>0)==36

def test2():
    x = make_surrogate_data()
    x = np.reshape(x,(x.shape[0],x.shape[1],1))
    thx = threshold_array(x,  th=3., smin=5, nn=18)
    assert np.sum(thx>0)==36

def test3():
    x = make_surrogate_data()
    x = np.reshape(x,(x.shape[0],x.shape[1],1))
    thx = threshold_array(x,  th=3., smin=10, nn=18)
    assert np.sum(thx>0)==0

def test_4():
    x = make_surrogate_data()
    x = np.reshape(x,(x.shape[0],x.shape[1],1))
    thx = threshold_z_array(x, correction=None, pval=0.001, smin=0, nn=18)
    print np.sum(thx>0)
    assert np.sum(thx>0)==36

def test_5():
    x = make_surrogate_data()
    x = np.reshape(x,(x.shape[0],x.shape[1],1))
    thx = threshold_z_array(x, correction='bon', pval=0.05, smin=0, nn=18)
    print np.sum(thx>0)
    assert np.sum(thx>0)==4

def test_6():
    x = make_surrogate_data()
    x = np.reshape(x,(x.shape[0],x.shape[1],1))
    thx = threshold_z_array(x, correction='fdr', pval=0.05, smin=0, nn=18)
    print np.sum(thx>0)
    assert np.sum(thx>0)==20

def test_7():
    x = make_surrogate_data()
    x = np.reshape(x,(x.shape[0],x.shape[1],1))
    thx = threshold_z_array(x, correction='fdr', pval=0.05, smin=0, nn=18,
                            method='emp')
    print np.sum(thx>0)
    assert np.sum(thx>0)==84


if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])

