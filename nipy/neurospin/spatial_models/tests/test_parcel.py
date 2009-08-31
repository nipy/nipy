import numpy as np
import nipy.neurospin.spatial_models.hierarchical_parcellation as hp
import nipy.neurospin.utils.simul_2d_multisubject_fmri_dataset as simul
import nipy.neurospin.spatial_models.parcellation as fp

def make_data_field():
    nsubj = 1
    dimx = 60
    dimy = 60
    pos = 3*np.array([[ 6,  7],
                      [10, 10],
                      [15, 10]])
    ampli = np.array([5, 7, 6])
    sjitter = 6.0
    dataset = simul.make_surrogate_array(nbsubj=nsubj, dimx=dimx,
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
    feature = np.hstack((beta,mu*coord/np.std(anat_coord)))
    g = fg.WeightedGraph(nvox)
    g.from_3d_grid(xyz.astype(np.int),nn)
    g = ff.Field(nvox,g.edges,g.weights,feature)
    return g

def test_parcel_one_subj_1():
    nbparcel = 10
    g = make_data_field()
    u,J0 = g.ward(nbparcel)
    assert(np.unique(u)==np.arange(nbparcel))
    

def test_parcel_one_subj_2():
    nbparcel = 10
    g = make_data_field()
    seeds = np.argsort(np.random.rand(g.V))[:nbparcel]
    seeds, u, J1 = g.geodesic_kmeans(seeds)
    assert(np.unique(u)==np.arange(nbparcel))


def test_parcel_one_subj_3():
    nbparcel = 10
    g = make_data_field()
    w,J0 = g.ward(nbparcel)
    seeds, u, J1 = g.geodesic_kmeans(label=w)
    assert(np.unique(u)==np.arange(nbparcel))

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
    dataset = simul.make_surrogate_array(nbsubj=nsubj, dimx=dimx,
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
    for s in range(nbsubj):
        control *= (np.unique(Label)==np.arange(nbparcel))
    assert(control)
