import warnings
import numpy as np
from neuroimaging.testing import *

from neuroimaging.core import api
from neuroimaging.io.files import coerce2nifti
from neuroimaging.io.nifti_ref import coerce_coordmap
from neuroimaging.core.reference.coordinate_map import reorder_input, reorder_output

def setup():
    # Suppress warnings during tests to reduce noise
    warnings.simplefilter("ignore")

def teardown():
    # Clear list of warning filters
    warnings.resetwarnings()


def setup_cmap():
    shape = (64,64,30,191)
    cmap = api.Affine.from_start_step('ijkl', 'xyzt', [0]*4, np.arange(4))
    return shape, cmap

def test_coerce():
    #this test just creates an image that is already ready for nifti output

    shape, cmap = setup_cmap()
    img = api.Image(np.zeros(shape), cmap)
    newimg = coerce2nifti(img)
    yield assert_equal, newimg.coordmap.input_coords.coord_names, \
        img.coordmap.input_coords.coord_names 
    yield assert_equal, newimg.coordmap.output_coords.coord_names, \
        img.coordmap.output_coords.coord_names
    yield assert_equal, newimg.shape, shape
    yield assert_true, np.allclose(newimg.affine, img.affine)
    yield assert_equal, np.asarray(newimg).shape, shape

def test_coerce2():
    # this example has to be coerced, which means that there will be a
    # warning about non-diagonal and pixdims not agreeing

    lorder = [0,2,3,1]
    shape, cmap = setup_cmap()
    cmap = reorder_input(cmap, lorder)
    img = api.Image(np.zeros(tuple([shape[i] for i in lorder])), cmap)
    newimg = coerce2nifti(img)
    neworder = [newimg.coordmap.input_coords.coord_names[i] for i in lorder]
    yield assert_equal, neworder, img.coordmap.input_coords.coord_names
    yield assert_equal, newimg.coordmap.output_coords.coord_names, \
        img.coordmap.output_coords.coord_names
    yield assert_equal, shape, newimg.shape
    yield assert_equal, np.asarray(newimg).shape, shape

def test_coerce3():
    # this example has to be coerced, which means that there will be a
    # warning about non-diagonal and pixdims not agreeing


    lorder = [0,2,3,1]
    shape = (64,64,191,30)
    shape, cmap = setup_cmap()
    cmap = reorder_output(cmap, lorder)
    img = api.Image(np.zeros(shape), cmap)
    newimg = coerce2nifti(img)
    neworder = [newimg.coordmap.output_coords.coord_names[i] for i in lorder]
    yield assert_equal, neworder, img.coordmap.output_coords.coord_names
    yield assert_equal, newimg.coordmap.input_coords.coord_names, \
        img.coordmap.input_coords.coord_names
    yield assert_equal, shape, newimg.shape
    yield assert_equal, np.asarray(newimg).shape, shape



