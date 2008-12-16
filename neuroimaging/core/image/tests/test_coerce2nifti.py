import numpy as np
from neuroimaging.core import api
from neuroimaging.core.image.image import coerce2nifti
from neuroimaging.core.reference.nifti import coerce_coordmap
from neuroimaging.core.reference.coordinate_map import reorder_input, reorder_output

def setup_cmap():
    shape = (64,64,30,191)
    cmap = api.Affine.from_start_step('ijkl', 'xyzt', [0]*4, np.arange(4))
    return shape, cmap

def test_coerce():
    """
    this test just creates an image that is already ready for nifti output
    """
    shape, cmap = setup_cmap()
    img = api.Image(np.zeros(shape), cmap)
    newimg = coerce2nifti(img)
    assert newimg.coordmap.input_coords.axisnames == img.coordmap.input_coords.axisnames 
    assert newimg.coordmap.output_coords.axisnames == img.coordmap.output_coords.axisnames
    assert newimg.shape == shape
    assert np.allclose(newimg.affine, img.affine)
    assert np.asarray(newimg).shape == shape

def test_coerce2():
    """
    this example has to be coerced, which means that there will be a 
    warning about non-diagonal and pixdims not agreeing
    """

    lorder = [0,2,3,1]
    shape, cmap = setup_cmap()
    cmap = reorder_input(cmap, lorder)
    img = api.Image(np.zeros(tuple([shape[i] for i in lorder])), cmap)
    newimg = coerce2nifti(img)
    assert img.coordmap.input_coords.axisnames == [newimg.coordmap.input_coords.axisnames[i] for i in lorder]
    assert newimg.coordmap.output_coords.axisnames == img.coordmap.output_coords.axisnames
    assert shape == newimg.shape
    assert np.asarray(newimg).shape == shape

def test_coerce3():
    """
    this example has to be coerced, which means that there will be a 
    warning about non-diagonal and pixdims not agreeing
    """

    lorder = [0,2,3,1]
    shape = (64,64,191,30)
    shape, cmap = setup_cmap()
    cmap = reorder_output(cmap, lorder)
    img = api.Image(np.zeros(shape), cmap)
    newimg = coerce2nifti(img)
    assert img.coordmap.output_coords.axisnames == [newimg.coordmap.output_coords.axisnames[i] for i in lorder]
    assert newimg.coordmap.input_coords.axisnames == img.coordmap.input_coords.axisnames
    assert shape == newimg.shape
    assert np.asarray(newimg).shape == shape



