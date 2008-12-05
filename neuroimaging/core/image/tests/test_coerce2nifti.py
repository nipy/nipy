import numpy as np
from neuroimaging.core import api
from neuroimaging.core.image.image import coerce2nifti
from neuroimaging.core.reference.nifti import coerce_coordmap


def setup_cmap():
    shape = (64,64,30,191)
    output_axes = [api.RegularAxis(s, step=i+1) for i, s in enumerate('xyzt')]
    output_coords = api.DiagonalCoordinateSystem('output', output_axes[:4])

    input_axes = [api.VoxelAxis(s, length=shape[i]) for i, s in enumerate('ijkl')]
    input_coords = api.VoxelCoordinateSystem('input', input_axes)

    cmap = api.CoordinateMap(api.Affine(output_coords.affine), input_coords, output_coords)
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
    cmap = cmap.reorder_input(lorder)
    img = api.Image(np.zeros(shape), cmap)
    newimg = coerce2nifti(img)
    assert img.coordmap.input_coords.axisnames == [newimg.coordmap.input_coords.axisnames[i] for i in lorder]
    assert newimg.coordmap.output_coords.axisnames == img.coordmap.output_coords.axisnames
    assert shape == tuple([newimg.shape[i] for i in lorder])
    assert tuple([np.asarray(newimg).shape[i] for i in lorder]) == shape

def test_coerce3():
    """
    this example has to be coerced, which means that there will be a 
    warning about non-diagonal and pixdims not agreeing
    """

    lorder = [0,2,3,1]
    shape = (64,64,191,30)
    shape, cmap = setup_cmap()
    cmap = cmap.reorder_output(lorder)
    img = api.Image(np.zeros(shape), cmap)
    newimg = coerce2nifti(img)
    assert img.coordmap.output_coords.axisnames == [newimg.coordmap.output_coords.axisnames[i] for i in lorder]
    assert newimg.coordmap.input_coords.axisnames == img.coordmap.input_coords.axisnames
    assert shape == newimg.shape
    assert np.asarray(newimg).shape == shape



