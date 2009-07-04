from nipy.testing import *

import numpy as np
from nipy.io.api import load_image
from nipy.core.image import lpi_image
from nipy.core.api import Image
from nipy.core.reference.coordinate_map import AffineTransform, CoordinateSystem


fmri_file = 'filtered_func_data.img'

def generate_im():
    data = np.random.standard_normal((13,17,19,5))
    affine = np.array([[4,0,0,3],
                       [0,3.4,0,1],
                       [0,0,1.3,2],
                       [0,0,0,1]])

    full_affine = np.array([[4,0,0,0,3],
                            [0,3.4,0,0,1],
                            [0,0,1.3,0,2],
                            [0,0,0,2.5,0],
                            [0,0,0,0,1]])

    affine_mapping = AffineTransform(full_affine,
                                     CoordinateSystem('ijkl'),
                                     CoordinateSystem('xyzl'))
    lpi_im = lpi_image.LPIImage(data, affine, 'ijkl',
                                metadata={'abc':np.random.standard_normal(4)})
    im = Image(data, affine_mapping)
    return im, lpi_im


def test_reordered_axes():

    _, lpi_im = generate_im()

    lpi_reordered = lpi_im.reordered_axes([2,0,1,3])
    yield (assert_equal, np.array(lpi_reordered), 
           np.transpose(np.array(lpi_im), [2,0,1,3]))

    lpi_reordered = lpi_im.reordered_axes('kijl')
    yield (assert_equal, np.array(lpi_reordered), 
           np.transpose(np.array(lpi_im), [2,0,1,3]))

    yield assert_equal, lpi_im.metadata, lpi_reordered.metadata
    yield assert_equal, lpi_im.metadata, lpi_reordered.metadata

    yield assert_raises, ValueError, lpi_im.reordered_axes, [3,0,1,2]

def test_lpi_image_fmri():


    im, a = generate_im()

    A = np.identity(4)
    A[:3,:3] = im.affine[:3,:3]
    A[:3,-1] = im.affine[:3,-1]

    yield assert_almost_equal, A, a.affine

    # Now, change the order of the axes and create a new LPIImage
    # that is not-diagonal

    cm = im.coordmap
    cm_reordered = cm.reordered_domain(['j','k','i', 'l'])
    transposed_data = np.transpose(np.array(im), [1,2,0,3])

    im_reordered = Image(transposed_data, cm_reordered)
    B = np.identity(4)
    B[:3,:3] = im_reordered.affine[:3,:3]
    B[:3,-1] = im_reordered.affine[:3,-1]

    a2=lpi_image.LPIImage(np.array(im_reordered), B,
                                  im_reordered.coordmap.function_domain.coord_names)
    # Now, reorder it

    a3 = a2.xyz_ordered()

    yield assert_almost_equal, a3.affine, a.affine

    # as a subclass of Image, it still has a coordmap
    # describing ALL its axes


    yield assert_equal, a.coordmap.function_domain.coord_names , ('i', 'j', 'k', 'l')
    
    yield assert_equal, a.coordmap.function_range.coord_names , ('x', 'y', 'z', 'l')

    yield assert_equal, a2.coordmap.function_domain.coord_names , ('j', 'k', 'i', 'l')
    
    yield assert_equal, a2.coordmap.function_range.coord_names , ('x', 'y', 'z', 'l')

    yield assert_equal, a3.coordmap.function_domain.coord_names , ('i', 'j', 'k', 'l')
    
    yield assert_equal, a3.coordmap.function_range.coord_names , ('x', 'y', 'z', 'l')

    # But it lpi_transform is ony a 3d coordmap

    yield assert_equal, a.lpi_transform.function_domain.coord_names , ('i', 'j', 'k')
    
    yield assert_equal, a.lpi_transform.function_range.coord_names , ('x', 'y', 'z')

    yield assert_equal, a2.lpi_transform.function_domain.coord_names , ('j', 'k', 'i')
    
    yield assert_equal, a2.lpi_transform.function_range.coord_names , ('x', 'y', 'z')

    yield assert_equal, a3.lpi_transform.function_domain.coord_names , ('i', 'j', 'k')
    
    yield assert_equal, a3.lpi_transform.function_range.coord_names , ('x', 'y', 'z')



def test_resample():
    # XXX as written in the proposal, I don't think these will work for an LPIImage 
    # with data.ndim == 4 or should they?

    im, lpi_im = generate_im()

    lpi_im_resampled = lpi_im.resampled_to_affine(lpi_im.lpi_transform)
    #yield assert_almost_equal, np.array(lpi_im_resampled), np.array(lpi_im)

    lpi_im_resampled2 = lpi_im.resampled_to_img(lpi_im)
    #yield assert_almost_equal, np.array(lpi_im_resampled2), np.array(lpi_im)


def test_values_in_world():
    # XXX this shouldn't work for an LPIImage with data.ndim == 4 in the proposal for LPIImage, should they?

    im, lpi_im = generate_im()

    xyz_vals = lpi_im.lpi_transform(np.array([[3,4,5],
                                                    [4,7,8]]))
    x = xyz_vals[:,0]
    y = xyz_vals[:,1]
    z = xyz_vals[:,2]

    v1, v2 = lpi_im.values_in_world(x,y,z)
    yield assert_almost_equal, v1, np.array(lpi_im)[3,4,5]
    yield assert_equal, v1.shape, (lpi_im.shape[3],)
    yield assert_almost_equal, v2, np.array(lpi_im)[4,7,8]


