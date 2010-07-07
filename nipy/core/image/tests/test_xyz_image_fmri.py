# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from nipy.testing import *

import numpy as np
from nipy.io.api import load_image
from nipy.core.image import xyz_image
from nipy.core.api import Image
from nipy.core.reference.coordinate_map import AffineTransform, CoordinateSystem

fmri_file = 'filtered_func_data.img'

lps = xyz_image.lps_output_coordnames
ras = xyz_image.ras_output_coordnames

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

    affine_mapping = AffineTransform(CoordinateSystem('ijkl'),
                                     CoordinateSystem(lps + ('l',)),
                                     full_affine)
    xyz_im = xyz_image.XYZImage(data, affine, 'ijkl',
                                metadata={'abc':np.random.standard_normal(4)})
    im = Image(data, affine_mapping)
    return im, xyz_im


def test_reordered_axes():

    _, xyz_im = generate_im()

    xyz_reordered = xyz_im.reordered_axes([2,0,1,3])
    yield (assert_equal, np.array(xyz_reordered), 
           np.transpose(np.array(xyz_im), [2,0,1,3]))

    xyz_reordered = xyz_im.reordered_axes('kijl')
    yield (assert_equal, np.array(xyz_reordered), 
           np.transpose(np.array(xyz_im), [2,0,1,3]))

    yield assert_equal, xyz_im.metadata, xyz_reordered.metadata
    yield assert_equal, xyz_im.metadata, xyz_reordered.metadata

    yield assert_raises, ValueError, xyz_im.reordered_axes, [3,0,1,2]

def test_xyz_image_fmri():


    im, a = generate_im()

    A = np.identity(4)
    A[:3,:3] = im.affine[:3,:3]
    A[:3,-1] = im.affine[:3,-1]

    yield assert_almost_equal, A, a.affine

    # Now, change the order of the axes and create a new XYZImage
    # that is not-diagonal

    cm = im.coordmap
    cm_reordered = cm.reordered_domain(['j','k','i', 'l'])
    transposed_data = np.transpose(np.array(im), [1,2,0,3])

    im_reordered = Image(transposed_data, cm_reordered)
    B = np.identity(4)
    B[:3,:3] = im_reordered.affine[:3,:3]
    B[:3,-1] = im_reordered.affine[:3,-1]

    a2=xyz_image.XYZImage(np.array(im_reordered), B,
                                  im_reordered.coordmap.function_domain.coord_names)
    # Now, reorder it

    a3 = a2.xyz_ordered()

    yield assert_almost_equal, a3.affine, a.affine

    # as a subclass of Image, it still has a coordmap
    # describing ALL its axes


    yield assert_equal, a.coordmap.function_domain.coord_names , ('i', 'j', 'k', 'l')
    
    yield assert_equal, a.coordmap.function_range.coord_names , lps + ('l',)

    yield assert_equal, a2.coordmap.function_domain.coord_names , ('j', 'k', 'i', 'l')
    
    yield assert_equal, a2.coordmap.function_range.coord_names , lps + ('l',)

    yield assert_equal, a3.coordmap.function_domain.coord_names , ('i', 'j', 'k', 'l')
    
    yield assert_equal, a3.coordmap.function_range.coord_names , lps + ('l',)

    # But it xyz_transform is ony a 3d coordmap

    yield assert_equal, a.xyz_transform.function_domain.coord_names , ('i', 'j', 'k')
    
    yield assert_equal, a.xyz_transform.function_range.coord_names , lps

    yield assert_equal, a2.xyz_transform.function_domain.coord_names , ('j', 'k', 'i')
    
    yield assert_equal, a2.xyz_transform.function_range.coord_names , lps

    yield assert_equal, a3.xyz_transform.function_domain.coord_names , ('i', 'j', 'k')
    
    yield assert_equal, a3.xyz_transform.function_range.coord_names , lps



def test_resample():
    # XXX as written in the proposal, I don't think these will work for an XYZImage 
    # with data.ndim == 4 or should they?

    im, xyz_im = generate_im()

    xyz_im_resampled = xyz_im.resampled_to_affine(xyz_im.xyz_transform)
    #yield assert_almost_equal, np.array(xyz_im_resampled), np.array(xyz_im)

    xyz_im_resampled2 = xyz_im.resampled_to_img(xyz_im)
    #yield assert_almost_equal, np.array(xyz_im_resampled2), np.array(xyz_im)


def test_values_in_world():
    # XXX this shouldn't work for an XYZImage with data.ndim == 4 in the proposal for XYZImage, should they?

    im, xyz_im = generate_im()

    xyz_vals = xyz_im.xyz_transform(np.array([[3,4,5],
                                                    [4,7,8]]))
    x = xyz_vals[:,0]
    y = xyz_vals[:,1]
    z = xyz_vals[:,2]

    v1, v2 = xyz_im.values_in_world(x,y,z)
    yield assert_almost_equal, v1, np.array(xyz_im)[3,4,5]
    yield assert_equal, v1.shape, (xyz_im.shape[3],)
    yield assert_almost_equal, v2, np.array(xyz_im)[4,7,8]


