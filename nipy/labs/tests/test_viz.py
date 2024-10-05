""" Tests for visualization
"""

import numpy as np

from nipy.labs.viz import coord_transform, mni_sform, plot_map


def test_example():
    # Example from tutorial.
    # First, create a fake activation map: a 3D image in MNI space with
    # a large rectangle of activation around Broca Area
    mni_sform_inv = np.linalg.inv(mni_sform)
    # Color an asymmetric rectangle around Broca area:
    x, y, z = -52, 10, 22
    x_map, y_map, z_map = (int(coord) for coord in coord_transform(x, y, z,
                                                                   mni_sform_inv))
    map = np.zeros((182, 218, 182))
    map[x_map-30:x_map+30, y_map-3:y_map+3, z_map-10:z_map+10] = 1

    # We use a masked array to add transparency to the parts that we are
    # not interested in:
    thresholded_map = np.ma.masked_less(map, 0.5)

    # And now, visualize it:
    plot_map(thresholded_map, mni_sform, cut_coords=(x, y, z), vmin=0.5)
