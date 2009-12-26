"""
Misc tools to find activations and cut on maps
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD

# Standard scientific libraries imports (more specific imports are
# delayed, so that the part module can be used without them).
import numpy as np
from scipy import stats, ndimage

# Local imports
from nipy.neurospin.utils.mask import compute_mask, _largest_cc
from nipy.neurospin.utils.emp_null import ENN

################################################################################
# Functions for automatic choice of cuts coordinates
################################################################################

def largest_cc(mask):
    """ Largest connect components extraction code that fails gracefully 
        when mask is too big for C extension code.
    """
    try:
        mask = _largest_cc(mask)
    except TypeError:
        pass
    return mask.astype(np.bool)


def threshold_connect_components(map, threshold, copy=True):
    """ Given a map with some coefficients set to zero, segment the
        connect components with number of voxels smaller than the
        threshold and set them to 0.

        Parameters
        ----------
        map: ndarray
            The map to segment
        threshold:
            The minimum number of voxels to keep a cluster.
        copy: bool, optional
            If copy is false, the input array is modified inplace
    """
    labels, n_labels = ndimage.label(map)
    weights = ndimage.sum(np.ones_like(labels), 
                          labels, index=range(n_labels)) 
    if copy:
        map = map.copy()
    for index, weight in enumerate(weights):
        if index == 0:
            continue
        if weight < threshold:
            map[labels == index] = 0
    return map


def coord_transform(x, y, z, affine):
    """ Convert the x, y, z coordinates from one image space to another
        space. 
        
        Parameters
        ----------
        x : number or ndarray
            The x coordinates in the input space
        y : number or ndarray
            The y coordinates in the input space
        z : number or ndarray
            The z coordinates in the input space
        affine : 2D 4x4 ndarray
            affine that maps from input to output space.

        Returns
        -------
        x : number or ndarray
            The x coordinates in the output space
        y : number or ndarray
            The y coordinates in the output space
        z : number or ndarray
            The z coordinates in the output space

        Warning: The x, y and z have their Talairach ordering, not 3D
        numy image ordering.
    """
    coords = np.c_[np.atleast_1d(x).flat, 
                   np.atleast_1d(y).flat, 
                   np.atleast_1d(z).flat,
                   np.ones_like(np.atleast_1d(z).flat)].T
    x, y, z, _ = np.dot(affine, coords)
    return x.squeeze(), y.squeeze(), z.squeeze()


def test_coord_transform_trivial():
    sform = np.eye(4)
    x = np.random.random((10,))
    y = np.random.random((10,))
    z = np.random.random((10,))

    x_, y_, z_ = coord_transform(x, y, z, sform)
    np.testing.assert_array_equal(x, x_)
    np.testing.assert_array_equal(y, y_)
    np.testing.assert_array_equal(z, z_)

    sform[:, -1] = 1
    x_, y_, z_ = coord_transform(x, y, z, sform)
    np.testing.assert_array_equal(x+1, x_)
    np.testing.assert_array_equal(y+1, y_)
    np.testing.assert_array_equal(z+1, z_)


def find_activation(map, mask=None, pvalue=0.05, upper_only=False):
    """ Use the empirical normal null estimator to find threshold for 
        negative and positive activation.

        Parameters
        ----------
        map : 3D ndarray
            The activation map, as a 3D image.
        mask : 3D ndarray, boolean, optional
            The brain mask. If None, the mask is computed from the map.
        pvalue : float, optional
            The pvalue of the false discovery rate used.
        upper_only : boolean, optional
            If true, only a threshold for positive activation is estimated.

        Returns
        -------
        vmin : float, optional
            The upper threshold for negative activation, not returned 
            if upper_only is True.
        vmax : float
            The lower threshod for positive activation.
    """
    
    if mask is None:
        mask = compute_mask(map)
    map = map[mask]
    vmax = ENN(map).threshold(alpha=pvalue)
    if upper_only:
        return vmax
    vmin = -ENN(-map).threshold(alpha=pvalue)
    return vmin, vmax


def find_cut_coords(map, mask=None, activation_threshold=None):
    """ Find the center of the largest activation connect component.

        Parameters
        -----------
        map : 3D ndarray
            The activation map, as a 3D image.
        mask : 3D ndarray, boolean, optional
            The brain mask. If None, the mask is computed from the map.
        activation_threshold : float, optional
            The lower threshold to the positive activation. If None, the 
            activation threshold is computed using find_activation.

        Returns
        -------
        x: float
            the x coordinate in voxels.
        y: float
            the y coordinate in voxels.
        z: float
            the z coordinate in voxels.
    """
    #
    my_map = map.copy()
    if activation_threshold is None:
        vmin, vmax = find_activation(map, mask=mask)
        mask = (map<vmin) | (map>vmax)
    else:
        mask = map>activation_threshold
    if np.any(mask):
        mask = largest_cc(mask)
        my_map[np.logical_not(mask)] = 0
        second_threshold = stats.scoreatpercentile(my_map[mask], 60)
        if (my_map>second_threshold).sum() > 50:
            my_map[np.logical_not(largest_cc(my_map>second_threshold))] = 0
    cut_coords = ndimage.center_of_mass(my_map)
    return cut_coords


def test_find_cut_coords():
    map = np.zeros((100, 100, 100))
    x_map, y_map, z_map = 50, 10, 40
    map[x_map-30:x_map+30, y_map-3:y_map+3, z_map-10:z_map+10] = 1
    x, y, z = find_cut_coords(map, mask=np.ones(map.shape, np.bool))
    np.testing.assert_array_equal(
                        (int(round(x)), int(round(y)), int(round(z))),
                                (x_map, y_map, z_map))


################################################################################
def get_bounds(shape, affine):
    """ Return the world-space bounds occupied by an array given an affine.
    """
    bounds = np.zeros((4, 6))
    bounds[:3, -3:] = np.identity(3)*shape
    bounds[-1, :] = 1
    bounds = np.dot(affine, bounds)
    xmin = float(bounds[0, 0])
    xmax = float(bounds[0, 3])
    zmin = float(bounds[2, 0])
    zmax = float(bounds[2, 5])
    ymin = float(bounds[1, 0])
    ymax = float(bounds[1, 4])
    return xmin, xmax, ymin, ymax, zmin, zmax


def get_mask_bounds(mask, affine):
    """ Return the world-space bounds occupied by a mask given an affine.

        Notes
        -----

        The mask should have only one connect component.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = get_bounds(mask.shape, affine)
    x_slice, y_slice, z_slice = ndimage.find_objects(mask)[0]
    x_width, y_width, z_width = mask.shape
    xmin, xmax = (xmin + x_slice.start*(xmax - xmin)/x_width,
                  xmin + x_slice.stop *(xmax - xmin)/x_width)
    ymin, ymax = (ymin + y_slice.start*(ymax - ymin)/y_width,
                  ymin + y_slice.stop *(ymax - ymin)/y_width)
    zmin, zmax = (zmin + z_slice.start*(zmax - zmin)/z_width,
                  zmin + z_slice.stop *(zmax - zmin)/z_width)

    return xmin, xmax, ymin, ymax, zmin, zmax
 

