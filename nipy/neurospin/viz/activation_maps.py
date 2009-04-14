#!/usr/bin/env python

"""
Functions to do automatic visualization of activation-like maps.

For 2D-only visualization, only matplotlib is required.
For 3D visualization, Mayavi, version 3.0 or greater, is required.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD

# Standard library imports
import os
import tempfile
import sys

# Standard scientific libraries imports (more specific imports are
# delayed, so that the part module can be used without them).
import numpy as np
import pylab as pl
from scipy import ndimage, stats
from nifti import NiftiImage

# Local imports
from fff2.utils.mask import compute_mask, _largest_cc
from fff2.utils.emp_null import ENN
import fff2.data

# The sform for MNI templates
mni_sform = np.array([[-1, 0, 0,   90],
                      [ 0, 1, 0, -126],
                      [ 0, 0, 1,  -72],
                      [ 0, 0, 0,   1]])

mni_sform_inv = np.linalg.inv(mni_sform)

mni_shape = np.array((182., 218., 182.))

class SformError(Exception):
    pass

class NiftiIndexError(IndexError):
    pass

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
# Caching of the MNI template. 
################################################################################

class _AnatCache(object):
    """ Class to store the anat array in cache, to avoid reloading it
        each time.
    """
    anat        = None
    anat_sform  = None
    blurred     = None

    @classmethod
    def get_anat(cls):
        if cls.anat is not None:
            return cls.anat, cls.anat_sform, cls.anat_max
        anat_im = NiftiImage(
                    os.path.join(os.path.dirname(
                        os.path.realpath(fff2.data.__file__)),
                        'MNI152_T1_1mm_brain.nii.gz'
                    ))
        anat = anat_im.data.T
        anat = anat.astype(np.float)
        anat_mask = ndimage.morphology.binary_fill_holes(anat > 0)
        anat = np.ma.masked_array(anat, np.logical_not(anat_mask))
        cls.anat_sform = anat_im.sform
        cls.anat = anat
        cls.anat_max = anat.max()
        return cls.anat, cls.anat_sform, cls.anat_max

    @classmethod
    def get_blurred(cls):
        if cls.blurred is not None:
            return cls.blurred
        anat, _, _ = cls.get_anat()
        cls.blurred = ndimage.gaussian_filter(
                (ndimage.morphology.binary_fill_holes(
                    ndimage.gaussian_filter(
                            (anat > 4800).astype(np.float), 6)
                        > 0.5
                    )).astype(np.float),
                2).T.ravel()
        return cls.blurred


################################################################################
# 2D plotting 
################################################################################


def plot_map_2d(map, sform, cut_coords, anat=None, anat_sform=None,
                    vmin=None, figure_num=None, axes=None, title='',
                    mask=None):
    """ Plot three cuts of a given activation map (Frontal, Axial, and Lateral)

        Parameters
        ----------
        map : 3D ndarray
            The activation map, as a 3D image.
        sform : 4x4 ndarray
            The affine matrix going from image voxel space to MNI space.
        cut_coords: 3-tuple of floats
            The MNI coordinates of the point where the cut is performed, in 
            MNI coordinates and order.
        anat : 3D ndarray, optional
            The anatomical image to be used as a background. If None, the 
            MNI152 T1 1mm template is used.
        anat_sform : 4x4 ndarray, optional
            The affine matrix going from the anatomical image voxel space to 
            MNI space. This parameter is not used when the default 
            anatomical is used, but it is compulsory when using an
            explicite anatomical image.
        vmin : float, optional
            The lower threshold of the positive activation. This
            parameter is used to threshold the activation map.
        figure_num : integer, optional
            The number of the matplotlib figure used. If None is given, a
            new figure is created.
        axes : 4 tuple of float: (xmin, xmax, ymin, ymin), optional
            The coordinates, in matplotlib figure space, of the axes
            used to display the plot. If None, the complete figure is 
            used.
        title : string, optional
            The title dispayed on the figure.
        mask : 3D ndarray, boolean, optional
            The brain mask. If None, the mask is computed from the map.

        Notes
        -----
        All the 3D arrays are in numpy convention: (x, y, z)

        Cut coordinates are in Talairach coordinates. Warning: Talairach
        coordinates are (y, x, z), if (x, y, z) are in voxel-ordering
        convention.
    """
    if anat is None:
        anat, anat_sform, vmax_anat = _AnatCache.get_anat()
    else:
        vmax_anat = anat.max()

    if mask is not None and (
                    np.all(mask) or np.all(np.logical_not(mask))):
        mask = None

    vmin_map  = map.min()
    vmax_map  = map.max()
    if vmin is not None and np.isfinite(vmin):
        map = np.ma.masked_less(map, vmin)
    elif mask is not None:
        map = np.ma.masked_array(map, np.logical_not(mask))
        vmin_map  = map.min()
        vmax_map  = map.max()

    if isinstance(map, np.ma.core.MaskedArray):
        use_mask = False
        if map._mask is False or np.all(np.logical_not(map._mask)):
            map = np.asarray(map)
        elif map._mask is True or np.all(map._mask):
            map = np.asarray(map)
        if use_mask and mask is not None:
            map = np.ma.masked_array(map, np.logical_not(mask))

    # Calculate the bounds
    anat_bounds = np.zeros((4, 6))
    anat_bounds[:3, -3:] = np.identity(3)*anat.shape
    anat_bounds[-1, :] = 1
    anat_bounds = np.dot(anat_sform, anat_bounds)

    map_bounds = np.zeros((4, 6))
    map_bounds[:3, -3:] = np.identity(3)*map.shape
    map_bounds[-1, :] = 1
    map_bounds = np.dot(sform, map_bounds)

    # The coordinates of the center of the cut in different spaces.
    y, x, z = cut_coords
    x_map, y_map, z_map = [int(round(c)) for c in 
                            coord_transform(x, y, z,
                                    np.linalg.inv(sform))]
    x_anat, y_anat, z_anat = [int(round(c)) for c in 
                            coord_transform(x, y, z,
                                    np.linalg.inv(anat_sform))]


    fig = pl.figure(figure_num, figsize=(6.6, 2.6))
    if axes is None:
        axes = (0., 1., 0., 1.)
        pl.clf()
    ax_xmin, ax_xmax, ax_ymin, ax_ymax = axes
    ax_width = ax_xmax - ax_xmin
    ax_height = ax_ymax - ax_ymin
    
    # Calculate the axes ratio size in a 'clever' way
    shapes = np.array(anat.shape, 'f')
    shapes *= ax_width/shapes.sum()
    
    ###########################################################################
    # Frontal
    pl.axes([ax_xmin, ax_ymin, shapes[0], ax_height])
    if y_anat < anat.shape[1]:
        pl.imshow(np.rot90(anat[:, y_anat, :]), 
                                cmap=pl.cm.gray,
                                vmin=-.5*vmax_anat,
                                vmax=vmax_anat, 
                                extent=(anat_bounds[0, 3],
                                        anat_bounds[0, 0],
                                        anat_bounds[2, 0],
                                        anat_bounds[2, 5]))
    xmin, xmax = pl.xlim()
    ymin, ymax = pl.ylim()
    pl.hlines(z, xmin, xmax, color=(.5, .5, .5))
    pl.vlines(-x, ymin, ymax, color=(.5, .5, .5))
    if y_map < map.shape[1]:
        pl.imshow(np.rot90(map[:, y_map, :]),
                                vmin=vmin_map,
                                vmax=vmax_map,
                                extent=(map_bounds[0, 3],
                                        map_bounds[0, 0],
                                        map_bounds[2, 0],
                                        map_bounds[2, 5]))
    pl.text(ax_xmin +shapes[0] + shapes[1] - 0.01, ax_ymin + 0.07, '%i' % x,
             horizontalalignment='right',
             verticalalignment='bottom',
             transform=fig.transFigure)
    
    pl.axis('off')
    
    ###########################################################################
    # Lateral
    pl.axes([ax_xmin + shapes[0], ax_ymin, shapes[1], ax_height])
    if x_anat < anat.shape[0]:
        pl.imshow(np.rot90(anat[x_anat, ...]), cmap=pl.cm.gray,
                                vmin=-.5*vmax_anat,
                                vmax=vmax_anat, 
                                extent=(anat_bounds[1, 0],
                                        anat_bounds[1, 4],
                                        anat_bounds[2, 0],
                                        anat_bounds[2, 5]))
    xmin, xmax = pl.xlim()
    ymin, ymax = pl.ylim()
    pl.hlines(z, xmin, xmax, color=(.5, .5, .5))
    pl.vlines(y, ymin, ymax, color=(.5, .5, .5))
    if x_map < map.shape[0]:
        pl.imshow(np.rot90(map[x_map, ...]),
                                vmin=vmin_map,
                                vmax=vmax_map,
                                extent=(map_bounds[1, 0],
                                        map_bounds[1, 4],
                                        map_bounds[2, 0],
                                        map_bounds[2, 5]))
    pl.text(ax_xmin + shapes[-1] - 0.01, ax_ymin + 0.07, '%i' % y, 
             horizontalalignment='right',
             verticalalignment='bottom',
             transform=fig.transFigure)
    
    pl.axis('off')

    ###########################################################################
    # Axial
    pl.axes([ax_xmin + shapes[0] + shapes[1], ax_ymin, shapes[-1],
                ax_height])
    if z_anat < anat.shape[2]:
        pl.imshow(np.rot90(anat[..., z_anat]), 
                                cmap=pl.cm.gray,
                                vmin=-.5*vmax_anat,
                                vmax=vmax_anat, 
                                extent=(anat_bounds[0, 0],
                                        anat_bounds[0, 3],
                                        anat_bounds[1, 0],
                                        anat_bounds[1, 4]))
    xmin, xmax = pl.xlim()
    ymin, ymax = pl.ylim()
    pl.hlines(y,  xmin, xmax, color=(.5, .5, .5))
    pl.vlines(x, ymin, ymax, color=(.5, .5, .5))
    if z_map < map.shape[2]:
        pl.imshow(np.rot90(map[..., z_map]),
                                vmin=vmin_map,
                                vmax=vmax_map,
                                extent=(map_bounds[0, 0],
                                        map_bounds[0, 3],
                                        map_bounds[1, 0],
                                        map_bounds[1, 4]))
    pl.text(ax_xmax - 0.01, ax_ymin + 0.07, '%i' % z, 
             horizontalalignment='right',
             verticalalignment='bottom',
             transform=fig.transFigure)
    
    pl.axis('off')
    
    pl.text(ax_xmin + 0.01, ax_ymax - 0.01, title, 
             horizontalalignment='left',
             verticalalignment='top',
             transform=fig.transFigure)

    pl.axis('off')


def demo_plot_map_2d():
    map = np.zeros((182, 218, 182))
    # Color a asymetric rectangle around Broadman area 26:
    x, y, z = -6, -53, 9
    x_map, y_map, z_map = coord_transform(x, y, z, mni_sform_inv)
    map[x_map-30:x_map+30, y_map-3:y_map+3, z_map-10:z_map+10] = 1
    map = np.ma.masked_less(map, 0.5)
    plot_map_2d(map, mni_sform, cut_coords=(x, y, z),
                                figure_num=512)


def plot_map_3d(map, sform, cut_coords=None, anat=None, anat_sform=None,
    vmin=None, figure_num=None, mask=None):
    """ Plot a 3D volume rendering view of the activation, with an
        outline of the brain.

        Parameters
        ----------
        map : 3D ndarray
            The activation map, as a 3D image.
        sform : 4x4 ndarray
            The affine matrix going from image voxel space to MNI space.
        cut_coords: 3-tuple of floats, optional
            The MNI coordinates of a 3D cursor to indicate a feature
            or a cut, in MNI coordinates and order.
        anat : 3D ndarray, optional
            The anatomical image to be used as a background. If None, the 
            MNI152 T1 1mm template is used.
        anat_sform : 4x4 ndarray, optional
            The affine matrix going from the anatomical image voxel space to 
            MNI space. This parameter is not used when the default 
            anatomical is used, but it is compulsory when using an
            explicite anatomical image.
        vmin : float, optional
            The lower threshold of the positive activation. This
            parameter is used to threshold the activation map.
        figure_num : integer, optional
            The number of the Mayavi figure used. If None is given, a
            new figure is created.
        mask : 3D ndarray, boolean, optional
            The brain mask. If None, the mask is computed from the map.

        Notes
        -----
        All the 3D arrays are in numpy convention: (x, y, z)

        Cut coordinates are in Talairach coordinates. Warning: Talairach
        coordinates are (y, x, z), if (x, y, z) are in voxel-ordering
        convention.

        If you are using a VTK version below 5.2, there is no way to
        avoid opening a window during the rendering under Linux. This is 
        necessary to use the graphics card for the rendering. You must
        maintain this window on top of others and on the screen.

        If you are running on Windows, or using a recent version of VTK,
        you can force offscreen mode using::

            from enthought.mayavi import mlab
            mlab.options.offscreen = True
    """

    from enthought.mayavi import mlab
    from enthought.mayavi.sources.api import ArraySource
    if anat is None:
        anat, anat_sform, anat_max = _AnatCache.get_anat()

    fig = mlab.figure(figure_num, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0),
                                                     size=(400, 350))
    if figure_num is None:
        mlab.clf()
    fig.scene.disable_render = True
    
    center = np.r_[0, 0, 0, 1]

    ###########################################################################
    # Display the map using volume rendering
    spacing = np.diag(sform)[:3]
    origin = np.dot(sform, center)[:3]
    # XXX: VTK orients X (negative spacing) in the wrong direction
    spacing[0] *= -1
    origin[0] *= -1
    map_src = ArraySource(scalar_data=np.asarray(map),
                          name='Activation map',
                          spacing=spacing,
                          origin=origin)
    #map_src.update_image_data = True
    vol = mlab.pipeline.volume(map_src)

    # Change the opacity function
    from enthought.tvtk.util.ctf import PiecewiseFunction
    vmin_map = map.min()
    vmax_map = map.max()
    if vmin is None:
        vmin = find_activation(map, upper_only=True, mask=mask)
    otf = PiecewiseFunction()
    otf.add_point(vmin_map, 0)
    otf.add_point(max(0, vmin), 0)
    otf.add_point(vmax_map, 1)
    vol._volume_property.set_scalar_opacity(otf)
    vol.update_ctf = True
    
    ###########################################################################
    # Display the cortical surface (flattenned)
    spacing = np.diag(anat_sform)[:3]
    # XXX: VTK orients X (negative spacing) in the wrong direction
    origin = np.dot(anat_sform, center)[:3]
    spacing[0] *= -1
    origin[0] *= -1
    anat_src = ArraySource(scalar_data=np.asarray(anat), 
                           name='Anat',
                           # XXX: we inflate a bit the anatomy
                           spacing=1.05*spacing,
                           origin=origin)
    #anat_src.update_image_data = True
    
    anat_src.image_data.point_data.add_array(_AnatCache.get_blurred())
    anat_src.image_data.point_data.get_array(1).name = 'blurred'
            
    cortex_surf = mlab.pipeline.set_active_attribute(
                    mlab.pipeline.contour(
                        mlab.pipeline.set_active_attribute(
                                anat_src, point_scalars='blurred'), 
                    ), point_scalars='scalar')
        
    # XXX: the choice in vmin and vmax should be tuned to show the
    # sulci better
    cortex = mlab.pipeline.surface(cortex_surf,
                opacity=0.5, colormap='copper', vmin=4800, vmax=5000)
    cortex.enable_contours = True
    cortex.contour.filled_contours = True
    cortex.contour.auto_contours = False
    cortex.contour.contours = [0, 5000, 7227.8]
    # XXX: Why do I need to cull the front face?
    cortex.actor.property.backface_culling = True
    #cortex.actor.property.frontface_culling = True

    cortex.actor.mapper.interpolate_scalars_before_mapping = True
    cortex.actor.property.interpolation = 'flat'

    # Add opacity variation to the colormap
    cmap = cortex.module_manager.scalar_lut_manager.lut.table.to_array()
    cmap[128:, -1] = 0.7*255
    cortex.module_manager.scalar_lut_manager.lut.table = cmap
    
    ###########################################################################
    # Draw the cursor
    if cut_coords is not None:
        y0, x0, z0 = cut_coords
        line1 = mlab.plot3d((-90, 90), (y0, y0), (z0, z0), 
                            color=(.5, .5, .5), tube_radius=0.25)
        line2 = mlab.plot3d((-x0, -x0), (-126, 91), (z0, z0), 
                            color=(.5, .5, .5), tube_radius=0.25)
        line3 = mlab.plot3d((-x0, -x0), (y0, y0), (-72, 109), 
                            color=(.5, .5, .5), tube_radius=0.25)
    
    mlab.view(38.5, 70.5, 300, (-2.7, -12, 9.1))
    fig.scene.disable_render = False


def demo_plot_map_3d():
    map = np.zeros((182, 218, 182))
    # Color a asymetric rectangle around Broadman area 26:
    x, y, z = -6, -53, 9
    x_map, y_map, z_map = coord_transform(x, y, z, mni_sform_inv)
    map[x_map-30:x_map+30, y_map-3:y_map+3, z_map-10:z_map+10] = 1
    map = map.T
    plot_map_3d(map, mni_sform, cut_coords=(x, y, z), vmin=0.5,
                                figure_num=512)


def plot_map(map, sform, cut_coords, anat=None, anat_sform=None,
    vmin=None, figure_num=None, title='', mask=None):
    """ Plot a together a 3D volume rendering view of the activation, with an
        outline of the brain, and 2D cuts. If Mayavi is not installed,
        falls back to 2D views only.

        Parameters
        ----------
        map : 3D ndarray
            The activation map, as a 3D image.
        sform : 4x4 ndarray
            The affine matrix going from image voxel space to MNI space.
        cut_coords: 3-tuple of floats, optional
            The MNI coordinates of the cut to perform, in MNI coordinates 
            and order. If None is given, the cut_coords are automaticaly
            estimated.
        anat : 3D ndarray, optional
            The anatomical image to be used as a background. If None, the 
            MNI152 T1 1mm template is used.
        anat_sform : 4x4 ndarray, optional
            The affine matrix going from the anatomical image voxel space to 
            MNI space. This parameter is not used when the default 
            anatomical is used, but it is compulsory when using an
            explicite anatomical image.
        vmin : float, optional
            The lower threshold of the positive activation. This
            parameter is used to threshold the activation map.
        figure_num : integer, optional
            The number of the matplotlib and Mayavi figures used. If None is 
            given, a new figure is created.
        title : string, optional
            The title dispayed on the figure.
        mask : 3D ndarray, boolean, optional
            The brain mask. If None, the mask is computed from the map.

        Notes
        -----
        All the 3D arrays are in numpy convention: (x, y, z)

        Cut coordinates are in Talairach coordinates. Warning: Talairach
        coordinates are (y, x, z), if (x, y, z) are in voxel-ordering
        convention.
    """
    try:
        from enthought.mayavi import version
        if not int(version.version[0]) > 2:
            raise ImportError
    except ImportError:
        print >> sys.stderr, 'Mayavi > 3.x not installed, plotting only 2D'
        return plot_map_2d(map, sform, cut_coords=cut_coords, anat=anat,
                                anat_sform=anat_sform, vmin=vmin,
                                title=title,
                                figure_num=figure_num, mask=mask)

    from enthought.mayavi import mlab
    plot_map_3d(map, sform, cut_coords=cut_coords, anat=anat,
                anat_sform=anat_sform, vmin=vmin,
                figure_num=figure_num, mask=mask)
    filename = tempfile.mktemp('.png')
    mlab.savefig(filename)
    image3d = pl.imread(filename)
    os.unlink(filename)
    
    fig = pl.figure(figure_num, figsize=(10.6, 2.6))
    pl.axes((-0.01, 0, 0.3, 1))
    pl.imshow(image3d)
    pl.axis('off')
    
    plot_map_2d(map, sform, cut_coords=cut_coords, anat=anat,
                anat_sform=anat_sform, vmin=vmin, mask=mask,
                figure_num=fig.number, axes=(0.28, 1, 0, 1.), title=title)


def demo_plot_map():
    map = np.zeros((182, 218, 182))
    # Color a asymetric rectangle around Broadman area 26:
    x, y, z = -6, -53, 9
    x_map, y_map, z_map = coord_transform(x, y, z, mni_sform_inv)
    map[x_map-30:x_map+30, y_map-3:y_map+3, z_map-10:z_map+10] = 1
    plot_map(map, mni_sform, cut_coords=(x, y, z), vmin=0.5,
                                figure_num=512)


def auto_plot_map(map, sform, vmin=None, cut_coords=None, do3d=False, 
                    anat=None, anat_sform=None, title='',
                    figure_num=None, mask=None, auto_sign=True):
    """ Automatic plotting of an activation map.

        Plot a together a 3D volume rendering view of the activation, with an
        outline of the brain, and 2D cuts. If Mayavi is not installed,
        falls back to 2D views only.

        Parameters
        ----------
        map : 3D ndarray
            The activation map, as a 3D image.
        sform : 4x4 ndarray
            The affine matrix going from image voxel space to MNI space.
        vmin : float, optional
            The lower threshold of the positive activation. This
            parameter is used to threshold the activation map.
        cut_coords: 3-tuple of floats, optional
            The MNI coordinates of the point where the cut is performed, in 
            MNI coordinates and order. If None is given, the cut_coords are 
            automaticaly estimated.
        do3d : boolean, optional
            If do3d is True, a 3D plot is created if Mayavi is installed.
        anat : 3D ndarray, optional
            The anatomical image to be used as a background. If None, the 
            MNI152 T1 1mm template is used.
        anat_sform : 4x4 ndarray, optional
            The affine matrix going from the anatomical image voxel space to 
            MNI space. This parameter is not used when the default 
            anatomical is used, but it is compulsory when using an
            explicite anatomical image.
        title : string, optional
            The title dispayed on the figure.
        figure_num : integer, optional
            The number of the matplotlib and Mayavi figures used. If None is 
            given, a new figure is created.
        mask : 3D ndarray, boolean, optional
            The brain mask. If None, the mask is computed from the map.
        auto_sign : boolean, optional
            If auto_sign is True, the sign of the activation is
            automaticaly computed: negative activation can thus be
            plotted.

        Returns
        -------
        vmin : float
            The lower threshold of the activation used.
        cut_coords : 3-tuple of floats
            The Talairach coordinates of the cut performed for the 2D
            view.

        Notes
        -----
        All the 3D arrays are in numpy convention: (x, y, z)

        Cut coordinates are in Talairach coordinates. Warning: Talairach
        coordinates are (y, x, z), if (x, y, z) are in voxel-ordering
        convention.
    """
    if do3d:
        if do3d == 'offscreen':
            try:
                from enthought.mayavi import mlab
                mlab.options.offscreen = True
            except:
                pass
        plotter = plot_map
    else:
        plotter = plot_map_2d
    if mask is None:
        mask = compute_mask(map)
    if vmin is None:
        vmin = np.inf
        pvalue = 0.04
        while not np.isfinite(vmin):
            pvalue *= 1.25
            vmax, vmin = find_activation(map, mask=mask, pvalue=pvalue)
            if not np.isfinite(vmin) and auto_sign:
                if np.isfinite(vmax):
                    vmin = -vmax
                    if mask is not None:
                        map[mask] *= -1
                    else:
                        map *= -1
    if cut_coords is None:
        x, y, z = find_cut_coords(map, activation_threshold=vmin)
        # XXX: Careful with Voxel/MNI ordering
        y, x, z = coord_transform(x, y, z, sform)
        cut_coords = (x, y, z)
    plotter(map, sform, vmin=vmin, cut_coords=cut_coords,
                anat=anat, anat_sform=anat_sform, title=title,
                figure_num=figure_num, mask=mask)
    return vmin, cut_coords


def plot_niftifile(filename, outputname=None, do3d=False, vmin=None,
            cut_coords=None, anat_filename=None, figure_num=None,
            mask_filename=None, auto_sign=True):
    """ Given a nifti filename, plot a view of it to a file (png by
        default).

        Parameters
        ----------
        filename : string 
            The name of the Nifti file of the map to be plotted 
        outputname : string, optional 
            The file name of the output file created. By default
            the name of the input file with a png extension is used. 
        do3d : boolean, optional
            If do3d is True, a 3D plot is created if Mayavi is installed.
        vmin : float, optional
            The lower threshold of the positive activation. This
            parameter is used to threshold the activation map.
        cut_coords: 3-tuple of floats, optional
            The MNI coordinates of the point where the cut is performed, in 
            MNI coordinates and order. If None is given, the cut_coords are 
            automaticaly estimated.
        anat : string, optional
            Name of the Nifti image file to be used as a background. If None, 
            the MNI152 T1 1mm template is used.
        title : string, optional
            The title dispayed on the figure.
        figure_num : integer, optional
            The number of the matplotlib and Mayavi figures used. If None is 
            given, a new figure is created.
        mask_filename : string, optional
            Name of the Nifti file to be used as brain mask. If None, the 
            mask is computed from the map.
        auto_sign : boolean, optional
            If auto_sign is True, the sign of the activation is
            automaticaly computed: negative activation can thus be
            plotted.

        Notes
        -----

        Cut coordinates are in Talairach coordinates. Warning: Talairach
        coordinates are (y, x, z), if (x, y, z) are in voxel-ordering
        convention.
    """

    if outputname is None:
        outputname = os.path.splitext(filename)[0] + '.png'
    if not os.path.exists(filename):
        raise OSError, 'File %s does not exist' % filename
        
    nim = NiftiImage(filename)
    sform = nim.sform
    if any(np.linalg.eigvals(sform)==0):
        raise SformError, "sform affine is not inversible"
    if anat_filename is not None:
        anat_im = NiftiImage(anat_filename)
        anat = anat_im.data.T
        anat_sform = anat_im.sform
    else:
        anat = None
        anat_sform = None

    if mask_filename is not None:
        mask_im = NiftiImage(mask_filename)
        mask = mask_im.data.T.astype(np.bool)
        if not np.allclose(mask_im.sform, sform):
            raise SformError, 'Mask does not have same sform as image'
        if not np.allclose(mask.shape, nim.data.shape[-3:]):
            raise NiftiIndexError, 'Mask does not have same shape as image'

    output_files = list()

    if nim.data.ndim == 3:
        map = nim.data.T
        auto_plot_map(map, sform, vmin=vmin, cut_coords=cut_coords,
                do3d=do3d, anat=anat, anat_sform=anat_sform, mask=mask,
                title=os.path.basename(filename), figure_num=figure_num,
                auto_sign=auto_sign)
        pl.savefig(outputname)
        output_files.append(outputname)
    elif nim.data.ndim == 4:
        outputname, outputext = os.path.splitext(outputname)
        if len(nim.data) < 10:
            fmt = '%s_%i%s'
        elif len(nim.data) < 100:
            fmt = '%s_%02i%s'
        elif len(nim.data) < 1000:
            fmt = '%s_%03i%s'
        else:
            fmt = '%s_%04i%s'
        if mask is None:
            mask = compute_mask(nim.data.mean(axis=0)).T
        for index, data in enumerate(nim.data):
            map = data.T
            auto_plot_map(map, sform, vmin=vmin, cut_coords=cut_coords,
                    do3d=do3d, anat=anat, anat_sform=anat_sform,
                    title='%s, %i' % (os.path.basename(filename), index),
                    figure_num=figure_num, mask=mask, auto_sign=auto_sign)
            this_outputname = fmt % (outputname, index, outputext)
            pl.savefig(this_outputname)
            pl.clf()
            output_files.append(this_outputname)
    else:
        raise NiftiIndexError, 'File %s: incorrect number of dimensions'
    return output_files


if __name__ == '__main__':
    test_coord_transform_trivial()
    test_find_cut_coords()

    filename = sys.argv[1]
    print "Rendering a visualization of %s" % filename
    plot_niftifile(filename, do3d=True)
