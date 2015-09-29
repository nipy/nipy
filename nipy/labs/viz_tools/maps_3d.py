# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
3D visualization of activation maps using Mayavi

"""
from __future__ import absolute_import

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD

import os
import tempfile

# Standard scientific libraries imports (more specific imports are
# delayed, so that the part module can be used without them).
import numpy as np
from scipy import stats

# Local imports
from .anat_cache import mni_sform, mni_sform_inv, _AnatCache
from .coord_tools import coord_transform

# A module global to avoid creating multiple time an offscreen engine.
off_screen_engine = None

################################################################################
# Helper functions
def affine_img_src(data, affine, scale=1, name='AffineImage',
                reverse_x=False):
    """ Make a Mayavi source defined by a 3D array and an affine, for
        wich the voxel of the 3D array are mapped by the affine.
        
        Parameters
        -----------
        data: 3D ndarray
            The data arrays
        affine: (4 x 4) ndarray
            The (4 x 4) affine matrix relating voxels to world
            coordinates.
        scale: float, optional
            An optional addition scaling factor.
        name: string, optional
            The name of the Mayavi source created.
        reverse_x: boolean, optional
            Reverse the x (lateral) axis. Useful to compared with
            images in radiologic convention.

        Notes
        ------
        The affine should be diagonal.
    """
    # Late import to avoid triggering wx imports before needed.
    try:
        from mayavi.sources.api import ArraySource
    except ImportError:
        # Try out old install of Mayavi, with namespace packages
        from enthought.mayavi.sources.api import ArraySource
    center = np.r_[0, 0, 0, 1]
    spacing = np.diag(affine)[:3].copy()
    origin = np.dot(affine, center)[:3]
    if reverse_x:
        # Radiologic convention
        spacing[0] *= -1
        origin[0] *= -1
    src = ArraySource(scalar_data=np.asarray(data, dtype=np.float),
                           name=name,
                           spacing=scale*spacing,
                           origin=scale*origin)
    return src 


################################################################################
# Mayavi helpers
def autocrop_img(img, bg_color):
    red, green, blue = bg_color

    outline =  ( (img[..., 0] != red)
                +(img[..., 1] != green)
                +(img[..., 2] != blue)
                )
    outline_x = outline.sum(axis=0)
    outline_y = outline.sum(axis=1)

    outline_x = np.where(outline_x)[0]
    outline_y = np.where(outline_y)[0]

    if len(outline_x) == 0:
        return img
    else:
        x_min = outline_x.min()
        x_max = outline_x.max()
    if len(outline_y) == 0:
        return img
    else:
        y_min = outline_y.min()
        y_max = outline_y.max()
    return img[y_min:y_max, x_min:x_max]



def m2screenshot(mayavi_fig=None, mpl_axes=None, autocrop=True):
    """ Capture a screeshot of the Mayavi figure and display it in the
        matplotlib axes.
    """
    import pylab as pl
    # Late import to avoid triggering wx imports before needed.
    try:
        from mayavi import mlab
    except ImportError:
        # Try out old install of Mayavi, with namespace packages
        from enthought.mayavi import mlab

    if mayavi_fig is None:
        mayavi_fig = mlab.gcf()
    else:
        mlab.figure(mayavi_fig)
    if mpl_axes is not None:
        pl.axes(mpl_axes)

    filename = tempfile.mktemp('.png')
    mlab.savefig(filename, figure=mayavi_fig)
    image3d = pl.imread(filename)
    if autocrop:
        bg_color = mayavi_fig.scene.background
        image3d = autocrop_img(image3d, bg_color)
    pl.imshow(image3d)
    pl.axis('off')
    os.unlink(filename)
    # XXX: Should switch back to previous MPL axes: we have a side effect
    # here.


################################################################################
# Anatomy outline
################################################################################

def plot_anat_3d(anat=None, anat_affine=None, scale=1,
                 sulci_opacity=0.5, gyri_opacity=0.3,
                 opacity=None,
                 skull_percentile=78, wm_percentile=79,
                 outline_color=None):
    """ 3D anatomical display

    Parameters
    ----------
    skull_percentile : float, optional
        The percentile of the values in the image that delimit the skull from
        the outside of the brain. The smaller the fraction of you field of view
        is occupied by the brain, the larger this value should be.
    wm_percentile : float, optional
        The percentile of the values in the image that delimit the white matter
        from the grey matter. Typical this is skull_percentile + 1
    """
    # Late import to avoid triggering wx imports before needed.
    try:
        from mayavi import mlab
    except ImportError:
        # Try out old install of Mayavi, with namespace packages
        from enthought.mayavi import mlab
    fig = mlab.gcf()
    disable_render = fig.scene.disable_render
    fig.scene.disable_render = True
    if anat is None:
        anat, anat_affine, anat_max = _AnatCache.get_anat()
        anat_blurred = _AnatCache.get_blurred()
        skull_threshold = 4800
        inner_threshold = 5000
        upper_threshold = 7227.8
    else:
        from scipy import ndimage
        # XXX: This should be in a separate function
        voxel_size = np.sqrt((anat_affine[:3, :3]**2).sum()/3.)
        skull_threshold = stats.scoreatpercentile(anat.ravel(), 
                skull_percentile)
        inner_threshold = stats.scoreatpercentile(anat.ravel(), 
                wm_percentile)
        upper_threshold = anat.max()
        anat_blurred = ndimage.gaussian_filter(
                            (ndimage.morphology.binary_fill_holes(
                                ndimage.gaussian_filter(
                                    (anat > skull_threshold).astype(np.float), 
                                    6./voxel_size)
                                    > 0.5
                                )).astype(np.float),
                            2./voxel_size).T.ravel()

    if opacity is None:
        try:
            from tvtk.api import tvtk
        except ImportError:
            # Try out old install of Mayavi, with namespace packages
            from enthought.tvtk.api import tvtk
        version = tvtk.Version()
        if (version.vtk_major_version, version.vtk_minor_version) < (5, 2):
            opacity = .99
        else:
            opacity = 1
    ###########################################################################
    # Display the cortical surface (flattenned)
    anat_src = affine_img_src(anat, anat_affine, scale=scale, name='Anat')
    
    anat_src.image_data.point_data.add_array(anat_blurred)
    anat_src.image_data.point_data.get_array(1).name = 'blurred'
    anat_src.image_data.point_data.update()
    anat_blurred = mlab.pipeline.set_active_attribute(
                                anat_src, point_scalars='blurred')

    anat_blurred.update_pipeline()
    # anat_blurred = anat_src
    
    cortex_surf = mlab.pipeline.set_active_attribute(
                            mlab.pipeline.contour(anat_blurred), 
                    point_scalars='scalar')
        
    # XXX: the choice in vmin and vmax should be tuned to show the
    # sulci better
    cortex = mlab.pipeline.surface(cortex_surf,
                colormap='copper', 
                opacity=opacity,
                vmin=skull_threshold, 
                vmax=inner_threshold)
    cortex.enable_contours = True
    cortex.contour.filled_contours = True
    cortex.contour.auto_contours = False
    cortex.contour.contours = [0, inner_threshold, upper_threshold]
    #cortex.actor.property.backface_culling = True
    # XXX: Why do we do 'frontface_culling' to see the front.
    cortex.actor.property.frontface_culling = True

    cortex.actor.mapper.interpolate_scalars_before_mapping = True
    cortex.actor.property.interpolation = 'flat'

    # Add opacity variation to the colormap
    cmap = cortex.module_manager.scalar_lut_manager.lut.table.to_array()
    cmap[128:, -1] = gyri_opacity*255
    cmap[:128, -1] = sulci_opacity*255
    cortex.module_manager.scalar_lut_manager.lut.table = cmap

    if outline_color is not None:
        outline = mlab.pipeline.iso_surface(
                            anat_blurred,
                            contours=[0.4],
                            color=outline_color, 
                            opacity=.9)
        outline.actor.property.backface_culling = True


    fig.scene.disable_render = disable_render
    return cortex
 

################################################################################
# Maps
################################################################################

def plot_map_3d(map, affine, cut_coords=None, anat=None, anat_affine=None,
    threshold=None, offscreen=False, vmin=None, vmax=None, cmap=None,
    view=(38.5, 70.5, 300, (-2.7, -12, 9.1)),
    ):
    """ Plot a 3D volume rendering view of the activation, with an
        outline of the brain.

        Parameters
        ----------
        map : 3D ndarray
            The activation map, as a 3D image.
        affine : 4x4 ndarray
            The affine matrix going from image voxel space to MNI space.
        cut_coords: 3-tuple of floats, optional
            The MNI coordinates of a 3D cursor to indicate a feature
            or a cut, in MNI coordinates and order.
        anat : 3D ndarray, optional
            The anatomical image to be used as a background. If None, the
            MNI152 T1 1mm template is used. If False, no anatomical
            image is used.
        anat_affine : 4x4 ndarray, optional
            The affine matrix going from the anatomical image voxel space to 
            MNI space. This parameter is not used when the default 
            anatomical is used, but it is compulsory when using an
            explicite anatomical image.
        threshold : float, optional
            The lower threshold of the positive activation. This
            parameter is used to threshold the activation map.
        offscreen: boolean, optional
            If True, Mayavi attempts to plot offscreen. Will work only
            with VTK >= 5.2.
        vmin : float, optional
            The minimal value, for the colormap
        vmax : float, optional
            The maximum value, for the colormap
        cmap : a callable, or a pylab colormap
            A callable returning a (n, 4) array for n values between
            0 and 1 for the colors. This can be for instance a pylab
            colormap.

        Notes
        -----

        If you are using a VTK version below 5.2, there is no way to
        avoid opening a window during the rendering under Linux. This is
        necessary to use the graphics card for the rendering. You must
        maintain this window on top of others and on the screen.

    """
    # Late import to avoid triggering wx imports before needed.
    try:
        from mayavi import mlab
    except ImportError:
        # Try out old install of Mayavi, with namespace packages
        from enthought.mayavi import mlab
    if offscreen:
        global off_screen_engine
        if off_screen_engine is None:
            try:
                from mayavi.core.off_screen_engine import OffScreenEngine
            except ImportError:
                # Try out old install of Mayavi, with namespace packages
                from enthought.mayavi.core.off_screen_engine import OffScreenEngine
            off_screen_engine = OffScreenEngine()
        off_screen_engine.start()
        fig = mlab.figure('__private_plot_map_3d__', 
                                bgcolor=(1, 1, 1), fgcolor=(0, 0, 0),
                                size=(400, 330),
                                engine=off_screen_engine)
        mlab.clf(figure=fig)
    else:
        fig = mlab.gcf()
        fig = mlab.figure(fig, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0),
                                                     size=(400, 350))
    disable_render = fig.scene.disable_render
    fig.scene.disable_render = True
    if threshold is None:
        threshold = stats.scoreatpercentile(
                                np.abs(map).ravel(), 80)
    contours = []
    lower_map = map[map <= -threshold]
    if np.any(lower_map):
        contours.append(lower_map.max())
    upper_map = map[map >= threshold]
    if np.any(upper_map):
        contours.append(map[map > threshold].min())


    ###########################################################################
    # Display the map using iso-surfaces
    if len(contours) > 0:
        map_src = affine_img_src(map, affine)
        module = mlab.pipeline.iso_surface(map_src,
                                        contours=contours,
                                        vmin=vmin, vmax=vmax)
        if hasattr(cmap, '__call__'):
            # Stick the colormap in mayavi
            module.module_manager.scalar_lut_manager.lut.table \
                    = (255*cmap(np.linspace(0, 1, 256))).astype(np.int)
    else:
        module = None

    if not anat is False:
        plot_anat_3d(anat=anat, anat_affine=anat_affine, scale=1.05,
                     outline_color=(.9, .9, .9),
                     gyri_opacity=.2)

    ###########################################################################
    # Draw the cursor
    if cut_coords is not None:
        x0, y0, z0 = cut_coords
        mlab.plot3d((-90, 90), (y0, y0), (z0, z0), 
                    color=(.5, .5, .5), tube_radius=0.25)
        mlab.plot3d((x0, x0), (-126, 91), (z0, z0), 
                    color=(.5, .5, .5), tube_radius=0.25)
        mlab.plot3d((x0, x0), (y0, y0), (-72, 109), 
                            color=(.5, .5, .5), tube_radius=0.25)

    mlab.view(*view)
    fig.scene.disable_render = disable_render
    
    return module


def demo_plot_map_3d():
    map = np.zeros((182, 218, 182))
    # Color a asymetric rectangle around Broca area:
    x, y, z = -52, 10, 22
    x_map, y_map, z_map = coord_transform(x, y, z, mni_sform_inv)
    map[x_map-5:x_map+5, y_map-3:y_map+3, z_map-10:z_map+10] = 1
    plot_map_3d(map, mni_sform, cut_coords=(x, y, z))


