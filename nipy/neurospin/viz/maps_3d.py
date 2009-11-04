"""
3D visualization of activation maps using Mayavi

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD


# Standard scientific libraries imports (more specific imports are
# delayed, so that the part module can be used without them).
import numpy as np

from enthought.mayavi import mlab
from enthought.mayavi.sources.api import ArraySource

# Local imports
from .anat_cache import mni_sform, mni_sform_inv, _AnatCache
from .coord_tools import coord_transform


################################################################################
# Helper functions
def affine_img_src(data, affine, scale=1, name='AffineImage'):
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

        Notes
        ------
        The affine should be diagonal.
    """
    center = np.r_[0, 0, 0, 1]
    spacing = np.diag(affine)[:3]
    # XXX: VTK orients X (negative spacing) in the wrong direction
    origin = np.dot(affine, center)[:3]
    spacing[0] *= -1
    origin[0] *= -1
    src = ArraySource(scalar_data=np.asarray(data), 
                           name=name,
                           spacing=scale*spacing,
                           origin=origin)
    return src 

################################################################################
# Anatomy outline
################################################################################

def plot_anat_3d(anat=None, anat_sform=None, scale=1):
    if anat is None:
        anat, anat_sform, anat_max = _AnatCache.get_anat()
    ###########################################################################
    # Display the cortical surface (flattenned)
    anat_src = affine_img_src(anat, anat_sform, scale=scale,
                                name='Anat')
    
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
    cortex.actor.property.backface_culling = True
    #cortex.actor.property.frontface_culling = True

    cortex.actor.mapper.interpolate_scalars_before_mapping = True
    cortex.actor.property.interpolation = 'flat'

    # Add opacity variation to the colormap
    cmap = cortex.module_manager.scalar_lut_manager.lut.table.to_array()
    cmap[128:, -1] = 0.7*255
    cortex.module_manager.scalar_lut_manager.lut.table = cmap
    return cortex
 

################################################################################
# Maps
################################################################################

def plot_map_3d(map, sform, cut_coords=None, anat=None, anat_sform=None,
    vmin=None, figure_num=None, mask=None, **kwargs):
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
            MNI152 T1 1mm template is used. If False, no anatomical
            image is used.
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
        kwargs: extra keyword arguments, optional
            The extra keyword arguments are passed to Mayavi's
            mlab.pipeline.volume

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
    fig = mlab.figure(figure_num, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0),
                                                     size=(400, 350))
    if figure_num is None:
        mlab.clf()
    fig.scene.disable_render = True
    
    center = np.r_[0, 0, 0, 1]

    ###########################################################################
    # Display the map using volume rendering
    map_src = affine_img_src(map, sform)
    vol = mlab.pipeline.volume(map_src, vmin=vmin, **kwargs)
   
    if not anat is False:
        plot_anat_3d(anat=anat, anat_sform=anat_sform, scale=1.05)
   
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


