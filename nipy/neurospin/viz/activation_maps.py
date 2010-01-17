#!/usr/bin/env python

"""
Functions to do automatic visualization of activation-like maps.

For 2D-only visualization, only matplotlib is required.
For 3D visualization, Mayavi, version 3.0 or greater, is required.

For a demo, see the 'demo_plot_map_2d' function.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD

# Standard library imports
import warnings
import operator

# Standard scientific libraries imports (more specific imports are
# delayed, so that the part module can be used without them).
import numpy as np
import pylab as pl
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# Local imports
from nipy.neurospin.utils.mask import compute_mask
from nipy.neurospin.datasets import VolumeImg

from .anat_cache import mni_sform, mni_sform_inv, _AnatCache
from .coord_tools import coord_transform, find_activation, \
        find_cut_coords

from .ortho_slicer import OrthoSlicer

from . import cm

################################################################################
# Helper functions for 2D plotting of activation maps 
################################################################################
def _xyz_order(map, affine):
    img = VolumeImg(map, affine=affine, world_space='mine')
    img = img.xyz_ordered()
    map = img.get_data()
    affine = img.affine
    return map, affine


def plot_map_2d(map, affine, cut_coords, anat=None, anat_affine=None,
                    figure=None, axes=None, title=None,
                    annotate=True, draw_cross=True, **kwargs):
    """ Plot three cuts of a given activation map (Frontal, Axial, and Lateral)

        Parameters
        ----------
        map : 3D ndarray
            The activation map, as a 3D image.
        affine : 4x4 ndarray
            The affine matrix going from image voxel space to MNI space.
        cut_coords: 3-tuple of floats
            The MNI coordinates of the point where the cut is performed, in 
            MNI coordinates and order.
        anat : 3D ndarray or False, optional
            The anatomical image to be used as a background. If None, the 
            MNI152 T1 1mm template is used. If False, no anat is displayed.
        anat_affine : 4x4 ndarray, optional
            The affine matrix going from the anatomical image voxel space to 
            MNI space. This parameter is not used when the default 
            anatomical is used, but it is compulsory when using an
            explicite anatomical image.
        figure : integer or matplotlib figure, optional
            Matplotlib figure used or its number. If None is given, a
            new figure is created.
        axes : matplotlib axes or 4 tuple of float: (xmin, xmax, ymin, ymin), optional
            The axes, or the coordinates, in matplotlib figure space, 
            of the axes used to display the plot. If None, the complete 
            figure is used.
        title : string, optional
            The title dispayed on the figure.
        annotate: boolean, optional
            If annotate is True, positions and left/right annotation
            are added to the plot.
        draw_cross: boolean, optional
            If draw_cross is True, a cross is drawn on the plot to
            indicate the cut plosition.
        kwargs: extra keyword arguments, optional
            Extra keyword arguments passed to pylab.imshow

        Notes
        -----
        Arrays should be passed in numpy convention: (x, y, z)
        ordered.

        Use masked arrays to create transparency:

            import numpy as np
            map = np.ma.masked_less(map, 0.5)
            plot_map(map, affine)
    """
    map, affine = _xyz_order(map, affine)
    if anat is None:
        try:
            anat, anat_affine, vmax_anat = _AnatCache.get_anat()
        except OSError, e:
            anat = False
            warnings.warn(repr(e))
    if not isinstance(figure, Figure):
        fig = pl.figure(figure, figsize=(6.6, 2.6), facecolor='w')
    else:
        fig = figure
        if isinstance(axes, Axes):
            assert axes.figure is figure, ("The axes passed are not "
            "in the figure")
    if axes is None:
        axes = [0., 0., 1., 1.]
    if operator.isSequenceType(axes):
        axes = fig.add_axes(axes)
        axes.axis('off')

    ortho_slicer = OrthoSlicer(cut_coords, axes=axes)
    if anat is not False:
        anat_kwargs = kwargs.copy()
        anat_kwargs['cmap'] = pl.cm.gray
        anat_kwargs.pop('vmin', None)
        anat_kwargs.pop('vmax', None)
        anat, anat_affine = _xyz_order(anat, anat_affine)
        ortho_slicer.plot_map(anat, anat_affine, **anat_kwargs)
    ortho_slicer.plot_map(map, affine, **kwargs)
    if annotate:
        ortho_slicer.annotate()
    if draw_cross:
        ortho_slicer.draw_cross(color='k')

    if title is not None and not title == '':
        ortho_slicer.title(title)
    return ortho_slicer


def demo_plot_map_2d():
    map = np.zeros((182, 218, 182))
    # Color a asymetric rectangle around Broca area:
    x, y, z = -52, 10, 22
    x_map, y_map, z_map = coord_transform(x, y, z, mni_sform_inv)
    # Compare to values obtained using fslview. We need to add one as
    # voxels do not start at 0 in fslview.
    assert x_map == 142
    assert y_map +1 == 137
    assert z_map +1 == 95
    map[x_map-5:x_map+5, y_map-3:y_map+3, z_map-10:z_map+10] = 1
    map = np.ma.masked_less(map, 0.5)
    return plot_map_2d(map, mni_sform, cut_coords=(x, y, z),
                        title="Broca's area", figure=512)


def plot_map(map, affine, cut_coords, anat=None, anat_affine=None,
    figure=None, title=None, **kwargs):
    """ Plot a together a 3D volume rendering view of the activation, with an
        outline of the brain, and 2D cuts. If Mayavi is not installed,
        falls back to 2D views only.

        Parameters
        ----------
        map : 3D ndarray
            The activation map, as a 3D image.
        affine : 4x4 ndarray
            The affine matrix going from image voxel space to MNI space.
        cut_coords: 3-tuple of floats, optional
            The MNI coordinates of the cut to perform, in MNI coordinates 
            and order. If None is given, the cut_coords are automaticaly
            estimated.
        anat : 3D ndarray, optional
            The anatomical image to be used as a background. If None, the 
            MNI152 T1 1mm template is used.
        anat_affine : 4x4 ndarray, optional
            The affine matrix going from the anatomical image voxel space to 
            MNI space. This parameter is not used when the default 
            anatomical is used, but it is compulsory when using an
            explicite anatomical image.
        figure : integer or matplotlib figure, optional
            Matplotlib figure used or its number. If None is given, a
            new figure is created.
        title : string, optional
            The title dispayed on the figure.

        Notes
        -----
        Arrays should be passed in numpy convention: (x, y, z)
        ordered.

        Use masked arrays to create transparency::

            import numpy as np
            map = np.ma.masked_less(map, 0.5)
            plot_map(map, affine)
    """
    try:
        from enthought.mayavi import version
        if not int(version.version[0]) > 2:
            raise ImportError
    except ImportError:
        warnings.warn('Mayavi > 3.x not installed, plotting only 2D')
        return plot_map_2d(map, affine, cut_coords=cut_coords, anat=anat,
                           anat_affine=anat_affine, 
                           figure=figure, 
                           title=title, **kwargs)


    from .maps_3d import plot_map_3d, m2screenshot
    if not isinstance(figure, Figure):
        figure = pl.figure(figure, figsize=(10.6, 2.6), facecolor='w')

    plot_map_3d(np.asarray(map), affine, cut_coords=cut_coords, anat=anat,
                anat_affine=anat_affine, offscreen=True, **kwargs)

    ax = figure.add_axes((0.001, 0, 0.29, 1))
    ax.axis('off')
    m2screenshot(mpl_axes=ax)

    return plot_map_2d(map, affine, cut_coords=cut_coords, anat=anat,
                        anat_affine=anat_affine, 
                        figure=figure, axes=(0.3, 0, .7, 1.),
                        title=title, **kwargs)


def demo_plot_map():
    map = np.zeros((182, 218, 182))
    # Color a asymetric rectangle around Broca area:
    x, y, z = -52, 10, 22
    x_map, y_map, z_map = coord_transform(x, y, z, mni_sform_inv)
    map[x_map-30:x_map+30, y_map-3:y_map+3, z_map-10:z_map+10] = 1
    map = np.ma.masked_less(map, 0.5)
    return plot_map(map, mni_sform, cut_coords=(x, y, z),
                            title="Broca's area")


def auto_plot_map(map, affine, threshold=None, cut_coords=None, do3d=False, 
                    anat=None, anat_affine=None, title=None, mask=None,
                    figure_num=None, auto_sign=True):
    """ Automatic plotting of an activation map.

        Plot a together a 3D volume rendering view of the activation, with an
        outline of the brain, and 2D cuts. If Mayavi is not installed,
        falls back to 2D views only.

        Parameters
        ----------
        map : 3D ndarray
            The activation map, as a 3D image.
        affine : 4x4 ndarray
            The affine matrix going from image voxel space to MNI space.
        threshold : float, optional
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
        anat_affine : 4x4 ndarray, optional
            The affine matrix going from the anatomical image voxel space to 
            MNI space. This parameter is not used when the default 
            anatomical is used, but it is compulsory when using an
            explicite anatomical image.
        title : string, optional
            The title dispayed on the figure.
        mask : 3D ndarray, optional
            Boolean array of the voxels used.
        figure_num : integer, optional
            The number of the matplotlib and Mayavi figures used. If None is 
            given, a new figure is created.
        auto_sign : boolean, optional
            If auto_sign is True, the sign of the activation is
            automaticaly computed: negative activation can thus be
            plotted.

        Returns
        -------
        threshold : float
            The lower threshold of the activation used.
        cut_coords : 3-tuple of floats
            The Talairach coordinates of the cut performed for the 2D
            view.

        Notes
        -----
        Arrays should be passed in numpy convention: (x, y, z)
        ordered.

        Use masked arrays to create transparency:

            import numpy as np
            map = np.ma.masked_less(map, 0.5)
            plot_map(map, affine)

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
    else:
        mask = mask.astype(np.bool)
    if threshold is None:
        threshold = np.inf
        pvalue = 0.04
        while not np.isfinite(threshold):
            pvalue *= 1.25
            vmax, threshold = find_activation(map, mask=mask, pvalue=pvalue)
            if not np.isfinite(threshold) and auto_sign:
                if np.isfinite(vmax):
                    threshold = -vmax
                    if mask is not None:
                        map[mask] *= -1
                    else:
                        map *= -1
    if cut_coords is None:
        x_map, y_map, z_map = find_cut_coords(map,
                                activation_threshold=threshold)
        cut_coords = coord_transform(x_map, y_map, z_map, affine)
    map = np.ma.masked_less(map, threshold)
    plotter(map, affine, cut_coords=cut_coords,
                anat=anat, anat_affine=anat_affine, title=title,
                figure_num=figure_num)
    return threshold, cut_coords


