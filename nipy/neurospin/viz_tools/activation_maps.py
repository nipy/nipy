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
from nipy.neurospin.datasets import VolumeImg

from .anat_cache import mni_sform, mni_sform_inv, _AnatCache
from .coord_tools import coord_transform, find_cut_coords

from .ortho_slicer import OrthoSlicer


################################################################################
# Helper functions for 2D plotting of activation maps 
################################################################################
def _xyz_order(map, affine):
    img = VolumeImg(map, affine=affine, world_space='mine')
    img = img.xyz_ordered(resample=True, copy=False)
    map = img.get_data()
    affine = img.affine
    return map, affine


def _fast_abs_percentile(map):
    """ An algorithm to implement a fast version of the 80-percentile of
        the absolute value.
    """
    map = np.abs(map).ravel()
    map.sort()
    nb = map.size
    return map[.8*nb]


def plot_map(map, affine, cut_coords=None, anat=None, anat_affine=None,
                    figure=None, axes=None, title=None, threshold=None,
                    annotate=True, draw_cross=True, 
                    do3d=False, **kwargs):
    """ Plot three cuts of a given activation map (Frontal, Axial, and Lateral)

        Parameters
        ----------
        map : 3D ndarray
            The activation map, as a 3D image.
        affine : 4x4 ndarray
            The affine matrix going from image voxel space to MNI space.
        cut_coords: 3-tuple of floats or None
            The MNI coordinates of the point where the cut is performed, in 
            MNI coordinates and order.
            If None is given, the cut point is calculated automaticaly.
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
        threshold : a number, None, or 'auto'
            If None is given, the maps are not thresholded.
            If a number is given, it is used to threshold the maps:
            values below the threshold are plotted as transparent. If
            auto is given, the threshold is determined magically by
            analysis of the map.
        annotate: boolean, optional
            If annotate is True, positions and left/right annotation
            are added to the plot.
        draw_cross: boolean, optional
            If draw_cross is True, a cross is drawn on the plot to
            indicate the cut plosition.
        do3d: boolean, optional
            If True, Mayavi is used to plot a 3D view of the
            map in addition to the slicing.
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

    # Deal with automatic settings of plot parameters
    if threshold == 'auto':
        threshold = _fast_abs_percentile(map)  
    if cut_coords is None:
        x_map, y_map, z_map = find_cut_coords(map,
                                activation_threshold=threshold)
        cut_coords = coord_transform(x_map, y_map, z_map, affine)
    if threshold is not None:
        map = np.ma.masked_inside(map, -threshold, threshold, copy=False)
    
    if do3d:
        try:
            from enthought.mayavi import version
            if not int(version.version[0]) > 2:
                raise ImportError
        except ImportError:
            warnings.warn('Mayavi > 3.x not installed, plotting only 2D')
            do3d = False

    # Make sure that we have a figure
    if not isinstance(figure, Figure):
        if do3d:
            size = (10, 2.6)
        else:
            size = (6.6, 2.6)
        fig = pl.figure(figure, figsize=size, facecolor='w')
    else:
        fig = figure
        if isinstance(axes, Axes):
            assert axes.figure is figure, ("The axes passed are not "
            "in the figure")

    canonical_anat = False
    if anat is None:
        try:
            anat, anat_affine, vmax_anat = _AnatCache.get_anat()
            canonical_anat = True
        except OSError, e:
            anat = False
            warnings.warn(repr(e))


    # Use Mayavi for the 3D plotting
    if do3d:
        try:
            from .maps_3d import plot_map_3d, m2screenshot
            from enthought.tvtk.api import tvtk
            version = tvtk.Version()
            offscreen = True
            if (version.vtk_major_version, version.vtk_minor_version) < (5, 2):
                offscreen = False

            cmap = kwargs.get('cmap', pl.cm.cmap_d[pl.rcParams['image.cmap']])
            # Computing vmin and vmax is costly in time, and is needed
            # later, so we compute them now, and store them for future
            # use
            vmin = kwargs.get('vmin', map.min())
            kwargs['vmin'] = vmin
            vmax = kwargs.get('vmax', map.max())
            kwargs['vmax'] = vmax
            plot_map_3d(np.asarray(map), affine, cut_coords=cut_coords, 
                        anat=anat, anat_affine=anat_affine, 
                        offscreen=offscreen, cmap=cmap,
                        threshold=threshold,
                        vmin=vmin, vmax=vmax)

            ax = fig.add_axes((0.001, 0, 0.29, 1))
            ax.axis('off')
            m2screenshot(mpl_axes=ax)
            axes = (0.3, 0, .7, 1.)
            if offscreen:
                # Clean up, so that the offscreen engine doesn't become the
                # default
                from enthought.mayavi import mlab
                engine = mlab.get_engine()
                engine.close_scene(engine.current_scene)
                from enthought.mayavi.core.registry import registry
                for key, value in registry.engines.iteritems():
                    if value is engine:
                        registry.engines.pop(key)
                        break
        except ImportError:
            warnings.warn('Mayavi > 3.x not installed, plotting only 2D')

    if axes is None:
        axes = [0., 0., 1., 1.]
    if operator.isSequenceType(axes):
        axes = fig.add_axes(axes)
        axes.axis('off')

    ortho_slicer = OrthoSlicer(cut_coords, axes=axes)
    # Check that we should indeed plot an anat: we have one, and the
    # cut_coords are in its range
    x, y, z = cut_coords
    if (anat is not False 
                and np.all(
                 np.array(coord_transform(x, y, z, np.linalg.inv(anat_affine))) 
                            < anat.shape)):
        anat_kwargs = kwargs.copy()
        anat_kwargs['cmap'] = pl.cm.gray
        anat_kwargs.pop('alpha', 1.)
        if canonical_anat:
            # We special-case the 'canonical anat', as we don't need
            # to do a few transforms to it.
            anat_kwargs['vmin'] = 0
            anat_kwargs['vmax'] = vmax_anat
        else:
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


def demo_plot_map(do3d=False, **kwargs):
    """ Demo activation map plotting.
    """
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
    return plot_map(map, mni_sform, threshold='auto',
                        title="Broca's area", figure=512, do3d=do3d,
                        **kwargs)


