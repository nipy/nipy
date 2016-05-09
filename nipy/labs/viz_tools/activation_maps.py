#!/usr/bin/env python

"""
Functions to do automatic visualization of activation-like maps.

For 2D-only visualization, only matplotlib is required.
For 3D visualization, Mayavi, version 3.0 or greater, is required.

For a demo, see the 'demo_plot_map' function.

"""
from __future__ import absolute_import

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD

# Standard library imports
import warnings
import numbers

# Standard scientific libraries imports (more specific imports are
# delayed, so that the part module can be used without them).
import numpy as np

from nipy.utils.skip_test import skip_if_running_nose
from nipy.utils import is_numlike

# Import pylab
try:
    import pylab as pl
except ImportError:
    skip_if_running_nose('Could not import matplotlib')

from .anat_cache import mni_sform, mni_sform_inv, _AnatCache
from .coord_tools import (coord_transform,
                          find_maxsep_cut_coords
                          )

from .slicers import SLICERS, _xyz_order
from .edge_detect import _fast_abs_percentile

################################################################################
# Helper functions for 2D plotting of activation maps
################################################################################


def plot_map(map, affine, cut_coords=None, anat=None, anat_affine=None,
             slicer='ortho',
             figure=None, axes=None, title=None,
             threshold=None, annotate=True, draw_cross=True,
             do3d=False, threshold_3d=None,
             view_3d=(38.5, 70.5, 300, (-2.7, -12, 9.1)),
             black_bg=False, **kwargs):
    """ Plot three cuts of a given activation map (Frontal, Axial, and Lateral)

        Parameters
        ----------
        map : 3D ndarray
            The activation map, as a 3D image.
        affine : 4x4 ndarray
            The affine matrix going from image voxel space to MNI space.
        cut_coords: None, int, or a tuple of floats
            The MNI coordinates of the point where the cut is performed, in
            MNI coordinates and order.
            If slicer is 'ortho', this should be a 3-tuple: (x, y, z)
            For slicer == 'x', 'y', or 'z', then these are the
            coordinates of each cut in the corresponding direction.
            If None or an int is given, then a maximally separated sequence (
            with exactly cut_coords elements if cut_coords is not None) of
            cut coordinates along the slicer axis is computed automatically
        anat : 3D ndarray or False, optional
            The anatomical image to be used as a background. If None, the
            MNI152 T1 1mm template is used. If False, no anat is displayed.
        anat_affine : 4x4 ndarray, optional
            The affine matrix going from the anatomical image voxel space to
            MNI space. This parameter is not used when the default
            anatomical is used, but it is compulsory when using an
            explicite anatomical image.
        slicer: {'ortho', 'x', 'y', 'z'}
            Choose the direction of the cuts. With 'ortho' three cuts are
            performed in orthogonal directions
        figure : integer or matplotlib figure, optional
            Matplotlib figure used or its number. If None is given, a
            new figure is created.
        axes : matplotlib axes or 4 tuple of float: (xmin, ymin, width, height), optional
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
        do3d: {True, False or 'interactive'}, optional
            If True, Mayavi is used to plot a 3D view of the
            map in addition to the slicing. If 'interactive', the
            3D visualization is displayed in an additional interactive
            window.
        threshold_3d:
            The threshold to use for the 3D view (if any). Defaults to
            the same threshold as that used for the 2D view.
        view_3d: tuple,
            The view used to take the screenshot: azimuth, elevation,
            distance and focalpoint, see the docstring of mlab.view.
        black_bg: boolean, optional
            If True, the background of the image is set to be black. If
            you whish to save figures with a black background, you
            will need to pass "facecolor='k', edgecolor='k'" to pylab's
            savefig.
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

    nan_mask = np.isnan(np.asarray(map))
    if np.any(nan_mask):
        map = map.copy()
        map[nan_mask] = 0
    del nan_mask

    # Deal with automatic settings of plot parameters
    if threshold == 'auto':
        # Threshold epsilon above a percentile value, to be sure that some
        # voxels are indeed threshold
        threshold = _fast_abs_percentile(map) + 1e-5

    if do3d:
        try:
            try:
                from mayavi import version
            except ImportError:
                from enthought.mayavi import version
            if not int(version.version[0]) > 2:
                raise ImportError
        except ImportError:
            warnings.warn('Mayavi > 3.x not installed, plotting only 2D')
            do3d = False

    if (cut_coords is None or isinstance(cut_coords, numbers.Number)
        ) and slicer in ['x', 'y', 'z']:
        cut_coords = find_maxsep_cut_coords(map, affine, slicer=slicer,
                                            threshold=threshold,
                                            n_cuts=cut_coords)

    slicer = SLICERS[slicer].init_with_figure(data=map, affine=affine,
                                              threshold=threshold,
                                              cut_coords=cut_coords,
                                              figure=figure, axes=axes,
                                              black_bg=black_bg,
                                              leave_space=do3d)

    # Use Mayavi for the 3D plotting
    if do3d:
        from .maps_3d import plot_map_3d, m2screenshot
        try:
            from tvtk.api import tvtk
        except ImportError:
            from enthought.tvtk.api import tvtk
        version = tvtk.Version()
        offscreen = True
        if (version.vtk_major_version, version.vtk_minor_version) < (5, 2):
            offscreen = False
        if do3d == 'interactive':
            offscreen = False

        cmap = kwargs.get('cmap', pl.cm.cmap_d[pl.rcParams['image.cmap']])
        # Computing vmin and vmax is costly in time, and is needed
        # later, so we compute them now, and store them for future
        # use
        vmin = kwargs.get('vmin', map.min())
        kwargs['vmin'] = vmin
        vmax = kwargs.get('vmax', map.max())
        kwargs['vmax'] = vmax
        try:
            from mayavi import mlab
        except ImportError:
            from enthought.mayavi import mlab
        if threshold_3d is None:
            threshold_3d = threshold
        plot_map_3d(np.asarray(map), affine, cut_coords=cut_coords,
                    anat=anat, anat_affine=anat_affine, 
                    offscreen=offscreen, cmap=cmap,
                    threshold=threshold_3d,
                    view=view_3d,
                    vmin=vmin, vmax=vmax)

        ax = list(slicer.axes.values())[0].ax.figure.add_axes((0.001, 0, 0.29, 1))
        ax.axis('off')
        m2screenshot(mpl_axes=ax)
        if offscreen:
            # Clean up, so that the offscreen engine doesn't become the
            # default
            mlab.clf()
            engine = mlab.get_engine()
            try:
                from mayavi.core.registry import registry
            except:
                from enthought.mayavi.core.registry import registry
            for key, value in registry.engines.items():
                if value is engine:
                    registry.engines.pop(key)
                    break


    if threshold:
        map = np.ma.masked_inside(map, -threshold, threshold, copy=False)


    _plot_anat(slicer, anat, anat_affine, title=title,
               annotate=annotate, draw_cross=draw_cross)

    slicer.plot_map(map, affine, **kwargs)
    return slicer


def _plot_anat(slicer, anat, anat_affine, title=None,
               annotate=True, draw_cross=True, dim=False, cmap=pl.cm.gray):
    """ Internal function used to plot anatomy
    """
    canonical_anat = False
    if anat is None:
        try:
            anat, anat_affine, vmax_anat = _AnatCache.get_anat()
            canonical_anat = True
        except OSError as e:
            anat = False
            warnings.warn(repr(e))

    black_bg = slicer._black_bg
    # XXX: Check that we should indeed plot an anat: we have one, and the
    # cut_coords are in its range

    if anat is not False:
        if canonical_anat:
            # We special-case the 'canonical anat', as we don't need
            # to do a few transforms to it.
            vmin = 0
            vmax = vmax_anat
        elif dim:
            vmin = anat.min()
            vmax = anat.max()
        else:
            vmin = None
            vmax = None
            anat, anat_affine = _xyz_order(anat, anat_affine)
        if dim:
            vmean = .5*(vmin + vmax)
            ptp = .5*(vmax - vmin)
            if not is_numlike(dim):
                dim = .6
            if black_bg:
                vmax = vmean + (1+dim)*ptp
            else:
                vmin = vmean - (1+dim)*ptp
        slicer.plot_map(anat, anat_affine, cmap=cmap,
                              vmin=vmin, vmax=vmax)

        if annotate:
            slicer.annotate()
        if draw_cross:
            slicer.draw_cross()

    if black_bg:
        # To have a black background in PDF, we need to create a
        # patch in black for the background
        for ax in slicer.axes.values():
            ax.ax.imshow(np.zeros((2, 2, 3)),
                         extent=[-5000, 5000, -5000, 5000],
                         zorder=-500)

    if title is not None and not title == '':
        slicer.title(title)
    return slicer


def plot_anat(anat=None, anat_affine=None, cut_coords=None, slicer='ortho',
              figure=None, axes=None, title=None, annotate=True,
              draw_cross=True, black_bg=False, dim=False, cmap=pl.cm.gray):
    """ Plot three cuts of an anatomical image (Frontal, Axial, and Lateral)

        Parameters
        ----------
        anat : 3D ndarray, optional
            The anatomical image to be used as a background. If None is
            given, nipy tries to find a T1 template.
        anat_affine : 4x4 ndarray, optional
            The affine matrix going from the anatomical image voxel space to 
            MNI space. This parameter is not used when the default 
            anatomical is used, but it is compulsory when using an
            explicite anatomical image.
        figure : integer or matplotlib figure, optional
            Matplotlib figure used or its number. If None is given, a
            new figure is created.
        cut_coords: None, or a tuple of floats
            The MNI coordinates of the point where the cut is performed, in
            MNI coordinates and order.
            If slicer is 'ortho', this should be a 3-tuple: (x, y, z)
            For slicer == 'x', 'y', or 'z', then these are the
            coordinates of each cut in the corresponding direction.
            If None is given, the cuts is calculated automaticaly.
        slicer: {'ortho', 'x', 'y', 'z'}
            Choose the direction of the cuts. With 'ortho' three cuts are
            performed in orthogonal directions
        figure : integer or matplotlib figure, optional
            Matplotlib figure used or its number. If None is given, a
            new figure is created.
        axes : matplotlib axes or 4 tuple of float: (xmin, ymin, width, height), optional
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
        black_bg: boolean, optional
            If True, the background of the image is set to be black. If
            you whish to save figures with a black background, you
            will need to pass "facecolor='k', edgecolor='k'" to pylab's
            savefig.
        dim: float, optional
            If set, dim the anatomical image, such that
            vmax = vmean + (1+dim)*ptp if black_bg is set to True, or
            vmin = vmean - (1+dim)*ptp otherwise, where
            ptp = .5*(vmax - vmin)
        cmap: matplotlib colormap, optional
            The colormap for the anat

        Notes
        -----
        Arrays should be passed in numpy convention: (x, y, z)
        ordered.
    """
    slicer = SLICERS[slicer].init_with_figure(data=anat, affine=anat_affine,
                                          threshold=0, cut_coords=cut_coords,
                                          figure=figure, axes=axes,
                                          black_bg=black_bg)

    _plot_anat(slicer, anat, anat_affine, title=title,
               annotate=annotate, draw_cross=draw_cross, dim=dim, cmap=cmap)
    return slicer


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
                        title="Broca's area", do3d=do3d,
                        **kwargs)
