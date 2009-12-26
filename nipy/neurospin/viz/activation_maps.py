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
import sys

# Standard scientific libraries imports (more specific imports are
# delayed, so that the part module can be used without them).
import numpy as np
import matplotlib as mp
import pylab as pl

# Local imports
from nipy.neurospin.utils.mask import compute_mask
from nipy.io.imageformats import load

from anat_cache import mni_sform, mni_sform_inv, _AnatCache
from coord_tools import coord_transform, find_activation, \
        find_cut_coords

from ortho_slicer import OrthoSlicer

class SformError(Exception):
    pass

class NiftiIndexError(IndexError):
    pass


################################################################################
# Colormaps

def _rotate_cmap(cmap, name=None, swap_order=('green', 'red', 'blue')):
    """ Utility function to swap the colors of a colormap.
    """
    orig_cdict = cmap._segmentdata.copy()

    cdict = dict()
    cdict['green'] = [(p, c1, c2)
                        for (p, c1, c2) in orig_cdict[swap_order[0]]]
    cdict['blue'] = [(p, c1, c2)
                        for (p, c1, c2) in orig_cdict[swap_order[1]]]
    cdict['red'] = [(p, c1, c2)
                        for (p, c1, c2) in orig_cdict[swap_order[2]]]

    if name is None:
        name = '%s_rotated' % cmap.name
    return mp.colors.LinearSegmentedColormap(name, cdict, 512)


def _pigtailed_cmap(cmap, name=None, 
                    swap_order=('green', 'red', 'blue')):
    """ Utility function to make a new colormap by concatenating a
        colormap with its reverse.
    """
    orig_cdict = cmap._segmentdata.copy()

    cdict = dict()
    cdict['green'] = [(0.5*(1-p), c1, c2)
                        for (p, c1, c2) in reversed(orig_cdict[swap_order[0]])]
    cdict['blue'] = [(0.5*(1-p), c1, c2)
                        for (p, c1, c2) in reversed(orig_cdict[swap_order[1]])]
    cdict['red'] = [(0.5*(1-p), c1, c2)
                        for (p, c1, c2) in reversed(orig_cdict[swap_order[2]])]

    for color in ('red', 'green', 'blue'):
        cdict[color].extend([(0.5*(1+p), c1, c2) 
                                    for (p, c1, c2) in orig_cdict[color]])

    if name is None:
        name = '%s_reversed' % cmap.name
    return mp.colors.LinearSegmentedColormap(name, cdict, 512)


# Using a dict as a namespace, to micmic matplotlib's cm

_cm = dict(
    cold_hot     = _pigtailed_cmap(pl.cm.hot,      name='cold_hot'),
    brown_blue   = _pigtailed_cmap(pl.cm.bone,     name='brown_blue'),
    cyan_copper  = _pigtailed_cmap(pl.cm.copper,   name='cyan_copper'),
    cyan_orange  = _pigtailed_cmap(pl.cm.YlOrBr_r, name='cyan_orange'),
    blue_red     = _pigtailed_cmap(pl.cm.Reds_r,   name='blue_red'),
    brown_cyan   = _pigtailed_cmap(pl.cm.Blues_r,  name='brown_cyan'),
    purple_green = _pigtailed_cmap(pl.cm.Greens_r, name='purple_green',
                    swap_order=('red', 'blue', 'green')),
    purple_blue  = _pigtailed_cmap(pl.cm.Blues_r, name='purple_blue',
                    swap_order=('red', 'blue', 'green')),
    blue_orange  = _pigtailed_cmap(pl.cm.Oranges_r, name='blue_orange',
                    swap_order=('green', 'red', 'blue')),
    black_blue   = _rotate_cmap(pl.cm.hot, name='black_blue'),
    black_purple = _rotate_cmap(pl.cm.hot, name='black_purple',
                                    swap_order=('blue', 'red', 'green')),
    black_pink   = _rotate_cmap(pl.cm.hot, name='black_pink',
                            swap_order=('blue', 'green', 'red')),
    black_green  = _rotate_cmap(pl.cm.hot, name='black_green',
                            swap_order=('red', 'blue', 'green')),
    black_red    = pl.cm.hot,
    )

_cm.update(pl.cm.datad)

class _CM(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__.update(self)


cm = _CM(**_cm)

################################################################################
# Helper functions for 2D plotting of activation maps 
################################################################################


def plot_map_2d(map, affine, cut_coords, anat=None, anat_affine=None,
                    figure_num=None, axes=None, title='',
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
        figure_num : integer, optional
            The number of the matplotlib figure used. If None is given, a
            new figure is created.
        axes : 4 tuple of float: (xmin, xmax, ymin, ymin), optional
            The coordinates, in matplotlib figure space, of the axes
            used to display the plot. If None, the complete figure is 
            used.
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
        Use masked arrays to create transparency.
    """
    if anat is None:
        anat, anat_affine, vmax_anat = _AnatCache.get_anat()
    if anat is not False:
        # Make anat only positive
        anat = anat - anat.min()
        # XXX: Do we want to keep this feature?

    # XXX: Should not always create a figure.
    # Lets take either a number or a figure instance.
    fig = pl.figure(figure_num, figsize=(6.6, 2.6), facecolor='w')
    if axes is None:
        # XXX: This should be named 'rect' and not 'axes'
        axes = (0., 1., 0., 1.)
        pl.clf()
        # XXX: clf? Bad bad bad.

    ortho_slicer = OrthoSlicer(cut_coords)
    anat_kwargs = kwargs.copy()
    anat_kwargs['cmap'] = pl.cm.gray
    ortho_slicer.plot_map(anat, anat_affine, **anat_kwargs)
    ortho_slicer.plot_map(map, affine, **kwargs)
    if annotate:
        ortho_slicer.annotate()
    if draw_cross:
        ortho_slicer.draw_cross(color='k')

    # XXX: What about title?
    if title:
        ortho_slicer.ax.text(0.01, 0.99, title, 
                    transform=ortho_slicer.ax.transAxes,
                    horizontalalignment='left',
                    verticalalignment='top',
                    size=15, color='w',
                    bbox=dict(boxstyle="square,pad=.3", 
                              ec="0", fc="0", alpha=.9),
                    )

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
    plot_map_2d(map, mni_sform, cut_coords=(x, y, z),
                title="Broca's area", figure_num=512)


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

    from .maps_3d import plot_map_3d, m2screenshot
    plot_map_3d(map, sform, cut_coords=cut_coords, anat=anat,
                anat_sform=anat_sform, vmin=vmin,
                figure_num=figure_num, mask=mask)

    fig = pl.figure(figure_num, figsize=(10.6, 2.6))
    ax = pl.axes((-0.01, 0, 0.3, 1))
    m2screenshot(mpl_axes=ax)

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
        
    nim = load(filename)
    sform = nim.get_affine()
    if any(np.linalg.eigvals(sform)==0):
        raise SformError, "sform affine is not inversible"
    if anat_filename is not None:
        anat_im = load(anat_filename)
        anat = anat_im.data
        anat_sform = anat_im.get_affine()
    else:
        anat = None
        anat_sform = None

    if mask_filename is not None:
        mask_im = load(mask_filename)
        mask = mask_im.data.astype(np.bool)
        if not np.allclose(mask_im.get_affine(), sform):
            raise SformError, 'Mask does not have same sform as image'
        if not np.allclose(mask.shape, nim.data.shape[:3]):
            raise NiftiIndexError, 'Mask does not have same shape as image'
    else:
        mask = None

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
    demo_plot_map_2d()
    pl.show()

