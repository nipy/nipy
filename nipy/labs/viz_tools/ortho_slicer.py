# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
vmax = None
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
The OrthoSlicer class.

The main purpose of this class is to have auto adjust of axes size to
the data.
"""
import warnings

import numpy as np
from scipy import ndimage, signal

import pylab as pl
import matplotlib as mp
from matplotlib.transforms import Bbox

# Local imports
from .coord_tools import coord_transform, get_bounds, get_mask_bounds
from . import cm
from ..datasets import VolumeImg

try:
    # Protect this import as it is compiled code
    from nipy.labs.utils.routines import quantile
except ImportError, e:
    warnings.warn('Could not import fast quantile function: %s' % e)
    quantile = None

################################################################################
# Bugware to have transparency work OK with MPL < .99.1
if mp.__version__ < '0.99.1':
    # We wrap the lut as a callable and replace its evalution to put
    # alpha to zero where the mask is true. This is what is done in 
    # MPL >= .99.1
    from matplotlib import colors
    class CMapProxy(colors.Colormap):
        def __init__(self, lut):
            self.__lut = lut

        def __call__(self, arr, *args, **kwargs):
            results = self.__lut(arr, *args, **kwargs)
            if not isinstance(arr, np.ma.MaskedArray):
                return results
            else:
                results[arr.mask, -1] = 0
            return results

        def __getattr__(self, attr):
            # Dark magic: we are delegating any call to the lut instance
            # we wrap
            return self.__dict__.get(attr, getattr(self.__lut, attr))


def _xyz_order(map, affine):
    img = VolumeImg(map, affine=affine, world_space='mine')
    img = img.xyz_ordered(resample=True, copy=False)
    map = img.get_data()
    affine = img.affine
    return map, affine
                    
################################################################################
# Edge detection

def _fast_abs_percentile(map, percentile=80):
    """ A fast version of the percentile of the absolute value.
    """
    if hasattr(map, 'mask'):
        map = np.asarray(map[np.logical_not(map.mask)])
    map = np.abs(map)
    map = map.ravel()
    if quantile is not None:
        return quantile(map, .01*percentile)
    map.sort()
    nb = map.size
    return map[.01*percentile*nb]


def _orientation_kernel(t):
    """ structure elements for calculating the value of neighbors in several 
        directions
    """ 
    sin = np.sin
    pi  = np.pi
    t = pi*t
    arr =  np.array([[sin(t),        sin(t+.5*pi),  sin(t+pi)    ],
                     [sin(t+1.5*pi), 0,             sin(t+1.5*pi)],
                     [sin(t+pi),     sin(t+.5*pi),  sin(t)       ]])
    return np.round(.5*((1+arr))**2).astype(np.bool)


def _edge_detect(image, high_threshold=.75, low_threshold=.4):
    """ Edge detection for 2D images based on Canny filtering.

        Parameters
        ==========
        image: 2D array
            The image on which edge detection is applied
        high_threshold: float, optional
            The quantile defining the upper threshold of the hysteries 
            thresholding: decrease this to keep more edges
        low_threshold: float, optional
            The quantile defining the lower threshold of the hysteries 
            thresholding: decrease this to extract wider edges

        Returns
        ========
        grad_mag: 2D array of floats
            The magnitude of the gradient
        edge_mask: 2D array of booleans
            A mask of where have edges been detected

        Notes
        ======
        This function is based on a Canny filter, however it has been
        taylored to visualization purposes on brain images: don't use it
        in the general case.

        It computes the norm of the gradient, extracts the ridge by
        keeping only local maximum in each direction, and performs
        hysteresis filtering to keep only edges with high gradient
        magnitude.
    """
    # This code is loosely based on code by Stefan van der Waalt
    # Convert to floats to avoid overflows
    np_err = np.seterr(all='ignore')
    img = signal.wiener(image.astype(np.float))
    np.seterr(**np_err)
    # Where the noise variance is 0, Wiener can create nans
    img[np.isnan(img)] = image[np.isnan(img)]
    img /= img.max()
    grad_x = ndimage.sobel(img, mode='constant', axis=0)
    grad_y = ndimage.sobel(img, mode='constant', axis=1)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_angle = np.arctan2(grad_y, grad_x)
    # Scale the angles in the range [0, 2]
    grad_angle = (grad_angle + np.pi) / np.pi
    # Non-maximal suppression: an edge pixel is only good if its magnitude is
    # greater than its neighbors normal to the edge direction.
    thinner = np.zeros(grad_mag.shape, dtype=np.bool)
    for angle in np.arange(0, 2, .25):
        thinner = thinner | (
                (grad_mag > .85*ndimage.maximum_filter(grad_mag,
                                    footprint=_orientation_kernel(angle))) 
                & (((grad_angle - angle) % 2) < .75)
               )
    # Remove the edges next to the side of the image: they are not reliable
    thinner[0]     = 0
    thinner[-1]    = 0
    thinner[:, 0]  = 0
    thinner[:, -1] = 0

    thinned_grad = thinner * grad_mag
    # Hysteresis thresholding: find seeds above a high threshold, then
    # expand out until we go below the low threshold
    grad_values = thinned_grad[thinner]
    high = thinned_grad > _fast_abs_percentile(grad_values, 100*high_threshold)
    low = thinned_grad >  _fast_abs_percentile(grad_values, 100*low_threshold)
    edge_mask = ndimage.binary_dilation(high, structure=np.ones((3, 3)), 
                                        iterations=-1, mask=low)
    return grad_mag, edge_mask


def _edge_map(image):
    """ Return a maps of edges suitable for visualization.

        Parameters
        ==========
        image: 2D array
            The image that the edges are extracted from.

        Returns
        ========
        edge_mask: 2D masked array
            A mask of the edge as a masked array with parts without
            edges masked and the large extents detected with lower
            coefficients.
    """
    edge_mask = _edge_detect(image)[-1]
    edge_mask = edge_mask.astype(np.float)
    edge_mask = -np.sqrt(ndimage.distance_transform_cdt(edge_mask))
    edge_mask[edge_mask != 0] -= -.05+edge_mask.min()
    edge_mask = np.ma.masked_less(edge_mask, .01)
    return edge_mask

################################################################################
# class OrthoSlicer
################################################################################

class OrthoSlicer(object):
    """ A class to create 3 linked axes for plotting orthogonal
        cuts of 3D maps.

        Attributes
        ----------

        axes: dictionnary of axes
            The 3 axes used to plot each view.
        frame_axes: axes
            The axes framing the whole set of views.

        Notes
        -----

        The extent of the different axes are adjusted to fit the data
        best in the viewing area.
    """

    def __init__(self, cut_coords, axes=None, black_bg=False):
        """ Create 3 linked axes for plotting orthogonal cuts.

            Parameters
            ----------
            cut_coords: 3 tuple of ints
                The cut position, in world space.
            axes: matplotlib axes object, optional
                The axes that will be subdivided in 3.
            black_bg: boolean, optional
                If True, the background of the figure will be put to
                black. If you whish to save figures with a black background, 
                you will need to pass "facecolor='k', edgecolor='k'" to 
                pylab's savefig.

        """
        self._cut_coords = cut_coords
        if axes is None:
            axes = pl.axes((0., 0., 1., 1.))
            axes.axis('off')
        self.frame_axes = axes
        axes.set_zorder(1)
        bb = axes.get_position()
        self.rect = (bb.x0, bb.y0, bb.x1, bb.y1)
        x0, y0, x1, y1 = self.rect
        self._object_bounds = dict()
        self._black_bg = black_bg

        # Create our axes:
        self.axes = dict()
        for index, name in enumerate(('x', 'y', 'z')):
            ax = pl.axes([0.3*index*(x1-x0) + x0, y0, .3*(x1-x0), y1-y0])
            ax.axis('off')
            self.axes[name] = ax
            ax.set_axes_locator(self._locator)
            self._object_bounds[ax] = list()


    def _get_object_bounds(self, ax):
        """ Return the bounds of the objects on one axes.
        """
        xmins, xmaxs, ymins, ymaxs = np.array(self._object_bounds[ax]).T
        xmax = max(xmaxs.max(), xmins.max())
        xmin = min(xmins.min(), xmaxs.min())
        ymax = max(ymaxs.max(), ymins.max())
        ymin = min(ymins.min(), ymaxs.min())
        return xmin, xmax, ymin, ymax


    def _locator(self, axes, renderer):
        """ The locator function used by matplotlib to position axes.
            Here we put the logic used to adjust the size of the axes.
        """
        x0, y0, x1, y1 = self.rect
        width_dict = dict()
        ax_dict = self.axes
        x_ax = ax_dict['x']
        y_ax = ax_dict['y']
        z_ax = ax_dict['z']
        for ax in ax_dict.itervalues():
            xmin, xmax, ymin, ymax = self._get_object_bounds(ax)
            width_dict[ax] = (xmax - xmin)
        total_width = float(sum(width_dict.values()))
        for ax, width in width_dict.iteritems():
            width_dict[ax] = width/total_width*(x1 -x0)
        left_dict = dict()
        left_dict[x_ax] = x0
        left_dict[y_ax] = x0 + width_dict[x_ax]
        left_dict[z_ax] = x0 + width_dict[x_ax] + width_dict[y_ax]
        return Bbox([[left_dict[axes], y0], 
                     [left_dict[axes] + width_dict[axes], y1]])


    def draw_cross(self, cut_coords=None, **kwargs):
        """ Draw a crossbar on the plot to show where the cut is
            performed.

            Parameters
            ----------
            cut_coords: 3-tuple of floats, optional
                The position of the cross to draw. If none is passed, the 
                ortho_slicer's cut coordinnates are used.
            kwargs:
                Extra keyword arguments are passed to axhline
        """
        if cut_coords is None:
            cut_coords = self._cut_coords
        x, y, z = cut_coords
        kwargs = kwargs.copy()
        if not 'color' in kwargs:
            if self._black_bg:
                kwargs['color'] = '.8'
            else:
                kwargs['color'] = 'k'
        ax = self.axes['x']
        ax.axvline(x, ymin=.05, ymax=.95, **kwargs)
        ax.axhline(z, **kwargs)

        ax = self.axes['y']
        xmin, xmax, ymin, ymax = self._get_object_bounds(ax)
        ax.axvline(y, ymin=.05, ymax=.95, **kwargs)
        ax.axhline(z, xmax=.95, **kwargs)

        ax = self.axes['z']
        ax.axvline(x, ymin=.05, ymax=.95, **kwargs)
        ax.axhline(y, **kwargs)


    def annotate(self, left_right=True, positions=True, size=12, **kwargs):
        """ Add annotations to the plot.

            Parameters
            ----------
            left_right: boolean, optional
                If left_right is True, annotations indicating which side
                is left and which side is right are drawn.
            positions: boolean, optional
                If positions is True, annotations indicating the
                positions of the cuts are drawn.
            size: integer, optional
                The size of the text used.
            kwargs:
                Extra keyword arguments are passed to matplotlib's text
                function.
        """
        kwargs = kwargs.copy()
        if not 'color' in kwargs:
            if self._black_bg:
                kwargs['color'] = 'w'
            else:
                kwargs['color'] = 'k'

        bg_color = ('k' if self._black_bg else 'w')
        if left_right:
            ax_z = self.axes['z']
            ax_z.text(.1, .95, 'L', 
                    transform=ax_z.transAxes,
                    horizontalalignment='left',
                    verticalalignment='top',
                    size=size,
                    bbox=dict(boxstyle="square,pad=0", 
                              ec=bg_color, fc=bg_color, alpha=.8),
                    **kwargs)

            ax_z.text(.9, .95, 'R', 
                    transform=ax_z.transAxes,
                    horizontalalignment='right',
                    verticalalignment='top',
                    size=size,
                    bbox=dict(boxstyle="square,pad=0", 
                              ec=bg_color, fc=bg_color, alpha=.8),
                    **kwargs)

            ax_x = self.axes['x']
            ax_x.text(.1, .95, 'L', 
                    transform=ax_x.transAxes,
                    horizontalalignment='left',
                    verticalalignment='top',
                    size=size,
                    bbox=dict(boxstyle="square,pad=0", 
                              ec=bg_color, fc=bg_color, alpha=.8),
                    **kwargs)
            ax_x.text(.9, .95, 'R', 
                    transform=ax_x.transAxes,
                    horizontalalignment='right',
                    verticalalignment='top',
                    size=size,
                    bbox=dict(boxstyle="square,pad=0", 
                              ec=bg_color, fc=bg_color, alpha=.8),
                    **kwargs)

        if positions:
            x, y, z  = self._cut_coords
            ax_x.text(0, 0, 'y=%i' % y,
                    transform=ax_x.transAxes,
                    horizontalalignment='left',
                    verticalalignment='bottom',
                    size=size,
                    bbox=dict(boxstyle="square,pad=0", 
                              ec=bg_color, fc=bg_color, alpha=.9),
                    **kwargs)
            ax_y = self.axes['y']
            ax_y.text(0, 0, 'x=%i' % x,
                    transform=ax_y.transAxes,
                    horizontalalignment='left',
                    verticalalignment='bottom',
                    size=size,
                    bbox=dict(boxstyle="square,pad=0", 
                              ec=bg_color, fc=bg_color, alpha=.9),
                    **kwargs)
            ax_z.text(0, 0, 'z=%i' % z,
                    transform=ax_z.transAxes,
                    horizontalalignment='left',
                    verticalalignment='bottom',
                    size=size,
                    bbox=dict(boxstyle="square,pad=0", 
                              ec=bg_color, fc=bg_color, alpha=.9),
                    **kwargs)


    def title(self, text, x=0.01, y=0.99, size=15, color=None, 
                bgcolor=None, alpha=.9, **kwargs):
        """ Write a title to the view.

            Parameters
            ----------
            text: string
                The text of the title
            x: float, optional
                The horizontal position of the title on the frame in 
                fraction of the frame width.
            y: float, optional
                The vertical position of the title on the frame in 
                fraction of the frame height.
            size: integer, optional
                The size of the title text.
            color: matplotlib color specifier, optional
                The color of the font of the title.
            bgcolor: matplotlib color specifier, optional
                The color of the background of the title.
            alpha: float, optional
                The alpha value for the background.
            kwargs:
                Extra keyword arguments are passed to matplotlib's text
                function.
        """
        if color is None:
            color = 'k' if self._black_bg else 'w'
        if bgcolor is None:
            bgcolor = 'w' if self._black_bg else 'k'
        self.frame_axes.text(x, y, text, 
                    transform=self.frame_axes.transAxes,
                    horizontalalignment='left',
                    verticalalignment='top',
                    size=size, color=color,
                    bbox=dict(boxstyle="square,pad=.3", 
                              ec=bgcolor, fc=bgcolor, alpha=alpha),
                    **kwargs)


    def plot_map(self, map, affine, threshold=None, **kwargs):
        """ Plot a 3D map in all the views.

            Parameters
            -----------
            map: 3D ndarray
                The 3D map to be plotted. If it is a masked array, only
                the non-masked part will be plotted.
            affine: 4x4 ndarray
                The affine matrix giving the transformation from voxel
                indices to world space.
            threshold : a number, None, or 'auto'
                If None is given, the maps are not thresholded.
                If a number is given, it is used to threshold the maps:
                values below the threshold are plotted as transparent.
            kwargs:
                Extra keyword arguments are passed to imshow.
        """
        if threshold is not None:
            if threshold == 0:
                map = np.ma.masked_equal(map, 0, copy=False)
            else:
                map = np.ma.masked_inside(map, -threshold, threshold, 
                                          copy=False)

        self._map_show(map, affine, type='imshow', **kwargs)


    def contour_map(self, map, affine, **kwargs):
        """ Contour a 3D map in all the views.

            Parameters
            -----------
            map: 3D ndarray
                The 3D map to be plotted. If it is a masked array, only
                the non-masked part will be plotted.
            affine: 4x4 ndarray
                The affine matrix giving the transformation from voxel
                indices to world space.
            kwargs:
                Extra keyword arguments are passed to contour.
        """
        self._map_show(map, affine, type='contour', **kwargs)


    def _map_show(self, map, affine, type='imshow', **kwargs):
        map, affine = _xyz_order(map, affine)
        # Force the origin
        kwargs['origin'] = 'upper'
        if mp.__version__ < '0.99.1':
            cmap = kwargs.get('cmap', 
                        pl.cm.cmap_d[pl.rcParams['image.cmap']])
            kwargs['cmap'] = CMapProxy(cmap)
        x, y, z = self._cut_coords
        x_map, y_map, z_map = [int(round(c)) for c in 
                               coord_transform(x, y, z, np.linalg.inv(affine))]
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = get_bounds(map.shape, affine)

        xmin_, xmax_, ymin_, ymax_, zmin_, zmax_ = \
                                        xmin, xmax, ymin, ymax, zmin, zmax
        if hasattr(map, 'mask'):
            not_mask = np.logical_not(map.mask)
            xmin_, xmax_, ymin_, ymax_, zmin_, zmax_ = \
                            get_mask_bounds(not_mask, affine)
            if kwargs.get('vmin') is None and kwargs.get('vmax') is None:
                # Avoid dealing with masked arrays: they are slow
                masked_map = np.asarray(map)[not_mask]
                if kwargs.get('vmin') is None:
                    kwargs['vmin'] = masked_map.min()
                if kwargs.get('max') is None:
                    kwargs['vmax'] = masked_map.max()
        else:
            if not 'vmin' in kwargs:
                kwargs['vmin'] = map.min()
            if not 'vmax' in kwargs:
                kwargs['vmax'] = map.max()

        ax = self.axes['x']
        getattr(ax, type)(np.rot90(map[:, y_map, :]),
                  extent=(xmin, xmax, zmin, zmax),
                  **kwargs)
        self._object_bounds[ax].append((xmin_, xmax_, zmin_, zmax_))
        ax.axis(self._get_object_bounds(ax))

        ax = self.axes['y']
        getattr(ax, type)(np.rot90(map[x_map, :, :]),
                  extent=(ymin, ymax, zmin, zmax),
                  **kwargs)
        self._object_bounds[ax].append((ymin_, ymax_, zmin_, zmax_))
        ax.axis(self._get_object_bounds(ax))

        ax = self.axes['z']
        getattr(ax, type)(np.rot90(map[:, :, z_map]),
                  extent=(xmin, xmax, ymin, ymax),
                  **kwargs)
        self._object_bounds[ax].append((xmin_, xmax_, ymin_, ymax_))
        ax.axis(self._get_object_bounds(ax))


    def edge_map(self, map, affine, color='r'):
        """ Plot the edges of a 3D map in all the views.

            Parameters
            -----------
            map: 3D ndarray
                The 3D map to be plotted. If it is a masked array, only
                the non-masked part will be plotted.
            affine: 4x4 ndarray
                The affine matrix giving the transformation from voxel
                indices to world space.
            color: matplotlib color: string or (r, g, b) value
                The color used to display the edge map
        """
        map, affine = _xyz_order(map, affine)
        # Force the origin
        kwargs = dict(cmap=cm.alpha_cmap(color=color))
        kwargs['origin'] = 'upper'
        if mp.__version__ < '0.99.1':
            cmap = kwargs.get('cmap', 
                        pl.cm.cmap_d[pl.rcParams['image.cmap']])
            kwargs['cmap'] = CMapProxy(cmap)
        x, y, z = self._cut_coords
        x_map, y_map, z_map = [int(round(c)) for c in 
                               coord_transform(x, y, z, np.linalg.inv(affine))]
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = get_bounds(map.shape, affine)

        if y_map >= 0 and y_map < map.shape[1]:
            edge_mask = _edge_map(np.rot90(map[:, y_map, :]))
            getattr(self.axes['x'], 'imshow')(edge_mask,
                                        extent=(xmin, xmax, zmin, zmax), 
                                        vmin=0, **kwargs)

        if x_map >= 0 and x_map < map.shape[0]:
            edge_mask = _edge_map(np.rot90(map[x_map, :, :]))
            getattr(self.axes['y'], 'imshow')(edge_mask,
                                        extent=(ymin, ymax, zmin, zmax), 
                                        vmin=0, **kwargs)

        if z_map >= 0 and z_map < map.shape[-1]:
            edge_mask = _edge_map(np.rot90(map[:, :, z_map]))
            getattr(self.axes['z'], 'imshow')(edge_mask, 
                                        extent=(xmin, xmax, ymin, ymax), 
                                        vmin=0, **kwargs)


def demo_ortho_slicer():
    """ A small demo of the OrthoSlicer functionality.
    """
    pl.clf()
    oslicer = OrthoSlicer(cut_coords=(0, 0, 0))
    from anat_cache import _AnatCache
    map, affine, _ = _AnatCache.get_anat()
    oslicer.plot_map(map, affine, cmap=pl.cm.gray)
    return oslicer


