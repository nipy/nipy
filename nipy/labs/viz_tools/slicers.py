# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
The Slicer classes.

The main purpose of these classes is to have auto adjust of axes size to
the data with different layout of cuts.
"""

import operator

import numpy as np
from nipy.utils.skip_test import skip_if_running_nose

try:
    import matplotlib as mpl
    import pylab as pl
    from matplotlib import transforms
except ImportError:
    skip_if_running_nose('Could not import matplotlib')


# Local imports
from .coord_tools import coord_transform, get_bounds, get_mask_bounds, \
        find_cut_coords
from .edge_detect import _edge_map
from . import cm
from ..datasets import VolumeImg

################################################################################
# Bugware to have transparency work OK with MPL < .99.1
if mpl.__version__ < '0.99.1':
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
# class CutAxes
################################################################################

class CutAxes(object):
    """ An MPL axis-like object that displays a cut of 3D volumes
    """

    def __init__(self, ax, direction, coord):
        """ An MPL axis-like object that displays a cut of 3D volumes

            Parameters
            ==========
            ax: a MPL axes instance
                The axes in which the plots will be drawn
            direction: {'x', 'y', 'z'}
                The directions of the cut
            coord: float
                The coordinnate along the direction of the cut
        """
        self.ax = ax
        self.direction = direction
        self.coord = coord
        self._object_bounds = list()


    def do_cut(self, map, affine):
        """ Cut the 3D volume into a 2D slice

            Parameters
            ==========
            map: 3D ndarray
                The 3D volume to cut
            affine: 4x4 ndarray
                The affine of the volume
        """
        coords = [0, 0, 0]
        coords['xyz'.index(self.direction)] = self.coord
        x_map, y_map, z_map = [int(round(c)) for c in
                               coord_transform(coords[0],
                                               coords[1],
                                               coords[2],
                                               np.linalg.inv(affine))]
        if self.direction == 'y':
            cut = np.rot90(map[:, y_map, :])
        elif self.direction == 'x':
            cut = np.rot90(map[x_map, :, :])
        elif self.direction == 'z':
            cut = np.rot90(map[:, :, z_map])
        else:
            raise ValueError('Invalid value for direction %s' %
                             self.direction)
        return cut


    def draw_cut(self, cut, data_bounds, bounding_box,
                  type='imshow', **kwargs):
        # kwargs massaging
        kwargs['origin'] = 'upper'
        if mpl.__version__ < '0.99.1':
            cmap = kwargs.get('cmap',
                        pl.cm.cmap_d[pl.rcParams['image.cmap']])
            kwargs['cmap'] = CMapProxy(cmap)

        if self.direction == 'y':
            (xmin, xmax), (_, _), (zmin, zmax) = data_bounds
            (xmin_, xmax_), (_, _), (zmin_, zmax_) = bounding_box
        elif self.direction == 'x':
            (_, _), (xmin, xmax), (zmin, zmax) = data_bounds
            (_, _), (xmin_, xmax_), (zmin_, zmax_) = bounding_box
        elif self.direction == 'z':
            (xmin, xmax), (zmin, zmax), (_, _) = data_bounds
            (xmin_, xmax_), (zmin_, zmax_), (_, _) = bounding_box
        else:
            raise ValueError('Invalid value for direction %s' %
                             self.direction)
        ax = self.ax
        getattr(ax, type)(cut, extent=(xmin, xmax, zmin, zmax), **kwargs)

        self._object_bounds.append((xmin_, xmax_, zmin_, zmax_))
        ax.axis(self.get_object_bounds())


    def get_object_bounds(self):
        """ Return the bounds of the objects on this axes.
        """
        if len(self._object_bounds) == 0:
            # Nothing plotted yet
            return -.01, .01, -.01, .01
        xmins, xmaxs, ymins, ymaxs = np.array(self._object_bounds).T
        xmax = max(xmaxs.max(), xmins.max())
        xmin = min(xmins.min(), xmaxs.min())
        ymax = max(ymaxs.max(), ymins.max())
        ymin = min(ymins.min(), ymaxs.min())
        return xmin, xmax, ymin, ymax


    def draw_left_right(self, size, bg_color, **kwargs):
        if self.direction == 'x':
            return
        ax = self.ax
        ax.text(.1, .95, 'L',
                transform=ax.transAxes,
                horizontalalignment='left',
                verticalalignment='top',
                size=size,
                bbox=dict(boxstyle="square,pad=0",
                            ec=bg_color, fc=bg_color, alpha=.8),
                **kwargs)

        ax.text(.9, .95, 'R',
                transform=ax.transAxes,
                horizontalalignment='right',
                verticalalignment='top',
                size=size,
                bbox=dict(boxstyle="square,pad=0",
                            ec=bg_color, fc=bg_color, alpha=.8),
                **kwargs)


    def draw_position(self, size, bg_color, **kwargs):
        ax = self.ax
        ax.text(0, 0, '%s=%i' % (self.direction, self.coord),
                transform=ax.transAxes,
                horizontalalignment='left',
                verticalalignment='bottom',
                size=size,
                bbox=dict(boxstyle="square,pad=0",
                            ec=bg_color, fc=bg_color, alpha=.9),
                **kwargs)


################################################################################
# class BaseSlicer
################################################################################

class BaseSlicer(object):
    """ The main purpose of these class is to have auto adjust of axes size
        to the data with different layout of cuts.
    """
    # This actually encodes the figsize for only one axe
    _default_figsize = [2.2, 2.6]

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
        self._black_bg = black_bg
        self._init_axes()


    @staticmethod
    def find_cut_coords(data=None, affine=None, threshold=None,
                        cut_coords=None):
        # Implement this as a staticmethod or a classmethod when
        # subclassing
        raise NotImplementedError

    @classmethod
    def init_with_figure(cls, data=None, affine=None, threshold=None,
                         cut_coords=None, figure=None, axes=None,
                         black_bg=False, leave_space=False):
        cut_coords = cls.find_cut_coords(data, affine, threshold,
                                         cut_coords)

        if not isinstance(figure, pl.Figure):
            # Make sure that we have a figure
            figsize = cls._default_figsize[:]
            # Adjust for the number of axes
            figsize[0] *= len(cut_coords)
            facecolor = 'k' if black_bg else 'w'

            if leave_space:
                figsize[0] += 3.4
            figure = pl.figure(figure, figsize=figsize,
                            facecolor=facecolor)
        else:
            if isinstance(axes, pl.Axes):
                assert axes.figure is figure, ("The axes passed are not "
                "in the figure")

        if axes is None:
            axes = [0., 0., 1., 1.]
            if leave_space:
                axes = [0.3, 0, .7, 1.]
        if operator.isSequenceType(axes):
            axes = figure.add_axes(axes)
            axes.axis('off')
        return cls(cut_coords, axes, black_bg)


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

        data_bounds = get_bounds(map.shape, affine)
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = data_bounds

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

        bounding_box = (xmin_, xmax_), (ymin_, ymax_), (zmin_, zmax_)

        # For each ax, cut the data and plot it
        for cut_ax in self.axes.itervalues():
            try:
                cut = cut_ax.do_cut(map, affine)
            except IndexError:
                # We are cutting outside the indices of the data
                continue
            cut_ax.draw_cut(cut, data_bounds, bounding_box,
                            type=type, **kwargs)


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
        kwargs = dict(cmap=cm.alpha_cmap(color=color))
        data_bounds = get_bounds(map.shape, affine)

        # For each ax, cut the data and plot it
        for cut_ax in self.axes.itervalues():
            try:
                cut = cut_ax.do_cut(map, affine)
                edge_mask = _edge_map(cut)
            except IndexError:
                # We are cutting outside the indices of the data
                continue
            cut_ax.draw_cut(edge_mask, data_bounds, data_bounds,
                            type='imshow', **kwargs)

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
            for cut_ax in self.axes.values():
                cut_ax.draw_left_right(size=size, bg_color=bg_color,
                                       **kwargs)

        if positions:
            for cut_ax in self.axes.values():
                cut_ax.draw_position(size=size, bg_color=bg_color,
                                       **kwargs)


################################################################################
# class OrthoSlicer
################################################################################

class OrthoSlicer(BaseSlicer):
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
    @staticmethod
    def find_cut_coords(data=None, affine=None, threshold=None,
                        cut_coords=None):
        if cut_coords is None:
            if data is None or data is False:
                cut_coords = (0, 0, 0)
            else:
                x_map, y_map, z_map = find_cut_coords(data,
                                        activation_threshold=threshold)
                cut_coords = coord_transform(x_map, y_map, z_map, affine)
        return cut_coords


    def _init_axes(self):
        x0, y0, x1, y1 = self.rect
        # Create our axes:
        self.axes = dict()
        for index, direction in enumerate(('y', 'x', 'z')):
            ax = pl.axes([0.3*index*(x1-x0) + x0, y0, .3*(x1-x0), y1-y0])
            ax.axis('off')
            coord = self._cut_coords['xyz'.index(direction)]
            cut_ax = CutAxes(ax, direction, coord)
            self.axes[direction] = cut_ax
            ax.set_axes_locator(self._locator)


    def _locator(self, axes, renderer):
        """ The locator function used by matplotlib to position axes.
            Here we put the logic used to adjust the size of the axes.
        """
        x0, y0, x1, y1 = self.rect
        width_dict = dict()
        cut_ax_dict = self.axes
        x_ax = cut_ax_dict['x']
        y_ax = cut_ax_dict['y']
        z_ax = cut_ax_dict['z']
        for cut_ax in cut_ax_dict.itervalues():
            bounds = cut_ax.get_object_bounds()
            if not bounds:
                # This happens if the call to _map_show was not
                # succesful. As it happens asyncroniously (during a
                # refresh of the figure) we capture the problem and
                # ignore it: it only adds a non informative traceback
                bounds = [0, 1, 0, 1]
            xmin, xmax, ymin, ymax = bounds
            width_dict[cut_ax.ax] = (xmax - xmin)
        total_width = float(sum(width_dict.values()))
        for ax, width in width_dict.iteritems():
            width_dict[ax] = width/total_width*(x1 -x0)
        left_dict = dict()
        left_dict[y_ax.ax] = x0
        left_dict[x_ax.ax] = x0 + width_dict[y_ax.ax]
        left_dict[z_ax.ax] = x0 + width_dict[x_ax.ax] + width_dict[y_ax.ax]
        return transforms.Bbox([[left_dict[axes], y0],
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
        ax = self.axes['y'].ax
        ax.axvline(x, ymin=.05, ymax=.95, **kwargs)
        ax.axhline(z, **kwargs)

        ax = self.axes['x'].ax
        ax.axvline(y, ymin=.05, ymax=.95, **kwargs)
        ax.axhline(z, xmax=.95, **kwargs)

        ax = self.axes['z'].ax
        ax.axvline(x, ymin=.05, ymax=.95, **kwargs)
        ax.axhline(y, **kwargs)



def demo_ortho_slicer():
    """ A small demo of the OrthoSlicer functionality.
    """
    pl.clf()
    oslicer = OrthoSlicer(cut_coords=(0, 0, 0))
    from anat_cache import _AnatCache
    map, affine, _ = _AnatCache.get_anat()
    oslicer.plot_map(map, affine, cmap=pl.cm.gray)
    return oslicer


################################################################################
# class BaseStackedSlicer
################################################################################

class BaseStackedSlicer(BaseSlicer):
    """ A class to create linked axes for plotting stacked
        cuts of 3D maps.

        Attributes
        ----------

        axes: dictionnary of axes
            The axes used to plot each view.
        frame_axes: axes
            The axes framing the whole set of views.

        Notes
        -----

        The extent of the different axes are adjusted to fit the data
        best in the viewing area.
    """
    @classmethod
    def find_cut_coords(cls, data=None, affine=None, threshold=None,
                        cut_coords=None):
        if cut_coords is None:
            if data is None or data is False:
                bounds = ((-40, 40), (-30, 30), (-30, 75))
            else:
                if hasattr(data, 'mask'):
                    mask = np.logical_not(data.mask)
                else:
                    # The mask will be anything that is fairly different
                    # from the values in the corners
                    edge_value = float(data[0, 0, 0] + data[0, -1, 0]
                                     + data[-1, 0, 0] + data[0, 0, -1]
                                     + data[-1, -1, 0] + data[-1, 0, -1]
                                     + data[0, -1, -1] + data[-1, -1, -1]
                                    )
                    edge_value /= 6
                    mask = np.abs(data - edge_value) > .005*data.ptp()
                xmin, xmax, ymin, ymax, zmin, zmax = \
                                get_mask_bounds(mask, affine)
                bounds = (xmin, xmax), (ymin, ymax), (zmin, zmax)
            lower, upper = bounds['xyz'.index(cls._direction)]
            cut_coords = np.linspace(lower, upper, 10).tolist()
        return cut_coords


    def _init_axes(self):
        x0, y0, x1, y1 = self.rect
        # Create our axes:
        self.axes = dict()
        fraction = 1./len(self._cut_coords)
        for index, coord in enumerate(self._cut_coords):
            coord = float(coord)
            ax = pl.axes([fraction*index*(x1-x0) + x0, y0,
                          fraction*(x1-x0), y1-y0])
            ax.axis('off')
            cut_ax = CutAxes(ax, self._direction, coord)
            self.axes[coord] = cut_ax
            ax.set_axes_locator(self._locator)


    def _locator(self, axes, renderer):
        """ The locator function used by matplotlib to position axes.
            Here we put the logic used to adjust the size of the axes.
        """
        x0, y0, x1, y1 = self.rect
        width_dict = dict()
        cut_ax_dict = self.axes
        for cut_ax in cut_ax_dict.itervalues():
            bounds = cut_ax.get_object_bounds()
            if not bounds:
                # This happens if the call to _map_show was not
                # succesful. As it happens asyncroniously (during a
                # refresh of the figure) we capture the problem and
                # ignore it: it only adds a non informative traceback
                bounds = [0, 1, 0, 1]
            xmin, xmax, ymin, ymax = bounds
            width_dict[cut_ax.ax] = (xmax - xmin)
        total_width = float(sum(width_dict.values()))
        for ax, width in width_dict.iteritems():
            width_dict[ax] = width/total_width*(x1 -x0)
        left_dict = dict()
        left = float(x0)
        for coord, cut_ax in sorted(cut_ax_dict.items()):
            left_dict[cut_ax.ax] = left
            this_width = width_dict[cut_ax.ax]
            left += this_width
        print len(width_dict), left_dict[axes] + width_dict[axes]
        return transforms.Bbox([[left_dict[axes], y0],
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
        return


class XSlicer(BaseStackedSlicer):
    _direction = 'x'
    _default_figsize = [2.2, 2.3]


class YSlicer(BaseStackedSlicer):
    _direction = 'y'
    _default_figsize = [2.6, 2.3]


class ZSlicer(BaseStackedSlicer):
    _direction = 'z'


SLICERS = dict(ortho=OrthoSlicer,
               x=XSlicer,
               y=YSlicer,
               z=ZSlicer)
