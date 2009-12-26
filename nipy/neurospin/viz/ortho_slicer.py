import numpy as np

import pylab as pl
from matplotlib.transforms import Bbox

# Local imports
from coord_tools import coord_transform, get_bounds, get_mask_bounds

################################################################################
# class OrthoSlicer
################################################################################

class OrthoSlicer(object):

    def __init__(self, axes=None):
        if axes is None:
            axes = pl.axes((0., 0., 1., 1.))
            axes.axis('off')
        self.ax = axes
        bb = axes.get_position()
        self.rect = (bb.x0, bb.y0, bb.x1, bb.y1)
        self._object_bounds = dict()

        # Create our axes:
        self.axes = dict()
        for index, name in enumerate(('x', 'y', 'z')):
            ax = pl.axes([0.3*index, 0, .3, 1])
            ax.axis('off')
            self.axes[name] = ax
            ax.set_axes_locator(self._locator)
            self._object_bounds[ax] = list()


    def get_object_bounds(self, ax):
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
        """
        width_dict = dict()
        ax_dict = self.axes
        x_ax = ax_dict['x']
        y_ax = ax_dict['y']
        z_ax = ax_dict['z']
        for ax in ax_dict.itervalues():
            xmin, xmax, ymin, ymax = self.get_object_bounds(ax)
            width_dict[ax] = xmax - xmin
        total_width = float(sum(width_dict.values()))
        for ax, width in width_dict.iteritems():
            width_dict[ax] = width/total_width
        left_dict = dict()
        left_dict[x_ax] = 0
        left_dict[y_ax] = width_dict[x_ax]
        left_dict[z_ax] = width_dict[x_ax] + width_dict[y_ax]
        return Bbox([[left_dict[axes], 0], 
                     [left_dict[axes] + width_dict[axes], 1]])


    def plot_map(self, map, affine, cut_coords, **kwargs):
        """ Plot a 3D map in all the views.

            Parameters
            -----------
            map: 3D ndarray
                The 3D map to be plotted. If it is a masked array, only
                the non-masked part will be plotted.
            affine: 4x4 ndarray
                The affine matrix giving the transformation from voxel
                indices to world space.
            cut_coords: 3 tuple of ints
                The cut positions in world space.
        """
        y, x, z = cut_coords
        x_map, y_map, z_map = [int(round(c)) for c in 
                               coord_transform(x, y, z, np.linalg.inv(affine))]
        xmin, xmax, ymin, ymax, zmin, zmax = get_bounds(map.shape, affine)

        xmin_, xmax_, ymin_, ymax_, zmin_, zmax_ = \
                            xmin, xmax, ymin, ymax, zmin, zmax
        if hasattr(map, 'mask'):
            xmin_, xmax_, ymin_, ymax_, zmin_, zmax_ = \
                        get_mask_bounds(np.logical_not(map.mask), affine)

        ax = self.axes['x']
        ax.imshow(np.rot90(map[:, y_map, :]),
                  extent=(xmin, xmax, zmin, zmax),
                  **kwargs)
        self._object_bounds[ax].append((xmin_, xmax_, zmin_, zmax_))
        this_xmin, this_xmax, this_ymin, this_ymax = \
                                        self.get_object_bounds(ax)
        ax.set_xlim(this_xmin, this_xmax)
        ax.set_ylim(this_ymin, this_ymax)

        ax = self.axes['y']
        ax.imshow(np.rot90(map[x_map, :, :]),
                  extent=(ymin, ymax, zmin, zmax),
                  **kwargs)
        self._object_bounds[ax].append((ymin_, ymax_, zmin_, zmax_))
        this_xmin, this_xmax, this_ymin, this_ymax = \
                                        self.get_object_bounds(ax)
        ax.set_xlim(this_xmin, this_xmax)
        ax.set_ylim(this_ymin, this_ymax)

        ax = self.axes['z']
        ax.imshow(np.rot90(map[:, :, z_map]),
                  extent=(xmin, xmax, ymin, ymax),
                  **kwargs)
        self._object_bounds[ax].append((xmin_, xmax_, ymin_, ymax_))
        this_xmin, this_xmax, this_ymin, this_ymax = \
                                        self.get_object_bounds(ax)
        ax.set_xlim(this_xmin, this_xmax)
        ax.set_ylim(this_ymin, this_ymax)


if __name__ == '__main__':
    pl.clf()
    oslicer = OrthoSlicer()
    from anat_cache import _AnatCache
    map, affine, _ = _AnatCache.get_anat()
    oslicer.plot_map(map, affine, (0, 0, 0), cmap=pl.cm.gray)
    pl.show()
    pl.draw()


