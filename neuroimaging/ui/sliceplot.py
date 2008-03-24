"""Classes for plotting image slices using Matplotlib.

SlicePlot creates a single plot of a slice.
SliceViewer will create a multi-planar view containing the axial, coronal,
and sagittal slices.

"""

from matplotlib import cm
from matplotlib.axes import Subplot
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from numpy import ndarray, zeros
from pylab import figure

from neuroimaging.core.image import image

_default_alpha = 0.5
_bgcolor = (0.2, 0.2, 0.2)

class SlicePlot(object):
    """Class for plotting image slices.

    SlicePlot uses Matplotlib's imshow to plot image slices.

    Examples
    --------
    >>> from neuroimaging.ui import sliceplot
    >>> from neuroimaging.testing import anatfile
    >>> from neuroimaging.core.image import image
    >>> img = image.load(anatfile)
    >>> zdim, ydim, xdim = img.shape
    >>> sag_slice = img[:, :, xdim/2]
    >>> import pylab
    >>> fig = pylab.figure()
    >>> plt = fig.add_subplot(111)
    >>> sag_splot = sliceplot.SlicePlot(plt, sag_slice)
    >>> sag_splot.interpolation = 'bilinear'

    """

    def __init__(self, plt=None, data=None, parent=None, cmap=cm.gray, 
                 origin='lower', interpolation='nearest'):
        """Create a slice plot using matplotlib.
        
        Parameters
        ----------
        plt : matplotlib.axes.Axes
            created via pylab.axes, fig.add_subplot
        data : ndarray
        parent : object that owns this plot to receive event messages
        """

        if plt is None:
            fig = figure()
            plt = fig.add_subplot(1, 1, 1, axisbg=_bgcolor)
        self.plt = plt
        # plt is a matplotlib.axes.Subplot object
        # We could code it to accept a matplotlib.image.AxesImage also.

        self.parent = parent

        if data is None:
            # fill in data with small array to create plots
            data = zeros((100,100), dtype='int16')
        self.data = data
        self.imgaxes = self.plt.imshow(self.data, origin=origin, cmap=cmap,
                                       interpolation=interpolation)
        self.olayaxes = None
        self.crosshair = Crosshairs(self.plt)
        self._init_axes()
        self._init_event_handlers()
        self.draw()

    def _init_axes(self):
        """Initialize grid and labels."""
        self.set_grid(True, color='y')
        self.set_xlabel()
        self.set_ylabel()

    def _init_event_handlers(self):
        """Initialize event handlers, like mouse picking."""
        self.imgaxes.set_picker(1)
        self.plt.figure.canvas.mpl_connect('button_press_event', self._on_pick)
        self.plt.figure.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.plt.figure.canvas.mpl_connect('key_press_event', self._on_key)

    def draw(self):
        """Draw the plot."""
        # TODO: Should have an 'interactive' flag to enable/disable
        # all interactive updates?
        self.plt.figure.canvas.draw_idle()

    def _get_alpha(self):
        if self.olayaxes:
            return self.olayaxes.get_alpha()
        else:
            return None
    def _set_alpha(self, alpha):
        if self.olayaxes:
            self.olayaxes.set_alpha(alpha)
            self.draw()
        else:
            raise AttributeError, 'plot has now overlay image.'
    alpha = property(_get_alpha, _set_alpha,
                     doc="Matplotlib alpha value used for overlay images")

    def _get_olaycmap(self):
        return getattr(self.olayaxes, 'cmap', None)
    def _set_olaycmap(self, cmap):
        self.olayaxes.set_cmap(cmap)
        self.draw()
    olaycmap = property(_get_olaycmap, _set_olaycmap,
                        doc="Matplotlib colormap object used for overlay")

    def _get_cmap(self):
        return self.imgaxes.cmap
    def _set_cmap(self, cmap):
        self.imgaxes.set_cmap(cmap)
        self.draw()
    cmap = property(_get_cmap, _set_cmap,
                    doc="Matplotlib colormap object used to draw plot")

    def _get_interp(self):
        return self.imgaxes.get_interpolation()
    def _set_interp(self, interp):
        # Get valid interpolation keys from Matplotlib
        interp_methods = self.imgaxes._interpd.keys()
        if interp in interp_methods:
            self.imgaxes.set_interpolation(interp)
            self.draw()
        else:
            raise AttributeError, \
                "Invalid interpolation method, select from: %s"% interp_methods
    interpolation = property(_get_interp, _set_interp,
                             doc="Interpolation method for drawing plot")

    def _get_origin(self):
        # getattr(self.imgaxes, 'origin')
        return self.imgaxes.origin
    def _set_origin(self, origin):
        if origin in ('lower', 'upper'):
            self.imgaxes.origin = origin
            self.draw()
        else:
            raise AttributeError, "Valid origins: 'lower', 'upper'"
    origin = property(_get_origin, _set_origin, 
                      doc="plot origin {'lower', 'upper'}")

    def _on_pick(self, event):
        """Handle pick events in the plot.  Use Matplotlib picker."""
        if not event.inaxes:
            return
        if event.inaxes is self.plt:
            x = event.xdata
            y = event.ydata
            xi = int(x)
            yi = int(y)
            val = self.data[yi, xi]
            msg = 'Index(%.2f, %.2f) Intensity(%.2f)' % (x, y, val)
            self._set_message(msg)
            self.set_crosshair(x, y)
            #self.set_data(self.data)
            if self.parent and hasattr(self.parent, '_on_pick'):
                self.parent._on_pick(self, x, y)

    def _on_scroll(self, event):
        """Handle mouse wheel events in plot.
        
        Notes
        -----
        Mouse wheel is only supported in the SliceViewer.
        
        """

        if event.inaxes and event.inaxes is self.plt:
            if self.parent and hasattr(self.parent, '_on_scroll'):
                self.parent._on_scroll(self, event)

    def _on_key(self, event):
        """Handle key events in plot.

        Notes
        -----
        Arrow keys are supported in the SliceViewer.

        """
        if event.inaxes and event.inaxes is self.plt:
            if self.parent and hasattr(self.parent, '_on_key'):
                self.parent._on_key(self, event)

    def set_clim(self, vmin, vmax):
        self.imgaxes.set_clim(vmin, vmax)
    set_clim.__doc__ = AxesImage.set_clim.__doc__

    def set_crosshair(self, x, y):
        """Set location of the crosshair.

        Parameters
        ----------
        x : horizontal coordinate
        y : vertical coordinate

        """
        
        self.crosshair.set_data(x, y)
        self.draw()

    def set_data(self, data):
        """Update the data in this plot.
        
        Parameters
        ----------
        data : a 2D array

        """

        if data.ndim is not 2:
            raise ValueError, 'Data array must be a 2D array.'

        # Should something like this work via a property?
        #     sag_plot.data = mni_vol[:, :, 10]
        self.data = data
        self.imgaxes.set_data(self.data)
        vmin = self.data.min()
        vmax = self.data.max()
        self.set_clim(vmin, vmax)
        ydim, xdim = self.data.shape
        self.set_xlim((0, xdim))
        self.set_ylim((0, ydim))
        self.draw()

    def set_grid(self, b=None, **kwargs):
        """Set attributes of the grid for this SlicePlot.

        This is a simple wrapper around Matplotlib Subplot.grid.

        """

        self.plt.grid(b, **kwargs)
    # Append docstring from Subplot.grid
    set_grid.__doc__ = ''.join((set_grid.__doc__, Subplot.grid.__doc__))

    def set_overlay(self, data):
        """Set overlay data for this plot."""
        if self.olayaxes is None:
            alpha = _default_alpha
        else:
            alpha = self.alpha
        self.overlay = data
        self.plt.hold(True)
        self.olayaxes = self.plt.imshow(data, cmap=self.olaycmap, 
                                        alpha=alpha,
                                        interpolation=self.interpolation, 
                                        origin=self.origin)
        self.draw()

    def _set_message(self, msg):
        """Set message in the toolbar."""
        self.plt.figure.canvas.toolbar.set_message(msg)

    def set_xlabel(self, label='x', **kwargs):
        """Set attributes of the xlabel for this SlicePlot.
        
        This is a simple wrapper around Matplotlib Subplot.set_xlabel.

        """
        
        self.plt.set_xlabel(label, **kwargs)
    # Append docstring from Subplot.set_xlabel
    set_xlabel.__doc__ = ''.join((set_xlabel.__doc__, 
                                  Subplot.set_xlabel.__doc__))

    def set_xlim(self, xlim):
        self.plt.set_xlim(xlim)
    set_xlim.__doc__ = Subplot.set_xlim.__doc__

    def set_ylabel(self, label='y', **kwargs):
        """Set attributes of the ylabel for this SlicePlot.

        This is a simple wrapper around Matplotlib Subplot.set_ylabel.

        """

        self.plt.set_ylabel(label, **kwargs)
    # Append docstring from Subplot.set_ylabel
    set_ylabel.__doc__ = ''.join((set_ylabel.__doc__,
                                  Subplot.set_ylabel.__doc__))

    def set_ylim(self, ylim):
        self.plt.set_ylim(ylim)
    set_ylim.__doc__ = Subplot.set_ylim.__doc__

class Crosshairs(object):
    """Crosshairs for the SlicePlots.
    
    Draws crosshairs to the full extent of the plot.

    """

    def __init__(self, plt):
        # determine axes limits, draw crosshair to extents
        self._plt = plt
        xlim = self._plt.get_xlim()
        xmid = xlim[1] / 2
        ylim = self._plt.get_ylim()
        ymid = ylim[1] / 2

        self.horiz = Line2D(xlim, [ymid, ymid], color='red')
        self._plt.add_artist(self.horiz)

        self.vert = Line2D([xmid, xmid], ylim, color='red')
        self._plt.add_artist(self.vert)

    def set_data(self, x, y):
        """Set the location of the crosshair.
        
        Parameters
        ----------
        x : horizontal location
        y : vertical location

        """
        # Should the crosshair know about the image?
        ext = self._plt.images[0].get_extent()
        xlim = (ext[0], ext[1])
        ylim = (ext[2], ext[3])
        self.horiz.set_data(xlim, [y, y])
        self.vert.set_data([x, x], ylim)
        #self.horiz.set_data(self._plt.get_xlim(), [y, y])
        #self.vert.set_data([x, x], self._plt.get_ylim())

class SliceViewer(object):
    """View brain image slices in 3 orthogonal slice plots.

    Mouse click in a plot to set the coordinates of the corresponding plots
    to the picked location.

    Use the mouse-wheel to scroll through image slices.

    Notes
    -----

    This class depends on a current version of Matplotlib, as least v0.91.
    This dependancy is due to the mouse-wheel support (scroll_event).
 
    Examples
    --------
    
    >>> import neuroimaging.ui.sliceplot as spt
    >>> from neuroimaging.testing import anatfile
    >>> # Initialize viewer with a data file
    >>> viewer = spt.SliceViewer(anatfile)

    >>> import neuroimaging.ui.sliceplot as spt
    >>> from neuroimaging.testing import anatfile
    >>> viewer = spt.SliceViewer()
    >>> # Set the data after viewer is created
    >>> viewer.set_data(anatfile)

    """

    # image attrs
    _x = 0
    _y = 0
    _z = 0
    img = None
    # overlay image attrs
    olayimg = None
    _xo = 0
    _yo = 0
    _zo = 0
    # misc attrs
    scroll_delta = 1

    def __init__(self, data=None, title='SliceViewer'):
        self.fig = figure()
        # put some space between plots so it's easier to see axis
        self.fig.subplots_adjust(hspace=0.3, wspace=0.3)
        # create 3 orthogonal view subplots
        self._cor_subplot = self.fig.add_subplot(2, 2, 1, axisbg=_bgcolor)
        self._cor_subplot.axis('equal')
        self._sag_subplot = self.fig.add_subplot(2, 2, 2, axisbg=_bgcolor)
        self._sag_subplot.axis('equal')
        self._axl_subplot = self.fig.add_subplot(2, 2, 3, axisbg=_bgcolor)
        self._axl_subplot.axis('equal')

        # create SlicePlots
        self.cor_splot = SlicePlot(self._cor_subplot, parent=self)
        self.sag_splot = SlicePlot(self._sag_subplot, parent=self)
        self.axl_splot = SlicePlot(self._axl_subplot, parent=self)

        self._init_axes()

        if data:
            self.set_data(data)
        else:
            self.draw()

    def _init_axes(self):
        self.axl_splot.set_xlabel('x', size='medium', color='b')
        self.axl_splot.set_ylabel('y', size='medium', color='b',
                                  rotation='horizontal')
        self.cor_splot.set_xlabel('x', size='medium', color='b')
        self.cor_splot.set_ylabel('z', size='medium', color='b',
                                  rotation='horizontal')
        self.sag_splot.set_xlabel('y', size='medium', color='b')
        self.sag_splot.set_ylabel('z', size='medium', color='b',
                                  rotation='horizontal')
                
    def set_data(self, data, **kwargs):
        """Set the data for the viewer.

        Notes
        -----
        The kwargs is just used for testing at this point. Using it to
        test switching io to pynifti.

        """
        tmpimg = _image_loader(data, **kwargs)
        if tmpimg:
            self.img = tmpimg
            self._z, self._y, self._x = map(lambda x: x/2, self.img.shape)
            zdim, ydim, xdim = self.img.shape
            
            # Find min and max data values for set_clim calls below
            vmin = self.img._data.min()
            vmax = self.img._data.max()
            # Update Axial Plot
            self._set_axial_data()
            self.axl_splot.set_xlim((0, xdim))
            self.axl_splot.set_ylim((0, ydim))
            self.axl_splot.set_clim(vmin, vmax)
            # Update Coronal Plot
            self._set_coronal_data()
            self.cor_splot.set_clim(vmin, vmax)
            self.cor_splot.set_xlim((0, xdim))
            self.cor_splot.set_ylim((0, zdim))
            # Update Sagittal Plot
            self._set_sagittal_data()
            self.sag_splot.set_clim(vmin, vmax)
            self.sag_splot.set_xlim((0, ydim))
            self.sag_splot.set_ylim((0, zdim))
            self.draw()

    def set_overlay(self, data):
        """Set an overlay in the viewer."""
        
        tmpimg = _image_loader(data)
        if tmpimg:
            self.olayimg = tmpimg
            self._zo, self._yo, self._xo = map(lambda x: x/2, 
                                               self.olayimg.shape)
            self._set_axial_olay()
            self._set_coronal_olay()
            self._set_sagittal_olay()
            self.draw()

    def _get_alpha(self):
        aaxl = self.axl_splot.alpha
        acor = self.cor_splot.alpha
        asag = self.sag_splot.alpha
        if aaxl == acor == asag:
            return aaxl
        else:
            return (aaxl, acor, asag)
    def _set_alpha(self, alpha):
        self.axl_splot.alpha = alpha
        self.cor_splot.alpha = alpha
        self.sag_splot.alpha = alpha
        self.draw()
    alpha = property(fget=_get_alpha, fset=_set_alpha,
                     doc="get/set alpha value for all three plots")

    def _get_cmap(self):
        pass # is a get needed?  here for docstring to work in property
    def _set_cmap(self, cmap):
        self.axl_splot.cmap = cmap
        self.cor_splot.cmap = cmap
        self.sag_splot.cmap = cmap
        self.draw()
    cmap = property(fget=_get_cmap, fset=_set_cmap,
                    doc="set cmap for all three plots")
        
    def _get_olaycmap(self):
        pass # is a get needed?  here for docstring to work in property
    def _set_olaycmap(self, cmap):
        self.axl_splot.olaycmap = cmap
        self.cor_splot.olaycmap = cmap
        self.sag_splot.olaycmap = cmap
        self.draw()
    olaycmap = property(fget=_get_olaycmap, fset=_set_olaycmap,
                        doc="set overlay cmap for all three plots")

    def draw(self):
        """Show the figure."""
        self.axl_splot.draw()
        self.cor_splot.draw()
        self.sag_splot.draw()
        self.fig.show()

    def _on_pick(self, plt, x, y):
        """Update plot crosshairs and data on mouse pick.
        
        Parameters
        ----------
        plt : SlicePlot mouse pick happened in.
        x : x coordinate of mouse pick
        y : y coordinate of mouse pick

        """

        if plt is self.axl_splot:
            # Axial pixel mapping: plot.xyi => data.xyz
            self._x, self._y = x, y
            self._set_coronal_data()
            self._set_sagittal_data()
            self._xo, self._yo = x, y
            self._set_coronal_olay()
            self._set_sagittal_olay()
        elif plt is self.cor_splot:
            # Coronal pixel mapping: plot.xyi => data.xzy
            self._x, self._z = x, y
            self._set_axial_data()
            self._set_sagittal_data()
            self._xo, self._zo = x, y
            self._set_axial_olay()
            self._set_sagittal_olay()
        elif plt is self.sag_splot:
            # Sagittal pixel mapping: plot.xyi => data.yzx
            self._y, self._z = x, y
            self._set_axial_data()
            self._set_coronal_data()
            self._yo, self._zo = x, y
            self._set_axial_olay()
            self._set_coronal_olay()

    def _on_scroll(self, plt, event):
        """Update slice index using the mouse wheel.

        Notes
        -----
        Scrolling is not implemented with overlays.  This is getting too hacked.
        Added overlay code is getting to messy.  Need to generalize the
        image handling to support multiple images better.  And use nipy's 
        reference code for image sampling.  Do this after Paris Sprint.
        -chris 2008-03-04

        """

        zdim, ydim, xdim = self.img.shape
        delta = 0
        if plt is self.axl_splot:
            if event.button == 'up':
                if self._z < zdim-self.scroll_delta:
                    delta = self.scroll_delta
            else:
                if self._z > self.scroll_delta:
                    delta = -self.scroll_delta
            if delta:
                # in case the set_data functions are expensive, only call
                # it when we updated the slice
                self._z += delta
                self._set_axial_data()
                self._set_coronal_cross()
                self._set_sagittal_cross()
        elif plt is self.cor_splot:
            if event.button == 'up':
                if self._y < ydim-self.scroll_delta:
                    delta = self.scroll_delta
            else:
                if self._y > self.scroll_delta:
                    delta = -self.scroll_delta
            if delta:
                self._y += delta
                self._set_coronal_data()
                self._set_axial_cross()
                self._set_sagittal_cross()
        elif plt is self.sag_splot:
            if event.button == 'up':
                if self._x < xdim-self.scroll_delta:
                    delta = self.scroll_delta
            else:
                if self._x > self.scroll_delta:
                    delta = -self.scroll_delta
            if delta:
                self._x += delta
                self._set_sagittal_data()
                self._set_axial_cross()
                self._set_coronal_cross()
        # Set msg in toolbar for debugging
        # They all share one toolbar so, just call for one plot
        msg = '(%.2f, %.2f, %.2f)' % (self._z, self._y, self._x)
        self.axl_splot._set_message(msg)

    def _on_key(self, plt, event):
        """Handle arrow key events.
        
        Notes
        -----
        Only support up and down arrow keys.

        """

        if event.key == 'up':
            event.button = 'up'
            self._on_scroll(plt, event)
        elif event.key == 'down':
            event.button = 'down'
            self._on_scroll(plt, event)

    def _set_axial_data(self):
        """Set axial slice and crosshair to voxel specified by x, y, z."""
        if self.img:
            data = _axial_slice(self.img, self._z)
            self.axl_splot.set_data(data)
        self._set_axial_cross()

    def _set_axial_cross(self):
        """Set the axial crosshair to the current x, y coords."""
        self.axl_splot.set_crosshair(self._x, self._y)

    def _set_axial_olay(self):
        """Set overlay image for the axial slice."""
        if self.olayimg:
            data = _axial_slice(self.olayimg, self._zo)
            self.axl_splot.set_overlay(data)

    def _set_coronal_data(self):
        """Set coronal slice and crosshair to voxel specified by x, y, z."""
        if self.img:
            data = _coronal_slice(self.img, self._y)
            self.cor_splot.set_data(data)
        self._set_coronal_cross()

    def _set_coronal_cross(self):
        """Set the coronal crosshair to the current x, z coords."""
        self.cor_splot.set_crosshair(self._x, self._z)

    def _set_coronal_olay(self):
        """Set overlay image for the coronal slice."""
        if self.olayimg:
            data = _coronal_slice(self.olayimg, self._y)
            self.cor_splot.set_overlay(data)
            
    def _set_sagittal_data(self):
        """Set sagittal slice and crosshair to voxel specified by x, y, z."""
        if self.img:
            data = _sagittal_slice(self.img, self._x)
            self.sag_splot.set_data(data)
        self._set_sagittal_cross()

    def _set_sagittal_cross(self):
        """Set the sagittal crosshair to the current y, z coords."""
        self.sag_splot.set_crosshair(self._y, self._z)
    
    def _set_sagittal_olay(self):
        """Set overlay image for the sagittal slice."""
        if self.olayimg:
            data = _sagittal_slice(self.olayimg, self._z)
            self.sag_splot.set_overlay(data)

class _SlicePlotHandler(object):
    """Class for handling events between SlicePlots.
    
    Handles interfacing with the Image class and SlicePlots.
    """

    _observers = list()

    def __init__(self):
        pass

    def attach(self, observer):
        if observer not in _observers:
            _observers.append(observer)

    def detach(self, observer):
        if observer in _observers:
            _observers.remove(observer)

    def pick_event(self, observer, x, y):
        pass


# NOTE:  Slice functions below are convenience functions for accessing slices.
# Need to look into the reference code and find the appropriate way to
# resample the data.

def _axial_slice(img, zindex, xlim=None, ylim=None, t=0):
    """Return axial slice of the image."""
    
    if img.ndim is 4:
        img = img[t, :, :, :]
    zdim, ydim, xdim = img.shape
    if not xlim:
        xlim = [0, xdim]
    if not ylim:
        ylim = [0, ydim]
    return img[zindex, ylim[0]:ylim[1], xlim[0]:xlim[1]]

def _coronal_slice(img, yindex, xlim=None, zlim=None, t=0):
    """Return coronal slice of the image."""

    if img.ndim is 4:
        img = img[t, :, :, :]
    zdim, ydim, xdim = img.shape
    if not xlim:
        xlim = [0, xdim]
    if not zlim:
        zlim = [0, zdim]
    return img[zlim[0]:zlim[1], yindex, xlim[0]:xlim[1]]

def _sagittal_slice(img, xindex, ylim=None, zlim=None, t=0):
    """Return sagittal slice of the image."""

    if img.ndim is 4:
        img = img[t, :, :, :]
    zdim, ydim, xdim = img.shape
    if not ylim:
        ylim = [0, ydim]
    if not zlim:
        zlim = [0, zdim]
    return img[zlim[0]:zlim[1], ylim[0]:ylim[1], xindex]


def _image_loader(data, **kwargs):
    """Utility function to load images.
    
    Parameters
    ----------
    data : {image-like, string-like, array-like}
        data to load into the image viewer
    
    """

    img = None
    if isinstance(data, image.Image):
        # is there a unique attr we can use to check for image-like
        # objects instead of type-checking?
        img = data
    elif hasattr(data, 'join'):
        # data is string-like, try and open file
        try:
            # Hack:  This logic should go into image module
            if kwargs.get('pynifti', None):
                img = image._open_pynifti(data)
            else:
                img = image.load(data)
        except:
            raise IOError, "Unable to load image %s" % data
    elif isinstance(data, ndarray):
        # check attr for array-like instead of type-checking!
        # this works with memmaps.
        img = image.fromarray(data)
    else:
        raise ValueError, 'Unable to load image %s' % data
# Add file-like object also
    return img
