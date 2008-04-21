"""
A Slice class for visualizing slices of images.
"""

__docformat__ = 'restructuredtext'

import numpy as N
from neuroimaging import traits
from neuroimaging.core.reference import slices
from neuroimaging.algorithms.interpolation import ImageInterpolator

class Slice(object):


    def __init__(self, interpolator, grid, transpose=False):
        self.interpolator = interpolator
        self.grid = grid
        self.drawer_args = ()
        self.drawer_keywords = {}
        self.transpose = transpose

    def draw(self, *args, **keywords):
        """
        Draw the resulting slice.
        """
        pass

def coronal(grid, y=0.,
            shape=None,
            xlim=None,
            zlim=None):
    """
    Coronal slice through a grid, with optional xlim and zlim. Otherwise,
    these are taken from a bounding for grid.

    Shape refers to size of array in sampling the region by an interpolator.
    TODO: make these slices 2dslices, rather than this hack +1.0e-06
    """
    shape = shape or grid.shape
    if xlim is None:
        xlim = slices.bounding_box(grid)[2]
    if zlim is None:
        zlim = slices.bounding_box(grid)[0]
    return slices.yslice(y, zlim, [y,y+1.0e-06], xlim, (shape[0],2,shape[2]))

def transversal(grid, z=0.,
                shape=None,
                xlim=None,
                ylim=None):
    """
    Transversal slice through a grid, with optional xlim and ylim. Otherwise,
    these are taken from a bounding for grid.
    TODO: make these slices 2dslices, rather than this hack +1.0e-06
    Shape refers to size of array in sampling the region by an interpolator.
    """
    shape = shape or grid.shape
    if xlim is None:
        xlim = slices.bounding_box(grid)[2]
    if ylim is None:
        ylim = slices.bounding_box(grid)[1]
    return slices.zslice(z, [z,z+1.0e-06], ylim, xlim, (2, shape[1], shape[2]))

def sagittal(grid, x=0.,
            shape=None,
            ylim=None,
            zlim=None):
    """
    Sagittal slice through a grid, with optional ylim and zlim. Otherwise,
    these are taken from a bounding for grid.

    Shape refers to size of array in sampling the region by an interpolator.
    TODO: make these slices 2dslices, rather than this hack +1.0e-06
    """
    if ylim is None:
        ylim = slices.bounding_box(grid)[1]
    if zlim is None:
        zlim = slices.bounding_box(grid)[0]
    shape = shape or grid.shape
    return slices.xslice(x, zlim, ylim, [x,x+1.0e-06], (shape[0],shape[1],2))

def squeezeshape(shape):
    s = N.array(shape)
    keep = N.not_equal(s, 1)
    return tuple(s[keep])

## All pylab specific below

import pylab
import matplotlib
from neuroimaging.ui.visualization.cmap import cmap, getcmap

interpolation = traits.Trait('nearest', 'bilinear', 'blackman100',
                             'blackman256', 'blackman64', 'bicubic',
                             'sinc144', 'sinc256', 'sinc64',
                             'spline16', 'spline36')

origin = traits.Trait('lower', 'upper')

class RGBASlicePlot(Slice, traits.HasTraits):
    """
    A class that draws a slice of RGBA data. The grid instance is assumed to
    have a squeezeshape attribute representing the 'squeezed' shape of the
    data.
    """
    axes = traits.Any()
    figure = traits.Any()
    interpolation = interpolation
    origin = origin

    # traits to determine pylab axes instances
    height = traits.Trait(traits.Float())
    width = traits.Trait(traits.Float())
    xoffset = traits.Float(0.1, desc="Proportion (in matplotlib units) of figure to left and right of plotting region.")
    yoffset = traits.Float(0.1, desc="Proportion (in matplotlib units) of figure above and below plotting region.")

    mask = traits.Any(desc="Mask interpolator defined on the same output coordinate system as the image interpolator.")
    maskthresh = traits.Float(0.5, desc="Threshold for mask.")
    maskcolor = traits.ListFloat([0,0,0], desc="Color to display in masked regions.")

    transpose = traits.Trait(False, desc='Transpose data before ploting?')

    def numdata(self):
        """
        Return self.interpolator evaluated over self.grid.range.
        """
        return N.squeeze(self.interpolator(self.grid.range()))

    def __init__(self, interpolator, grid, transpose=False, **keywords):
        traits.HasTraits.__init__(self, **keywords)
        Slice.__init__(self, interpolator, grid, transpose)

    def mask_data(self, data):
        if self.mask is not None:
            alpha = N.greater_equal(N.squeeze(self.mask(self.grid.range())), 0.1)
            if self.transpose:
                alpha = alpha.T
            for i in range(3):
                data[:,:,i] = data[:,:,i]*alpha + (1 - alpha)*self.maskcolor[i]
        return data
            
    def draw(self, data=None, redraw=False, *args, **keywords):
        if data is None: data = self.RGBA(data=data)
        masked_data = self.mask_data(data)
        pylab.axes(self.axes)

        if not redraw:
            self.imshow = pylab.imshow(masked_data, interpolation=self.interpolation,
              origin=self.origin)
        else:
            self.imshow.set_data(masked_data)
            
    def getaxes(self):
        height = self.height and self.height or self.grid.squeezeshape[0]
        width = self.width and self.width or self.grid.squeezeshape[1]
        self.axes = pylab.axes([self.xoffset, self.yoffset, width, height])
        
    def RGBA(self, data=None):
        "Return the RGBA values for the interpolator over the slice's grid."
        if data is None:
            v = self.numdata()
        else:
            v = data
        if self.transpose:
            v = N.transpose(v, (1,0,2))
        return v
    

class RGBSlicePlot(RGBASlicePlot):

    alpha = traits.Float(1.0)

    def RGBA(self):
        """
        Return the RGBA values for the interpolator over the slice\'s grid.
        """
        v = self.numdata()
        if self.alpha == 1.:
            return RGBASlicePlot.RGBA(self, data=v)
        else:
            V = N.zeros(v.shape[0:2] + (4,))
            V[0:2,0:3] = v
            V[0:2,3] = self.alpha
            return RGBASlicePlot.RGBA(self, data=V)

    def draw(self, data=None, redraw=False, *args, **keywords):
        RGBASlicePlot.draw(self, data=self.RGBA(data=data), redraw=redraw)
            
class DataSlicePlot(RGBSlicePlot):

    colormap = cmap
    vmax = traits.Float(0.0)
    vmin = traits.Float(0.0)

    def _vmax_changed(self):
        self._vmin_changed()

    def _vmin_changed(self):
        self.norm = matplotlib.colors.normalize(vmax=self.vmax, vmin=self.vmin)

    def RGBA(self, data=None):
        """
        Return the RGBA values for the interpolator over the slice\'s grid.
        """
        
        if data is None:
            data = self.numdata()
        v = self.norm(data)
        _cmap = getcmap(self.colormap)
        return RGBASlicePlot.RGBA(self, data=N.array(_cmap(v)))

    def draw(self, data=None, redraw=False, *args, **keywords):
        RGBASlicePlot.draw(self, data=self.RGBA(data=data), redraw=redraw)

class SagittalPlot(DataSlicePlot):

    def __init__(self, img, x=0, shape=None, ylim=None, zlim=None, **keywords):
        self.img = img
        self.x = x
        self.shape = shape
        self.interpolator = ImageInterpolator(self.img)
        self.m = float(self.img[:].min())
        self.M = float(self.img[:].max())
        self.slice = sagittal(self.img.grid, x=self.x, shape=self.shape,
                              ylim=ylim, zlim=zlim,
                              **keywords)
    
        DataSlicePlot.__init__(self, self.interpolator, self.slice,
                                vmax=self.M,
                                vmin=self.m,
                               **keywords)


class CoronalPlot (DataSlicePlot):

    def __init__(self, img, y=0, xlim=None, zlim=None, shape=None, **keywords):
        self.img = img
        self.y = y
        self.shape = shape
        self.interpolator = ImageInterpolator(self.img)
        self.m = float(N.nanmin(self.img[:]))
        self.M = float(N.nanmax(self.img[:]))
        self.slice = coronal(self.img.grid, y=self.y, 
                             xlim=xlim, zlim=zlim,
                             shape=self.shape)
    
        DataSlicePlot.__init__(self, self.interpolator, self.slice,
                               vmax=self.M,
                               vmin=self.m,
                               **keywords)


class TransversalPlot (DataSlicePlot):

    def __init__(self, img, z=0, shape=None, xlim=None, ylim=None, **keywords):
        self.z = z
        self.img = img
        self.shape = shape
        self.interpolator = ImageInterpolator(self.img)
        self.m = float(self.img[:].min())
        self.M = float(self.img[:].max())
        self.slice = transversal(self.img.grid, z=self.z, shape=self.shape,
                                 xlim=xlim, ylim=ylim)
    
        DataSlicePlot.__init__(self, self.interpolator, self.slice,
                               vmax=self.M,
                               vmin=self.m,
                               **keywords)



