"""
A Slice class for visualizing slices of images.
"""

import numpy as N
from enthought import traits
from neuroimaging.reference import slices
from neuroimaging.image.interpolation import ImageInterpolator

class Slice(traits.HasTraits):

    transpose = traits.false

    def __init__(self, interpolator, grid, **keywords):
        traits.HasTraits.__init__(self, **keywords)
        self.interpolator = interpolator
        self.grid = grid
        self.drawer_args = ()
        self.drawer_keywords = {}

    def draw(self, *args, **keywords):
        """
        Draw the resulting slice.
        """
        pass


# Default MNI coordinates

zlim = slices.default_zlim
ylim = slices.default_ylim
xlim = slices.default_xlim

def coronal(image, y=0.,
            shape=(128,128),
            xlim=None,
            zlim=None):
    """
    Coronal slice through an image, with optional xlim and zlim. Otherwise,
    these are taken from a bounding for image.

    Shape refers to size of array in sampling the region by an interpolator.
    """
    if xlim is None:
        xlim = slices.bounding_box(image.grid)[2]
    if zlim is None:
        zlim = slices.bounding_box(image.grid)[0]
    return slices.yslice(y=y, ylim=[y,y+1.],
                                   xlim=xlim,
                                   zlim=zlim,
                                   shape=(shape[1],2,shape[0]))

def transversal(image, z=0.,
                shape=(128,128),
                xlim=None,
                ylim=None):
    """
    Transversal slice through an image, with optional xlim and ylim. Otherwise,
    these are taken from a bounding for image.

    Shape refers to size of array in sampling the region by an interpolator.
    """
    if xlim is None:
        xlim = slices.bounding_box(image.grid)[2]
    if ylim is None:
        ylim = slices.bounding_box(image.grid)[1]
    return slices.zslice(z=z, zlim=[z,z+1.],
                                   xlim=xlim,
                                   ylim=ylim,
                                   shape=(2,shape[0],shape[1]))

def sagittal(image, x=0.,
            shape=(128,128),
            ylim=None,
            zlim=None):
    """
    Sagittal slice through an image, with optional ylim and zlim. Otherwise,
    these are taken from a bounding for image.

    Shape refers to size of array in sampling the region by an interpolator.

    """
    if ylim is None:
        ylim = slices.bounding_box(image.grid)[1]
    if zlim is None:
        zlim = slices.bounding_box(image.grid)[0]
    return slices.xslice(x=x, xlim=[x,x+1.],
                                   ylim=ylim,
                                   zlim=zlim,
                                   shape=(shape[1],shape[0],2))



## All pylab specific below

import pylab
import matplotlib
import numpy as N
from cmap import cmap, getcmap

interpolation = traits.Trait('nearest', 'bilinear', 'blackman100',
                             'blackman256', 'blackman64', 'bicubic',
                             'sinc144', 'sinc256', 'sinc64',
                             'spline16', 'spline36')

origin = traits.Trait('lower', 'upper')

class RGBASlicePlot(Slice):
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

    def mask_data(self, data):
        if self.mask is not None:
            alpha = N.greater_equal(N.squeeze(self.mask(self.grid.range())), 0.1)
            if self.transpose:
                alpha = N.transpose(alpha)
            for i in range(3):
                data[:,:,i] = data[:,:,i]*alpha + (1 - alpha)*self.maskcolor[i]
        return data
            

    def draw(self, data=None, redraw=False, *args, **keywords):
        if data is None: data = self.RGBA(data=data)

        masked_data = self.mask_data(data)
        pylab.axes(self.axes)

        if not redraw:
            self.imshow = pylab.imshow(masked_data, interpolation=self.interpolation,
              aspect='auto', origin=self.origin)
        else:
            self.imshow.set_data(masked_data)
            
    def getaxes(self):
        height = self.height and self.height or self.grid.squeezeshape[0]
        width = self.width and self.width or self.grid.squeezeshape[1]
        self.axes = pylab.axes([self.xoffset, self.yoffset, width, height])
        
    def RGBA(self, data=None):
        "Return the RGBA values for the interpolator over the slice's grid."
        if data is None:
            v = N.squeeze(self.interpolator(self.grid.range()))
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
        v = N.squeeze(self.interpolator(self.grid.range()))
        if self.alpha == 1.:
            return RGBASlicePlot.RGBA(self, data=v)
        else:
            V = N.zeros(v.shape[0:2] + (4,), N.float64)
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
            data = N.squeeze(self.interpolator(self.grid.range()))
        v = self.norm(data)
        _cmap = getcmap(self.colormap)
        return RGBASlicePlot.RGBA(self, data=N.array(_cmap(v)))

    def draw(self, data=None, redraw=False, *args, **keywords):
        RGBASlicePlot.draw(self, data=self.RGBA(data=data), redraw=redraw)

class SagittalPlot(DataSlicePlot):

    x = traits.Float(0.)
    xlim = traits.ListFloat(xlim)
    ylim = traits.ListFloat(ylim)
    shape = (128,128)

    def __init__(self, img, **keywords):
        traits.HasTraits.__init__(self, **keywords)
        self.img = img
        self.interpolator = ImageInterpolator(self.img)
        self.m = float(self.img.readall().min())
        self.M = float(self.img.readall().max())
        self.slice = sagittal(self.img, x=self.x, ylim=self.ylim,
                              zlim=self.zlim, shape=self.shape)
    
        DataSlicePlot.__init__(self, self.interpolator, self.slice,
                                vmax=self.M,
                                vmin=self.m,
                                colormap=self.colormap,
                                interpolation=self.interpolation)


class CoronalPlot (DataSlicePlot):

    x = traits.Float(0.)
    xlim = traits.ListFloat(xlim)
    zlim = traits.ListFloat(zlim)
    shape = (128,128)

    def __init__(self, img, **keywords):
        traits.HasTraits.__init__(self, **keywords)
        self.img = img
        self.interpolator = ImageInterpolator(self.img)
        self.m = float(self.img.readall().min())
        self.M = float(self.img.readall().max())
        self.slice = coronal(self.img, x=self.x, xlim=self.xlim,
                              zlim=self.zlim, shape=self.shape)
    
        DataSlicePlot.__init__(self, self.interpolator, self.slice,
                                vmax=self.M,
                                vmin=self.m,
                                colormap=self.colormap,
                                interpolation=self.interpolation)


class TransversalPlot (DataSlicePlot):

    z = traits.Float(0.)
    xlim = traits.ListFloat(xlim)
    ylim = traits.ListFloat(ylim)
    shape = (128,128)

    def __init__(self, img, **keywords):
        traits.HasTraits.__init__(self, **keywords)
        self.img = img
        self.interpolator = ImageInterpolator(self.img)
        self.m = float(self.img.readall().min())
        self.M = float(self.img.readall().max())
        self.slice = transversal(self.img, z=self.z, ylim=self.ylim,
                                 xlim=self.xlim, shape=self.shape)
    
        DataSlicePlot.__init__(self, self.interpolator, self.slice,
                                vmax=self.M,
                                vmin=self.m,
                                colormap=self.colormap,
                                interpolation=self.interpolation)



