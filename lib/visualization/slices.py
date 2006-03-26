"""
A Slice class for visualizing slices of images.
"""

import numpy as N
import enthought.traits as traits
import neuroimaging.reference as reference
import neuroimaging.image as image

class Slice(traits.HasTraits):

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

zlim = reference.slices.default_zlim
ylim = reference.slices.default_ylim
xlim = reference.slices.default_xlim

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
        xlim = reference.slices.bounding_box(image.grid)[2]
    if zlim is None:
        zlim = reference.slices.bounding_box(image.grid)[0]
    return reference.slices.yslice(y=y, ylim=[y,y+1.],
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
        xlim = reference.slices.bounding_box(image.grid)[2]
    if ylim is None:
        ylim = reference.slices.bounding_box(image.grid)[1]
    return reference.slices.zslice(z=z, zlim=[z,z+1.],
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
        ylim = reference.slices.bounding_box(image.grid)[1]
    if zlim is None:
        zlim = reference.slices.bounding_box(image.grid)[0]
    return reference.slices.xslice(x=x, xlim=[x,x+1.],
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

class PylabRGBASlice(Slice):

    """
    A class that draws a slice of RGBA data. The grid instance is assumed to have
    a squeezeshape attribute representing the \'squeezed\' shape of the data.
    """

    axes = traits.Any()
    figure = traits.Any()
    interpolation = interpolation
    origin = origin

    # traits to determine pylab axes instance

    height = traits.Trait(traits.Float())
    width = traits.Trait(traits.Float())
    xoffset = traits.Float(0.1)
    yoffset = traits.Float(0.1)

    mask = traits.Any()
    maskthresh = traits.Float(0.1)
    maskcolor = traits.ListFloat([0,0,0])

    def draw(self, data=None, redraw=False, *args, **keywords):
        if data is None:
            data = self.RGBA(data=data)

        if self.mask is not None:
            alpha = N.greater_equal(N.squeeze(self.mask(self.grid.range())), 0.1)
            for i in range(3):
                data[:,:,i] = data[:,:,i] * alpha + (1 - alpha) * self.maskcolor[i]
            
        pylab.axes(self.axes)
        if not redraw:
            self.imshow = pylab.imshow(data,
                                       interpolation=self.interpolation,
                                       aspect='free',
                                       origin=self.origin)
        else:
            self.imshow.set_data(data)
            
    def getaxes(self):
        
        if self.height:
            height = self.height 
        else:
            height = self.grid.squeezeshape[0]

        if self.width:
            width = self.width
        else:
            width = self.grid.squeezeshape[1]

        self.axes = pylab.axes([self.xoffset, self.yoffset, width, height])
        
    def RGBA(self, data=None):
        """
        Return the RGBA values for the interpolator over the slice\'s grid.
        """
        if data is None:
            data = N.squeeze(self.interpolator(self.grid.range()))
        else:
            data = data
        return data

class PylabRGBSlice(PylabRGBASlice):

    alpha = traits.Float(1.0)

    def RGBA(self):
        """
        Return the RGBA values for the interpolator over the slice\'s grid.
        """
        v = N.squeeze(self.interpolator(self.grid.range()))
        if self.alpha == 1.:
            return v
        else:
            V = N.zeros(v.shape[0:2] + (4,), N.Float)
            V[0:2,0:3] = v
            V[0:2,3] = self.alpha
            return V

    def draw(self, data=None, redraw=False, *args, **keywords):
        PylabRGBASlice.draw(self, data=self.RGBA(data=data), redraw=redraw)
            
class PylabDataSlice(PylabRGBSlice):

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
        return N.array(_cmap(v))

    def draw(self, data=None, redraw=False, *args, **keywords):
        PylabRGBASlice.draw(self, data=self.RGBA(data=data), redraw=redraw)

class PylabSagittal(PylabDataSlice):

    x = traits.Float(0.)
    xlim = traits.ListFloat(xlim)
    ylim = traits.ListFloat(ylim)
    shape = (128,128)

    def __init__(self, img, **keywords):
        traits.HasTraits.__init__(self, **keywords)
        self.img = img
        self.interpolator = image.interpolation.ImageInterpolator(self.img)
        self.m = float(self.img.readall().min())
        self.M = float(self.img.readall().max())
        self.slice = sagittal(self.img, x=self.x, ylim=self.ylim,
                              zlim=self.zlim, shape=self.shape)
    
        PylabDataSlice.__init__(self, self.interpolator, self.slice,
                                vmax=self.M,
                                vmin=self.m,
                                colormap=self.colormap,
                                interpolation=self.interpolation)


class PylabCoronal(PylabDataSlice):

    x = traits.Float(0.)
    xlim = traits.ListFloat(xlim)
    zlim = traits.ListFloat(zlim)
    shape = (128,128)

    def __init__(self, img, **keywords):
        traits.HasTraits.__init__(self, **keywords)
        self.img = img
        self.interpolator = image.interpolation.ImageInterpolator(self.img)
        self.m = float(self.img.readall().min())
        self.M = float(self.img.readall().max())
        self.slice = coronal(self.img, x=self.x, xlim=self.xlim,
                              zlim=self.zlim, shape=self.shape)
    
        PylabDataSlice.__init__(self, self.interpolator, self.slice,
                                vmax=self.M,
                                vmin=self.m,
                                colormap=self.colormap,
                                interpolation=self.interpolation)


class PylabTransversal(PylabDataSlice):

    z = traits.Float(0.)
    xlim = traits.ListFloat(xlim)
    ylim = traits.ListFloat(ylim)
    shape = (128,128)

    def __init__(self, img, **keywords):
        traits.HasTraits.__init__(self, **keywords)
        self.img = img
        self.interpolator = image.interpolation.ImageInterpolator(self.img)
        self.m = float(self.img.readall().min())
        self.M = float(self.img.readall().max())
        self.slice = transversal(self.img, z=self.z, ylim=self.ylim,
                                 xlim=self.xlim, shape=self.shape)
    
        PylabDataSlice.__init__(self, self.interpolator, self.slice,
                                vmax=self.M,
                                vmin=self.m,
                                colormap=self.colormap,
                                interpolation=self.interpolation)



