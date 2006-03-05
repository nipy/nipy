"""

This is based heavily on cursor_demo.py, pcolor_demo2.py

"""

import enthought.traits as traits
import numpy as N
import matplotlib, sets
matplotlib.numerix = N

from neuroimaging.statistics.utils import reduceall
import pylab
from matplotlib.patches import Patch

norm = matplotlib.colors.normalize()

def default_combine_layers(layers):
    '''
    Combine one or more layers for overlaying functional data on anatomical.
    '''

    shape = layers[0].shape
    layers[0].getRGBA()
    rgba = N.zeros((N.product(layers[0].shape), 4), N.Float)
    alpha = 0.

    for layer in layers:
        if layer.shape != shape:
            raise ValueError, 'layers must have the same shape!'
        layer.getRGBA()
        layer.rgba.shape = rgba.shape
        _alpha = layer.rgba[:,3]
        alpha = alpha + _alpha
        _alpha = N.multiply.outer(_alpha, N.ones((3,)))
        rgba[:,0:3] = rgba[:,0:3] + _alpha * layer.rgba[:,0:3] 

    for i in range(3):
        rgba[:,i] = rgba[:,i] / alpha

    rgba[:,3] = N.ones((rgba.shape[0],))
    rgba.shape = layers[0].data.shape + (4,)
    return rgba

interpolation = traits.Trait('nearest', 'bilinear', 'blackman100',
                             'blackman256', 'blackman64', 'bicubic',
                             'sinc144', 'sinc256', 'sinc64',
                             'spline16', 'spline36')

from cmap import cmap, getcmap

class Viewer(traits.HasTraits):

    alpha = traits.Float(1.0)
    interpolation = interpolation
    cmap = cmap
    slices = traits.List()
    vmax = traits.Float()
    vmin = traits.Float()
    verbose = traits.false
    colorbar = traits.true
    view = traits.Trait('3 slice', 'montage')
    nx = traits.Int(1)
    ny = traits.Int(1)
    start = traits.Int(0)
    step = traits.Int(1)
    crosshair = traits.true
    crosshair_color = traits.List([1.0,0.0,0.0,0.4])
    crosshair_lw = traits.Float(2.0)
    origin = traits.Trait('lower')
    atlas = traits.Any()

    def __init__(self, data, explorer=None, args=(), keywords={}, flip=[False]*3, **extra):

        try:
            self.data = data.readall()
        except:
            self.data = data

        self.shape = data.shape

        traits.HasTraits.__init__(self, **extra)

        self.flip = flip # which axes do we flip?

        if explorer:
            self.set_explorer(explorer, args, keywords)

        for i in range(3):
            if self.flip[i]:
                self.data = _flip(self.data, i)

        self.slices = list(N.array(self.data.shape, N.Int) / 2)

        self.origin = 'lower'

        if not hasattr(self, 'width'):
            self.width, self.height = self._calcsizes(**extra)[0:2]
        self.figure = pylab.figure(figsize=(self.width, self.height))

        if self.vmin == self.vmax:

            self.vmax = reduceall(N.maximum, self.data)
            self.vmin = reduceall(N.minimum, self.data)
        if self.vmax < self.vmin:
            self.vmax, self.vmin = (self.vmin, self.vmax)
            
        self.cid = pylab.connect('button_press_event', self.on_click)

        self.im = range(3)
        self._crosshair = range(3)
        self.slicedata = range(3)

        if hasattr(self, 'explorer'):
            _args = (self, None,) + self.explorer_args 
            self.explorer(*_args, **self.explorer_keywords)

    def set_explorer(self, explorer, args, keywords):
        self.explorer = explorer
        self.explorer_args = args
        self.explorer_keywords = keywords

    def _calcsizes(self, *args, **extra):

        if self.view == '3 slice':
            return self._3slice_calcsizes(*args, **extra)
        else:
            return self._montage_calcsizes(*args, **extra)

    def _3slice_calcsizes(self, buffer=20, top=20, bottom=20, right=20, left=20, dpi=80, zoom=3.0, **keywords):
        """
        by default, in 91x109x91 coords
        transverse -- image 0 -- lower left
        saggital -- image 1 -- upper right
        coronal -- image 2 -- upper left
        """

        if not hasattr(self, 'scaling'):
            if not keywords.has_key('scaling'):
                self.scaling = [1.] * 3
            else:
                self.scaling = keywords['scaling']

        if not hasattr(self, 'axes'):
            if not keywords.has_key('axes'):
                self.axes = ((2,1), (0,2), (0,1))
            else:
                self.axes = keywords['axes']

        sizes = []
        self.shapes = {}

        for axis in self.axes:
            self.shapes[axis] = [self.shape[axis[0]], self.shape[axis[1]]]
            sizes += [(self.shape[axis[0]] * float(self.scaling[axis[0]]), self.shape[axis[1]] * float(self.scaling[axis[1]]))]
        
        lli = 0
        uli = 2
        uri = 1
        
        pheight = (sizes[lli][0] + sizes[2][0] + buffer + bottom + top)
        pwidth = (sizes[uri][1] + sizes[uli][1] + buffer + right + left) 
        
        height = pheight * 1. * zoom / dpi
        width = pwidth * 1. * zoom / dpi

        # lower left -- transversal

        lldy = sizes[lli][0] / pheight
        lly = bottom / pheight
        llx = left / pwidth

        # upper left -- saggital

        uldy = sizes[uli][0] / pheight
        uly = (bottom + buffer + sizes[lli][0]) / pheight

        uldx = sizes[uli][1] / pwidth
        ulx = left * 1. / pwidth

        # upper right -- coronal

        urdx = sizes[uri][1] / pwidth
        urx = (left + sizes[uli][1] + buffer) / pwidth

        ury = (sizes[lli][0] + bottom + buffer) / pheight
        urdy = uldy

        lldx = uldx
        
        ll = (llx, lly, lldx, lldy) # lower left
        ul = (ulx, uly, uldx, uldy) # upper left
        ur = (urx, ury, urdx, urdy) # upper right
        
        self.ll = ll
        self.ur = ur
        self.ul = ul
        
        return width, height, ll, ur, ul

    def _3slice_get_slices(self, axes, _slices):

        _slices = list(_slices)
        dim = _leftout(axes)
        slices = [0]*3
        slices[dim] = self.slices[dim]

        if self.origin is 'upper':
            _slices[1] = self.shape[axes[0]] - 1 - _slices[1]

        for i in range(2):
            slices[axes[1-i]] = _slices[i]

        for i in range(3):
            if self.flip[i]:
                slices[i] = self.shape[i] - 1 - slices[i]

        return slices

    def on_click(self, event):

        pylab.figure(num=self.figure.number)

        if self.view == '3 slice':
            self._3slice_event_handler(event)
        else:
            self._montage_event_handler(event)

        self.draw()

        if hasattr(self, 'explorer'):
            _args = (self, event) + self.explorer_args 
            self.explorer(*_args, **self.explorer_keywords)

    def draw(self, slices=None, *args, **extra):

        if hasattr(self, 'figure'):
            pylab.figure(num=self.figure.number)

        self.origin = 'lower'

        if slices is None:
            slices = self.slices

        if self.view == '3 slice':
            self._3slice_draw(slices, *args, **extra)
        else:
            self._montage_draw(*args, **extra)

    def _montage_event_handler(self, event):
        '''

        '''
        
        if event.inaxes:
            self.slices = self._montage_get_slices(event.xdata, event.ydata)

    def _montage_get_slices(self, xdata, ydata):

        i = self.ny - 1 - N.floor(ydata / self.shape[2])
        j = N.floor(xdata / self.shape[1])           
        if self.view == 'upper':
            k = self.nx * i + j
        else:
            k = self.nx * (self.ny - 1 - i) + j

        # TO DO: check LR!

        y = int(ydata - (self.ny - 1 - i) * self.shape[2])
        x = int(xdata - j * self.shape[1])

        self.x1, self.y1 = [xdata]*2, [(self.ny - 1 - i) * self.shape[2],
                                       (self.ny - 1 - i) * self.shape[2] + self.shape[2] - 1]
        self.x2, self.y2 = [j * self.shape[1], j * self.shape[1] + self.shape[1] - 1], [ydata]*2
        
        slices = list(N.array([k * self.step + self.start, y, x], N.Int))

        for i in range(3):
            if self.flip[i]:
                slices[i] = self.shape[i] - 1 - slices[i]

        return slices

    def _3slice_event_handler(self, event):
        if event.inaxes is not None:
            _slices = (int(event.xdata), int(event.ydata))
            location = event.inaxes.get_position()

            slices = self.slices
            if location == self.im[0].axes.get_position():
                slices = self._3slice_get_slices(self.axes[0], _slices)
            elif location == self.im[1].axes.get_position():
                slices = self._3slice_get_slices(self.axes[1], _slices)
            elif location == self.im[2].axes.get_position():
                slices = self._3slice_get_slices(self.axes[2], _slices)
            elif location == self.cax.get_position():
                pass

            if self.verbose:
                print 'Slices:', slices

            if slices:
                self._3slice_draw(slices, redraw=True)

    def _3slice_get_data(self, slices, axes):

        dim = _leftout(axes)
        _transpose = False

        _index = slices[dim]

        if dim == 0:
            slicedata = self.data[_index]
            if axes == (2,1):
                _transpose = True
        elif dim == 1:
            slicedata = self.data[:,_index,:]
            if axes == (2, 0):
                _transpose = True
        else:
            slicedata = self.data[:,:,_index]
            if axes == (1, 0):
                _transpose = True

        if _transpose:
            if len(slicedata.shape) == 3:
                slicedata = N.transpose(slicedata, (1, 0, 2))
            else:
                slicedata = N.transpose(slicedata)
                
        return slicedata

    def _cmap_changed(self, draw=True):
        try:
            if self.view == '3 slice':

                for i in range(3):
                    self.im[i].cmap = getcmap(self.cmap)
                axes = pylab.axes(self.cax)
                pylab.imshow(N.multiply.outer(pylab.frange(self.vmax, self.vmin, npts=100), (2,)), extent=(0, 1, self.vmin, self.vmax), cmap=getcmap(self.cmap), aspect='free')
                axes.set_xticklabels([])
                pylab.xticks([])

            elif self.view == 'montage':
                try:
                    self.p.cmap = getcmap(self.cmap)
                except:
                    pass 

        except:
            pass

        if draw:
            self.draw()

    def _vmax_changed(self):
        try:
            for i in range(3):
                self.im[i].norm.vmax = self.vmax
            self._cmap_changed(draw=True)
        except:
            pass

    def _vmin_changed(self):
        try:
            for i in range(3):
                self.im[i].norm.vmin = self.vmin
            self._cmap_changed(draw=True)
        except:
            pass

    def anytrait_changed(self):
        self.draw()

    def _montage_calcsizes(self, buffer=20, top=40, bottom=20, right=40, left=20, dpi=80, scale=1.0, **extra):

        mx = self.shape[2]
        my = self.shape[1]

        pwidth = self.nx * mx + top + bottom
        pheight = self.ny * my + left + right

        width = scale * pwidth * 1. / dpi
        height = scale * pheight * 1. / dpi
        position = (left * 1. / pwidth,
                    bottom * 1. / pheight,
                    self.nx * mx * 1. / pwidth,
                    self.ny * my * 1. / pheight)

        return width, height, position

    def _montage_get_data(self):

        mx = self.shape[2]
        my = self.shape[1]

        if len(self.data.shape) == 4:
            self._data = N.zeros((self.ny * my, self.nx * mx, 4), N.Float)
        else:
            self._data = N.zeros((self.ny * my, self.nx * mx), N.Float)

        for i in range(self.ny):
            for j in range(self.nx):
                k = self.start + self.step * (self.nx * i + j)
                if k < self.data.shape[0]:
                    self._data[i*my:(i+1)*my,j*mx:(j+1)*mx] = self.data[k]

    def _montage_get_crosshair(self):
        y = (self.slices[0] - self.start) / self.nx
        x = (self.slices[0] - self.start) - y * self.nx

        self._montage_get_slices(x * self.shape[2] + self.slices[2], y * self.shape[1] + self.slices[1])

    def _montage_draw(self, redraw=True):

        ## TO DO: set slices!

        mx = self.shape[2]
        my = self.shape[1]

        self.position = self._montage_calcsizes()[2]

        if not hasattr(self, '_data'):
            self._montage_get_data()

        if not redraw:
            if len(self.data.shape) == 4:
                self.p = pylab.imshow(self._data, cmap=getcmap(self.cmap), interpolation=self.interpolation, alpha=self.alpha, vmax=self.vmax, vmin=self.vmin, origin=self.origin, aspect='free')
            else:
                self.p = pylab.imshow(self._data, interpolation=self.interpolation, origin=self.origin, aspect='free')

            for I in range(1, self.nx):
                pylab.plot([I * mx]*2, [0, self._data.shape[0]], color=(0.3,0.3,0.3,1.0))
            for J in range(0, self.ny - 1):
                pylab.plot([0, self._data.shape[1]], [(self.ny - 1 - J) * my]*2, color=(0.3,0.3,0.3,1.0))

            axes = pylab.axes(self.p.axes)
            axes.set_xlim(0, self._data.shape[1])
            axes.set_ylim(0, self._data.shape[0])

        else:
            self.p.set_data(self._data)

        self.p.axes.set_xticks([])
        self.p.axes.set_xticklabels([])
        self.p.axes.set_yticks([])
        self.p.axes.set_yticklabels([])

        if self.colorbar:
            self.cax = pylab.axes([0.95, 0.1, 0.03, 0.3])
            pylab.imshow(N.multiply.outer(pylab.frange(self.vmax, self.vmin, npts=100), (2,)), extent=(0, 1, self.vmin, self.vmax), cmap=getcmap(self.cmap), aspect='free')
            self.cax.set_xticklabels([])
            pylab.xticks([])

        if self.crosshair:
            if self._crosshair == [0, 1, 2]:
                pylab.axes(self.p.axes)
                if not hasattr(self, 'x1'):
                    self._montage_get_crosshair()
                
                self._crosshair = pylab.plot(self.x1, self.y1, self.x2, self.y2, color=self.crosshair_color, linewidth=self.crosshair_lw)
            else:

                l1, l2 = self._crosshair

                l1.set_xdata(self.x1)
                l1.set_ydata(self.y1)

                l2.set_xdata(self.x2)
                l2.set_ydata(self.y2)

        pylab.draw()

    def show(self):
        self.draw(redraw=False)

    def _3slice_draw(self, slices, redraw=True, **extra):
        self.slices = list(slices)

        sizes = self._calcsizes(**extra)[2:]

        for i in range(3):

            self.slicedata[i] = self._3slice_get_data(slices, self.axes[i])
            if not redraw:

                a = pylab.axes(sizes[i])
                if len(self.slicedata[i].shape) == 2:
                    self.im[i] = pylab.imshow(self.slicedata[i], cmap=getcmap(self.cmap), interpolation=self.interpolation, alpha=self.alpha, vmax=self.vmax, vmin=self.vmin, origin=self.origin, aspect='free')
                else:
                    self.im[i] = pylab.imshow(self.slicedata[i], interpolation=self.interpolation, origin=self.origin, aspect='free')

                self.im[i].axes.set_xticklabels([])
                self.im[i].axes.set_yticklabels([])

                if self.crosshair:
                    pylab.axes(self.im[i].axes)
                    _x, _y = self.axes[i]
                    x1, y1 = [self.slices[_y] + 0.5]*2, [0,self.shape[_x]]
                    x2, y2 = [0,self.shape[_y]], [self.slices[_x] + 0.5]*2

                    if self.origin is 'upper':
                        y2 = [self.shape[_x] - 1 - self.slices[_x] + 0.5]*2

                    self._crosshair[i] = pylab.plot(x1, y1, x2, y2, color=self.crosshair_color, linewidth=self.crosshair_lw)


            else:
                 self.im[i].set_data(self.slicedata[i])

                 axes = pylab.axes(self.im[i].axes)
                 _shape = self.slicedata[i].shape

                 axes.set_xlim(0, _shape[1])
                 axes.set_ylim(0, _shape[0])

                 if self.crosshair:
                     _x, _y = self.axes[i]

                     x1, y1 = [self.slices[_y] + 0.5]*2, [0,self.shape[_x]]
                     x2, y2 = [0,self.shape[_y]], [self.slices[_x] + 0.5]*2

                     if self.origin is 'upper':
                         y2 = [self.shape[_x] - 1 - self.slices[_x] + 0.5]*2

                     l1, l2 = self._crosshair[i]

                     l1.set_xdata(x1)
                     l1.set_ydata(y1)

                     l2.set_xdata(x2)
                     l2.set_ydata(y2)
                    
                 pylab.draw()

        if self.colorbar:
            if not hasattr(self, 'cax'):
                self.cax = pylab.axes([0.87, 0.1, 0.03, 0.3])

                pylab.imshow(N.multiply.outer(pylab.frange(self.vmax, self.vmin, npts=100), (2,)), extent=(0, 1, self.vmin, self.vmax), cmap=getcmap(self.cmap), aspect='free')

                a = pylab.axes(self.cax)
                a.set_xticklabels([])
                pylab.xticks([])

class FunctionalOverlay(Viewer):
    anat_cmap = cmap
    thresh = traits.Float(0.0)

    def __init__(self, anatomy, functional, anat_cmap='gray', flip=[False]*3, **extra):
        self.anat_cmap = anat_cmap

        self.anatomy = DataLayer(anatomy.readall(**extra), cmap=self.anat_cmap)

        self.functional = DataLayer(functional.readall(**extra), cmap=self.cmap)
        self.functional.alpha = N.greater_equal(N.abs(self.functional.data), self.thresh)

        for i in range(3):
            if flip[i]:
                self.functional.data = _flip(self.functional.data, i)
                self.anatomy.data = _flip(self.anatomy.data, i)

        Viewer.__init__(self, self.functional.data, **extra)
        self.flip = flip

        self.functional.getRGBA()
        self.anatomy.getRGBA()

        alpha = self.functional.alpha
        self.data = self.functional.rgba * 0.
        self.scaling = N.array(anatomy.step)


        for i in range(4):
            self.data[:,:,:,i] = self.functional.rgba[:,:,:,i] * alpha + self.anatomy.rgba[:,:,:,i] * (1. - alpha)


    def anytrait_changed(self):
        if hasattr(self, 'functional'):
            alpha = self.functional.alpha
            for i in range(4):
                self.data[:,:,:,i] = self.functional.rgba[:,:,:,i] * alpha + self.anatomy.rgba[:,:,:,i] * (1. - alpha)

        try:
            self.draw(slices=self.slices, redraw=True)
        except:
            pass

    def _cmap_changed(self):
        if hasattr(self, 'functional'):
            self.functional.cmap = self.cmap
            
        Viewer._cmap_changed(self)
        
    def _anat_cmap_changed(self):
        if hasattr(self, 'anatomy'):
            self.anatomy.cmap = self.anat_cmap
        Viewer._cmap_changed(self)
        
    def _thresh_changed(self):
        if hasattr(self, 'functional'):
            self.functional.alpha = N.greater_equal(N.abs(self.functional.data), self.thresh).astype(N.Int)
            self.functional.getRGBA()


class MaskedViewer(Viewer):

    mask_color = traits.List([0.0,0.0,0.0,1.0]) # would be RGBAColor if traits.ui

    def __init__(self, data, mask, flip=[False]*3, **extra):
        self.functional = DataLayer(data.readall(**extra), cmap=self.cmap)
        self.mask = mask.readall(**extra)

        for i in range(3):
            if flip[i]:
                self.functional.data = _flip(self.functional.data, i)
                self.mask = _flip(self.mask, i)

        Viewer.__init__(self, self.functional.data, **extra)
        self.flip = flip

        self.scaling = N.array(data.step)

        self.functional.cmap = self.cmap
        self.draw(slices=self.slices, redraw=False)
        self.functional.getRGBA(force=True)

        self.data = 0. * self.functional.rgba

        self._cmap_changed()
        self.draw(slices=self.slices)

    def _mask_color_changed(self):
        self._maskdata()
        self.draw(slices=self.slices)

    def _cmap_changed(self):
        self.functional.cmap = self.cmap
        try:
            self._maskdata()
        except:
            pass
        
        Viewer._cmap_changed(self)
        
    def _maskdata(self):
        self.functional.getRGBA()

        for i in range(4):
            self.data[:,:,:,i] = self.functional.rgba[:,:,:,i] * self.mask + (1.0 - self.mask) * self.mask_color[i]

    def anytrait_changed(self):
        self._maskdata()
        self.draw(slices=self.slices)

class DataLayer:
    '''A class representing a layer, to be used to combine anatomy and functional data, for instance.'''
    cmap = cmap
    vmax = traits.Float(1.0)
    vmin = traits.Float(0.0)
    
    def __init__(self, data, alpha=1., cmap='spectral', **extra):
        self.shape = data.shape
        self.ndim = len(self.shape)

        for key, value in extra.items():
            setattr(self, key, value)

        self.data = data
        self.alpha = alpha
        self.cmap = cmap
        self.vmax = reduceall(N.maximum, self.data)
        self.vmin = reduceall(N.minimum, self.data)
        self.range = (self.vmax - self.vmin)

    def _vmax_changed(self):
        _cmap = getcmap(self.cmap)
        _cmap.vmax = self.vmax

    def _vmin_changed(self):
        _cmap = getcmap(self.cmap)
        _cmap.vmin = self.vmin

    def _cmap_changed(self):
        self.getRGBA(force=True)

    def norm(self, x):
        y = (x - self.vmin) / self.range
        return y

    def getRGBA(self, force=False):
        '''Based on the normalization and the alpha mask, return the space+4d RGBA array associated to a given layer.'''

        _cmap = getcmap(self.cmap)

        if not hasattr(self, 'rgba') or force:
            self.rgba = N.zeros((N.product(self.data.shape), 4), N.Float)
            self.rgba = N.array(_cmap(self.norm(self.data.flat)))
            if hasattr(self.alpha, 'shape'):
                self.rgba[:,3] = self.alpha.flat
            else:
                self.rgba[:,3] = self.alpha
            self.rgba.shape = self.shape + (4,)


class multipleLinePlot(traits.HasTraits):
    '''A simplistic plot to plat many time courses in one plot.'''

    offset = traits.Float(1.17)
    shift = traits.Float(0.1)
    labels = traits.ListStr()

    def __init__(self, fns, time, labels=None, **extra):
        self.fns = list(fns)
        self.time = time
        try:
            self.numRows = len(fns)
        except:
            self.numRows = 1

        if labels is None:
            self.labels = [''] * self.numRows
        else:
            self.labels = labels

    def anytrait_changed(self):
        if hasattr(self, 'a'):
            self.draw()

    def draw(self, **extra):
        data = []
        for i in range(self.numRows):
            _data = self.fns[i](self.time)
            _data = _data - min(_data)
            _range = max(_data) - min(_data)
            if _range > 0:
                _data = _data / _range
                _data = _data - N.add.reduce(_data) / _data.shape[0]
            else:
                _data = N.zeros(len(self.time), N.Float)
            y = i * self.offset + self.shift
            p = pylab.plot(self.time, _data + y, **extra)
            pylab.draw()

        self.a = p[0].axes
        self.a.set_yticks(range(self.numRows))
        if self.labels is not None:
            self.a.set_yticklabels(self.labels)

def _flip(data, dim):
    if dim == 0:
        _data = data[-1::-1,:,:]
    elif dim == 1:
        _data = data[:,-1::-1,:]
    elif dim == 2:
        _data = data[:,:,-1::-1]

    return _data
    
class multipleSlices(Viewer):

    view = traits.Trait('montage')
    zslice = traits.Int(0)

    def __init__(self, images, explorer=None, args=(), keywords={}, nx=1, ny=1, flip=[False]*3, **extra):

        extra['start'] = 0
        extra['step'] = 1

        self.nx = nx
        self.ny = ny

        self.images = images
        _key = images.keys()[0]
        _template = images[_key]

        self.data = N.zeros([self.nx * self.ny] + _template.shape[1:], N.Float) # dummy data

        Viewer.__init__(self, self.data, explorer=explorer, args=args, keywords=keywords, scaling = _template.step, flip=flip, **extra)

        self._montage_get_data()

        self.draw(redraw=False)
        for key, value in extra.items():
            setattr(self, key, value)
        
    def _montage_get_data(self):

        for i in range(self.ny):
            for j in range(self.nx):
                if self.images.has_key((i,j)):
                    self.data[i * self.nx + j] = self.images[(i,j)].getslice((self.zslice,))

        for i in range(3):
            if self.flip[i]:
                self.data = _flip(self.data, i)
                
        Viewer._montage_get_data(self)
        
    def _zslice_changed(self):
        try:
            self._montage_get_data()
            self.draw()
        except:
            pass



def _leftout(axes):
    return list(sets.Set(range(3)).difference(axes))[0]

