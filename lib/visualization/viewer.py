import fpformat
import neuroimaging
import numpy as N
import pylab
import neuroimaging.statistics.utils as utils
import neuroimaging.reference as reference
import slices as vizslice
import enthought.traits as traits
from cmap import cmap, interpolation

class BoxViewer(traits.HasTraits):

    x = traits.Float(N.inf)
    y = traits.Float(N.inf)
    z = traits.Float(N.inf)

    xlim = traits.List([-90.,90.])
    ylim = traits.List([-126.,90.])
    zlim = traits.List([-72.,108.])

    shape = traits.ListInt([91,109,91])

    slicenames = traits.ListStr(['coronal', 'sagittal', 'transversal'])

    # determines layout of slices

    buffer_pix = traits.Float(50.) # buffer in pixels
    z_pix = traits.Float(200.) # pixel length of z scale
    dpi = traits.Float(80.)

    interpolation = interpolation
    colormap = cmap

    """
    View an image in orthogonal coordinates, i.e. sampled on the grid
    of the MNI atlas which is the default orthogonal coordinate system.

    If default is False, then a bounding box for image is returned
    and used for the limits.
    """

    def __init__(self, image, x=None, y=None, z=None,
                 xlim=[-90.,90.],
                 ylim=[-126.,90.],
                 zlim=[-72.,108.],
                 default=False,
                 interpolation='nearest',
                 **keywords):

        self.interpolation = interpolation
        self.slices = {}

        if image.grid.mapping.output_coords.ndim != 3:
            raise ValueError, 'only 3d images can be viewed with BoxViewer'

        if default:
            self.xlim = xlim
            self.ylim = ylim
            self.zlim = zlim
            self.shape = [91,109,91]
        else:
            self.zlim, self.ylim, self.xlim = reference.slices.bounding_box(image.grid)
            self.shape = image.grid.shape


        if x is None:
            x = N.mean(self.xlim)
        if y is None:
            y = N.mean(self.ylim)
        if z is None:
            z = N.mean(self.zlim)

        if keywords.has_key('z_pix'):
            self.z_pix = keywords['z_pix']
        if keywords.has_key('buffer_pix'):
            self.z_pix = keywords['buffer_pix']
            
        _img = N.nan_to_num(image.readall())
        self.m = float(_img.min())
        self.M = float(_img.max())

        figwidth, figheight = self._setup_dims()
        self.figure = pylab.figure(figsize=(figwidth / self.dpi, figheight / self.dpi))

        self.interpolator = neuroimaging.image.interpolation.ImageInterpolator(image)
        self.cid = pylab.connect('button_press_event', self.on_click)
        traits.HasTraits.__init__(self, x=x, y=y, z=z, **keywords)

    def _setup_dims(self):

        dx = N.fabs(self.xlim[1] - self.xlim[0])
        dy = N.fabs(self.ylim[1] - self.ylim[0])
        dz = N.fabs(self.zlim[1] - self.zlim[0])

        figheight = (3 * self.buffer_pix +
                     self.z_pix * (dx + dz) / dz)

        figwidth = (3 * self.buffer_pix +
                    self.z_pix * (dy + dx) / dz)

        xwidth = dx * self.z_pix / (dz * figwidth)
        xheight = dx * self.z_pix / (dz * figheight)

        ywidth = dy * self.z_pix / (dz * figwidth)
        yheight = dy * self.z_pix / (dz * figheight)

        zwidth = dz * self.z_pix / (dz * figwidth)
        zheight = dz * self.z_pix / (dz * figheight)

        bufwidth = self.buffer_pix / figwidth
        bufheight = self.buffer_pix / figheight

        self.lengths = {}

        self.lengths[self.slicenames[0]] = (xwidth, zheight)
        self.lengths[self.slicenames[1]] = (ywidth, zheight)
        self.lengths[self.slicenames[2]] = (ywidth, xheight)
        self.lengths['color'] = (0.1, xheight)

        self.offsets = {}
        self.offsets[self.slicenames[0]] = (2 * bufwidth + ywidth,
                                            2 * bufheight + xheight)
        self.offsets[self.slicenames[1]] = (bufwidth,
                                             2 * bufheight + xheight)

        self.offsets[self.slicenames[2]] = (bufwidth,
                                            bufheight)
        self.offsets['color'] = (2 * bufwidth + ywidth,
                                 bufheight)

        self.ticks = {}
        self.ticks[self.slicenames[0]] = ((0, self.shape[2]-1),
                                 (0, self.shape[0]-1))
        self.ticks[self.slicenames[1]] = ((0, self.shape[1]-1),
                                  (0, self.shape[0]-1))
        self.ticks[self.slicenames[2]] = ((0, self.shape[1]-1),
                                     (0, self.shape[2]-1))
        self.ticks['color'] = ([], N.linspace(0,100,10).astype(N.Int))

        _str = lambda x: fpformat.fix(x, 0)
        self.ticklabels = {}
        self.ticklabels[self.slicenames[0]] = (map(_str, self.xlim),
                                               map(_str, self.zlim))
        self.ticklabels[self.slicenames[1]] = (map(_str, self.ylim),
                                               map(_str, self.zlim))
        self.ticklabels[self.slicenames[2]] = (map(_str, self.ylim),
                                               map(_str, self.xlim))

        return figwidth, figheight

    def _getslice(self, _slice):
        return vizslice.PylabDataSlice(self.interpolator,
                                       _slice,
                                       vmax=self.M,
                                       vmin=self.m,
                                       colormap=self.colormap,
                                       interpolation=self.interpolation)

    def _x_changed(self):
        _slice = neuroimaging.reference.slices.xslice(x=self.x,
                                                      xlim=self.xlim,
                                                      ylim=self.ylim,
                                                      zlim=self.zlim,
                                                      shape=self.shape)

        self._setup_slice(_slice, self.slicenames[1])

    def _y_changed(self):
        
        _slice = neuroimaging.reference.slices.yslice(y=self.y,
                                                      xlim=self.xlim,
                                                      ylim=self.ylim,
                                                      zlim=self.zlim,
                                                      shape=self.shape)
        self._setup_slice(_slice, self.slicenames[0])
        
    def _z_changed(self):
        _slice = neuroimaging.reference.slices.zslice(z=self.z,
                                                      xlim=self.xlim,
                                                      ylim=self.ylim,
                                                      zlim=self.zlim,
                                                      shape=self.shape)

        self._setup_slice(_slice, self.slicenames[2])

    def draw_colorbar(self):
        width, height = self.lengths['color']
        xoffset, yoffset = self.offsets['color']
        v = N.linspace(0,1.,100) * (self.M - self.m) + self.m

        if not hasattr(self, 'colorbar'):
            self.colorbar = pylab.axes([xoffset, yoffset, width, height])
            self.colorbar.set_xticks([])
            self.colorbar.set_yticks(N.linspace(0,100,11).astype(N.Int))
            v = N.linspace(0,1.,11) * (self.M - self.m) + self.m
            self.colorbar.set_yticklabels([fpformat.fix(v[i], 1) for i in range(v.shape[0])])
        else:
            pylab.axes(self.colorbar)

        self.slices[self.slicenames[0]].RGBA(data=v)
        v = N.linspace(0,1.,100) * (self.M - self.m) + self.m
        v.shape = (100,1)
        pylab.imshow(v, origin=self.slices[self.slicenames[0]].origin)
        
    def _setup_slice(self, _slice, which):
        if not self.slices.has_key(which):
            self.slices[which] = self._getslice(_slice)
        else:
            self.slices[which].grid = _slice

        self.slices[which].width, self.slices[which].height = self.lengths[which]

        self.slices[which].xoffset, self.slices[which].yoffset = self.offsets[which]
        self.slices[which].getaxes()

        a = self.slices[which].axes
        a.set_xticks(self.ticks[which][0])
        a.set_xticklabels(self.ticklabels[which][0])

        a.set_yticks(self.ticks[which][1])
        a.set_yticklabels(self.ticklabels[which][1])

    def on_click(self, event):

        if event.inaxes == self.slices[self.slicenames[1]].axes:
            vy, vz = event.xdata, event.ydata
            vx = 0
            world = self.slices[self.slicenames[1]].grid.mapping([vz,vy,vx])
            which = self.slicenames[1]
            self.z, self.y, self.x = world
        elif event.inaxes == self.slices[self.slicenames[2]].axes:
            vy, vx = event.xdata, event.ydata
            vz = 0
            world = self.slices[self.slicenames[2]].grid.mapping([vx,vy,vz])
            which = self.slicenames[2]
            self.z, self.y, self.x = world
        elif event.inaxes == self.slices[self.slicenames[0]].axes:
            vx, vz = event.xdata, event.ydata
            vy = 0
            world = self.slices[self.slicenames[0]].grid.mapping([vz,vx,vy])
            which = self.slicenames[0]
            self.z, self.y, self.x = world
        else:
            which = None

        if which is self.slicenames[0]:
            self.slices[self.slicenames[2]].draw(redraw=True)
            self.slices[self.slicenames[1]].draw(redraw=True)
        elif which is self.slicenames[1]:
            self.slices[self.slicenames[2]].draw(redraw=True)
            self.slices[self.slicenames[0]].draw(redraw=True)
        elif which is self.slicenames[2]:
            self.slices[self.slicenames[0]].draw(redraw=True)
            self.slices[self.slicenames[1]].draw(redraw=True)

        self.draw(redraw=True)


    def draw(self, redraw=False):
        pylab.figure(num=self.figure.number)
        for imslice in self.slices.values():
            imslice.draw(redraw=redraw)
        self.draw_colorbar()
        pylab.draw()
