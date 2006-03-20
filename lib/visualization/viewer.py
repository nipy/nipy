import neuroimaging
import numpy as N
import pylab
import neuroimaging.statistics.utils as utils
import neuroimaging.visualization.slices as vizslice
import enthought.traits as traits

class BoxViewer(traits.HasTraits):

    x = traits.Float(N.inf)
    y = traits.Float(N.inf)
    z = traits.Float(N.inf)

    xlim = traits.List([-90.,90.])
    ylim = traits.List([-126.,90.])
    zlim = traits.List([-72.,108.])

    shape = traits.ListInt([91,109,91])

    # determines layout of slices

    bufferlength = traits.Float(30.)
    imgheight = traits.Float(200.) # pixel length of z scale
    dpi = traits.Float(80.)

    """
    View an image in MNI coordinates, i.e. sampled on the grid of the MNI atlas. 
    """

    def _getslice(self, _slice):
        return vizslice.PylabDataSlice(self.interpolator,
                                       _slice,
                                       vmax=self.M,
                                       vmin=self.m,
                                       colormap='gray',
                                       interpolation='bicubic')

    def _x_changed(self):
        _slice = neuroimaging.reference.slices.sagittal(x=self.x,
                                                        xlim=self.xlim,
                                                        ylim=self.ylim,
                                                        zlim=self.zlim,
                                                        shape=self.shape)

        self._setup_sagittal(_slice)

    def _setup_sagittal(self, _slice):
        if not self.slices.has_key('sagittal'):
            self.slices['sagittal'] = self._getslice(_slice)
        else:
            self.slices['sagittal'].grid = _slice


        self.slices['sagittal'].width = (self.imgheight * self.dy) / (self.dz * self.figwidth)
        self.slices['sagittal'].xoffset = self.bufferlength / self.figwidth
        self.slices['sagittal'].yoffset = (2. * self.bufferlength + (self.imgheight * self.dx / self.dz)) / self.figheight
        self.slices['sagittal'].getaxes()

    def _y_changed(self):
        
        _slice = neuroimaging.reference.slices.coronal(y=self.y,
                                                       xlim=self.xlim,
                                                       ylim=self.ylim,
                                                       zlim=self.zlim,
                                                       shape=self.shape)
        self._setup_coronal(_slice)
        
    def _setup_dims(self):

        self.dx = N.fabs(self.xlim[1] - self.xlim[0])
        self.dy = N.fabs(self.ylim[1] - self.ylim[0])
        self.dz = N.fabs(self.zlim[1] - self.zlim[0])

        self.figheight = (3 * self.bufferlength +
                          self.imgheight * (self.dx + self.dz) / self.dz)

        self.figwidth = (3 * self.bufferlength +
                         self.imgheight * (self.dy + self.dx) / self.dz)

    def _setup_coronal(self, _slice):
        if not self.slices.has_key('coronal'):
            self.slices['coronal'] = self._getslice(_slice)
        else:
            self.slices['coronal'].grid = _slice

        self.slices['coronal'].height = self.imgheight / self.figheight
        try:
            print self.slices['coronal'].height, self.slices['transversal'].height
        except:
            pass
        
        self.slices['coronal'].xoffset = (2 * self.bufferlength + self.imgheight * self.dy / self.dz) / self.figwidth
        self.slices['coronal'].yoffset = (2 * self.bufferlength + self.imgheight * self.dx / self.dz) / self.figheight
        self.slices['coronal'].getaxes()

    def _z_changed(self):
        _slice = neuroimaging.reference.slices.transversal(z=self.z,
                                                           xlim=self.xlim,
                                                           ylim=self.ylim,
                                                           zlim=self.zlim,
                                                           shape=self.shape)

#        _slice = neuroimaging.reference.slices.transversal(z=self.z, transpose=True)

        self._setup_transversal(_slice)

    def _setup_transversal(self, _slice):
        if not self.slices.has_key('transversal'):
            self.slices['transversal'] = self._getslice(_slice)
        else:
            self.slices['transversal'].grid = _slice

        self.slices['transversal'].width = self.imgheight * self.dy / (self.dz * self.figwidth)
        self.slices['transversal'].height = self.imgheight * self.dx / (self.dz * self.figheight)
        self.slices['transversal'].xoffset = self.bufferlength / self.figwidth
        self.slices['transversal'].yoffset = self.bufferlength / self.figheight
        self.slices['transversal'].getaxes()

    def __init__(self, image, x=0, y=0, z=0,
                 xlim=[-90.,90.],
                 ylim=[-126.,90.],
                 zlim=[-72.,108.],
                 **keywords):
        self.slices = {}

        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim

        self._setup_dims()
        self.figure = pylab.figure(figsize=(self.figwidth / self.dpi, self.figheight / self.dpi))

        self.m = utils.reduceall(N.minimum, image.readall())
        self.M = utils.reduceall(N.maximum, image.readall())

        self.interpolator = neuroimaging.image.interpolation.ImageInterpolator(atlas)
        traits.HasTraits.__init__(self, x=x, y=y, z=z, **keywords)
        self.cid = pylab.connect('button_press_event', self.on_click)

    def on_click(self, event):

        print event.xdata, event.ydata, 'event'
        if event.inaxes == self.slices['sagittal'].axes:
            vy, vz = event.xdata, event.ydata
            vx = 0
            world = self.slices['sagittal'].grid.warp([vz,vy,vx])
            which = 'sagittal'
        elif event.inaxes == self.slices['transversal'].axes:
            vy, vx = event.xdata, event.ydata
            vz = 0
            world = self.slices['transversal'].grid.warp([vx,vy,vz])
            which = 'transversal'
        elif event.inaxes == self.slices['coronal'].axes:
            vx, vz = event.xdata, event.ydata
            vy = 0
            world = self.slices['coronal'].grid.warp([vz,vx,vy])
            which = 'coronal'

        self.z, self.y, self.x = world

        if which is 'coronal':
            self.slices['transversal'].draw(redraw=True)
            self.slices['sagittal'].draw(redraw=True)
        elif which is 'sagittal':
            self.slices['transversal'].draw(redraw=True)
            self.slices['coronal'].draw(redraw=True)
        elif which is 'transversal':
            self.slices['coronal'].draw(redraw=True)
            self.slices['sagittal'].draw(redraw=True)

        pylab.draw()

#        self.draw(redraw=True)


    def draw(self, redraw=False):
        pylab.figure(num=self.figure.number)
        for imslice in self.slices.values():
            imslice.draw(redraw=redraw)
        pylab.draw()
            
atlas = neuroimaging.image.Image('/home/jtaylo/.BrainSTAT/repository/kff.stanford.edu/~jtaylo/BrainSTAT/avg152T1.img')
v = BoxViewer(atlas)
import time
v.draw()

s = v.slices['sagittal'].axes
t = v.slices['transversal'].axes
c = v.slices['coronal'].axes


#pylab.savefig('out.png')
pylab.show()

toc = time.time()
v.x = 3.0
v.draw()
pylab.show()
tic = time.time()
print `tic-toc`
