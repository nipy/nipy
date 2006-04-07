#!/usr/bin/env python
from qt import *
from pylab import Figure, figaspect, gci, show, amax, amin, squeeze, asarray,\
    cm, angle, normalize, pi, arange, ravel, ones, outerproduct, floor,\
    fromfunction, zeros
from matplotlib.lines import Line2D
from matplotlib.image import AxesImage
from matplotlib.backends.backend_qtagg import \
  FigureCanvasQTAgg as FigureCanvas

from qtutils import LayoutWidgetMixin, RangeSlider, HBox

def iscomplex(a): return hasattr(a, "imag")

# Transforms for viewing different aspects of complex data
def ident_xform(data): return data
def abs_xform(data): return abs(data)
def phs_xform(data): return angle(data)
def real_xform(data): return data.real
def imag_xform(data): return data.imag


##############################################################################
class Dimension (object):
    def __init__(self, index, size, name):
        self.index = index
        self.size = size
        self.name = name


##############################################################################
class ViewerCanvas (FigureCanvas):
    """
    Handles common logic needed to get an mpl FigureCanvas to play nicely in
    a QT environment.
    """
    def __init__(self, parent, fig):
        FigureCanvas.__init__(self, fig)
        self.reparent(parent, QPoint(0,0))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
 

##############################################################################
class DimSpinner (QSpinBox):
    def __init__(self, parent, name, value, start, end, handler, *args):
        QSpinBox.__init__(self, *args)
        self.name = name
        #adj.connect("value-changed", handler)


##############################################################################
class DimSlider (RangeSlider):
    def __init__(self, parent, dim, *args):
        RangeSlider.__init__(self, parent, 0, 0, dim.size-1, 1,
            RangeSlider.Horizontal, *args)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.dim = dim


##############################################################################
class ContrastSlider (RangeSlider):
    def __init__(self, parent, *args):
        RangeSlider.__init__(self, parent, 1.0, 0.05, 2.0, 0.05,
            RangeSlider.Horizontal, *args)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)


##############################################################################
class ControlPanel (QGroupBox, LayoutWidgetMixin):

    #-------------------------------------------------------------------------
    def __init__(self, parent, shape, dim_names=[], iscomplex=False, *args):
        LayoutWidgetMixin.__init__(self, QVBoxLayout, (10,), QGroupBox, parent, *args)
        self._init_dimensions(shape, dim_names)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMargin(50)

        # spinner for row dimension
        #spinner_box = gtk.HBox()
        #self.row_spinner = DimSpinner(
        #    self, "row", len(shape)-2, 0, len(shape)-2, self.spinnerHandler, self)
        #spinner_box.add(gtk.Label("Row:"))
        #spinner_box.add(self.row_spinner)

        # spinner for column dimension
        #self.col_spinner = DimSpinner(
        #    self, "col", len(shape)-1, 1, len(shape)-1, self.spinnerHandler, self)
        #spinner_box.add(gtk.Label("Col:"))
        #spinner_box.add(self.col_spinner)
        #main_vbox.add(spinner_box)

        # radio buttons for different aspects of complex data
        xform_map = {
          "ident": ident_xform,
          "abs": abs_xform,
          "phs": phs_xform,
          "real": real_xform,
          "imag": imag_xform}
        self.radios = []
        #radio_box = gtk.HBox()
        #prev_button = None
        #for name in ("abs","phs","real","imag"):
        #    button = prev_button = QRadioButton(prev_button, name)
        #    button.transform = xform_map[name]
        #    if name=="abs": button.set_active(True)
        #    self.radios.append(button)
        #    radio_box.add(button)
        #if iscomplex:
        #    main_vbox.pack_end(radio_box, False, False, 0)
        #    main_vbox.pack_end(gtk.HSeparator(), False, False, 0)

        # slider for each data dimension
        self.sliders = [DimSlider(None, dim) for dim in self.dimensions]
        for slider, dim in zip(self.sliders, self.dimensions):
            self._add_slider(slider, "%s:"%dim.name)

        # start with the center row and column
        rowdim = self.getRowDim()
        self.sliders[rowdim.index].setValue(rowdim.size/2)
        coldim = self.getColDim()
        self.sliders[coldim.index].setValue(coldim.size/2)

        self.layout.addStretch()

        # slider for contrast adjustment
        self.contrast_slider = ContrastSlider(self)
        self._add_slider(self.contrast_slider, "Contrast:")

    #-------------------------------------------------------------------------
    def _add_slider(self, slider, label):
        box = HBox(self)
        box.setMargin(0)
        slider.reparent(box, QPoint(0,0))
        box.addWidget(slider)
        readout = slider.makeReadout(box)
        box.addWidget(readout)
        self.addWidget(QLabel(label, self))
        self.addWidget(box)

    #-------------------------------------------------------------------------
    def _init_dimensions(self, dim_sizes, dim_names):
        self.dimensions = []
        num_dims = len(dim_sizes)
        num_names = len(dim_names)
        if num_names != num_dims:
            dim_names = ["Dim %s"%i for i in range(num_dims)]
        for dim_num, (dim_size, dim_name) in\
          enumerate(zip(dim_sizes, dim_names)):
            self.dimensions.append( Dimension(dim_num, dim_size, dim_name) )
        self.slice_dims = (self.dimensions[-2].index, self.dimensions[-1].index)

    #-------------------------------------------------------------------------
    def connect(self,
        spinner_handler, radio_handler, slider_handler, contrast_handler):
        "Connect control elements to the given handler functions."

        # connect slice orientation spinners
        #self.row_spinner.get_adjustment().connect(
        #  "value-changed", spinner_handler)
        #self.col_spinner.get_adjustment().connect(
        #  "value-changed", spinner_handler)

        # connect radio buttons
        #for r in self.radios: r.connect("toggled", radio_handler, r.transform)

        # connect slice position sliders
        for s in self.sliders:
            s.connect(s, PYSIGNAL("range-value-changed"), slider_handler)

        # connect contrast slider
        self.contrast_slider.connect(self.contrast_slider,
          PYSIGNAL("range-value-changed"), contrast_handler)

    #-------------------------------------------------------------------------
    def getContrastLevel(self):
        return self.contrast_slider.getRangeValue()

    #-------------------------------------------------------------------------
    def getDimPosition(self, dnum):
        return int(self.sliders[dnum].getRangeValue())

    #-------------------------------------------------------------------------
    def setDimPosition(self, dnum, index):
        return self.sliders[dnum].setRangeValue(int(index))

    #-------------------------------------------------------------------------
    def getRowIndex(self): return self.getDimPosition(self.slice_dims[0])

    #-------------------------------------------------------------------------
    def getColIndex(self): return self.getDimPosition(self.slice_dims[1])

    #------------------------------------------------------------------------- 
    def setRowIndex(self, index): self.setDimPosition(self.slice_dims[0],index)

    #------------------------------------------------------------------------- 
    def setColIndex(self, index): self.setDimPosition(self.slice_dims[1],index)

    #------------------------------------------------------------------------- 
    def getRowDim(self): return self.dimensions[self.slice_dims[0]]

    #------------------------------------------------------------------------- 
    def getColDim(self): return self.dimensions[self.slice_dims[1]]

    #-------------------------------------------------------------------------
    def getIndexSlices(self):
        return tuple([
          dim.index in self.slice_dims and\
            slice(0, dim.size) or\
            self.getDimPosition(dim.index)
          for dim in self.dimensions])

    #-------------------------------------------------------------------------
    def spinnerHandler(self, adj):
        newval = int(adj.value)
        row_adj = self.row_spinner.get_adjustment()
        col_adj = self.col_spinner.get_adjustment()

        if adj.name == "row" and newval >= int(col_adj.value):
            col_adj.set_value(newval+1)
        if adj.name == "col" and newval <= int(row_adj.value):
            row_adj.set_value(newval-1)

        self.slice_dims = (int(row_adj.value), int(col_adj.value))


##############################################################################
class RowPlot (ViewerCanvas):

    #-------------------------------------------------------------------------
    def __init__(self, parent, data):
        fig = Figure(figsize=(3., 6.))
        ax  = fig.add_axes([0.05, 0.05, 0.85, 0.85])
        ax.xaxis.tick_top()
        ax.yaxis.tick_right()
        ViewerCanvas.__init__(self, parent, fig)
        self.setData(data)

    #-------------------------------------------------------------------------
    def setDataRange(self, data_min, data_max):
        self.figure.axes[0].set_ylim(data_min, data_max)

    #-------------------------------------------------------------------------
    def setData(self, data):
        ax = self.figure.axes[0]
        indices = range(len(data))
        if not hasattr(self, "data"): ax.plot(indices, data)
        else: ax.lines[0].set_data(indices, data)
        ax.set_xlim(-.5, len(data)-.5)
        self.data = data
        self.draw()


##############################################################################
class ColPlot (ViewerCanvas):

    #-------------------------------------------------------------------------
    def __init__(self, parent, data):
        fig = Figure(figsize=(6., 3.))
        fig.add_axes([0.1, 0.1, 0.85, 0.85])
        ViewerCanvas.__init__(self, parent, fig)
        self.setData(data)

    #-------------------------------------------------------------------------
    def setDataRange(self, data_min, data_max):
        self.figure.axes[0].set_xlim(data_min, data_max)

    #-------------------------------------------------------------------------
    def setData(self, data):
        ax = self.figure.axes[0]
        indices = range(len(data))
        if not hasattr(self, "data"): ax.plot(data, indices)
        else: ax.lines[0].set_data(data, indices)
        ax.set_ylim(len(data)-.5, -.5)
        self.data = data
        self.draw()


##############################################################################
class SlicePlot (ViewerCanvas):

    #-------------------------------------------------------------------------
    def __init__(self, parent, data, x, y, cmap=cm.bone, norm=None):
        self.norm = None
        fig = Figure(figsize=figaspect(data))
        ax  = fig.add_axes([0.05, 0.1, 0.85, 0.85])
        ax.yaxis.tick_right()
        ax.title.set_y(1.05) 
        ViewerCanvas.__init__(self, parent, fig)
        self.cmap = cmap
        self.setData(data, norm=norm)
        self._init_crosshairs(x, y)

    #-------------------------------------------------------------------------
    def _init_crosshairs(self, x, y):
        row_data, col_data = self._crosshairs_data(x, y)
        row_line = Line2D(row_data[0], row_data[1], color="r", alpha=.5)
        col_line = Line2D(col_data[0], col_data[1], color="r", alpha=.5)
        self.crosshairs = (row_line, col_line)
        ax = self.getAxes()
        ax.add_artist(row_line)
        ax.add_artist(col_line)

    #-------------------------------------------------------------------------
    def _crosshairs_data(self, x, y):
        data_height, data_width = self.data.shape
        row_data = ((x+.5-data_width/4., x+.5+data_width/4.), (y+.5, y+.5))
        col_data = ((x+.5, x+.5), (y+.5-data_height/4., y+.5+data_height/4.))
        return row_data, col_data

    #-------------------------------------------------------------------------
    def getAxes(self): return self.figure.axes[0]

    #-------------------------------------------------------------------------
    def getImage(self):
        images = self.getAxes().images
        return len(images) > 0 and images[0] or None
        
    #-------------------------------------------------------------------------
    def setImage(self, image): self.getAxes().images[0] = image

    #-------------------------------------------------------------------------
    def setData(self, data, norm=None):
        ax = self.getAxes()

        if self.getImage() is None:
            ax.imshow(data, interpolation="nearest",
              cmap=self.cmap, norm=self.norm, origin="lower")
        elif norm != self.norm:
            self.setImage(AxesImage(ax, interpolation="nearest",
              cmap=self.cmap, norm=norm, origin="lower"))

        self.getImage().set_data(data)
        self.norm = norm
        nrows, ncols = data.shape[:2]
        ax.set_xlim((0,ncols))
        ax.set_ylim((nrows,0))
        self.data = data
        self.draw()

    #------------------------------------------------------------------------- 
    def setCrosshairs(self, x, y):
        row_data, col_data = self._crosshairs_data(x, y)
        row_line, col_line = self.crosshairs
        row_line.set_data(*row_data)
        col_line.set_data(*col_data)
        self.draw()

    #-------------------------------------------------------------------------
    def getEventCoords(self, event):
        if event.xdata is not None: x = int(event.xdata)
        else: x = None
        if event.ydata is not None:y = int(event.ydata)
        else: y = None
        if x < 0 or x >= self.data.shape[0]: x = None
        if y < 0 or y >= self.data.shape[1]: y = None
        return (y,x)


##############################################################################
class ColorBar (ViewerCanvas):

    #-------------------------------------------------------------------------
    def __init__(self, parent, range, cmap=cm.bone, norm=None):
        #fig = Figure(figsize = (5,0.5))
        fig = Figure(figsize = (100,5))
        fig.add_axes((0.05, 0.55, 0.9, 0.3))
        ViewerCanvas.__init__(self, parent, fig)
        self.figure.axes[0].yaxis.set_visible(False)
        self.cmap = cmap
        self.draw()
        self.setRange(range, norm=norm)

    #-------------------------------------------------------------------------
    def setRange(self, range, norm=None):
        self.norm = norm
        dMin, dMax = range
        ax = self.figure.axes[0]

        if dMin == dMax:
            r_pts = zeros((128,))
            tx = asarray([0])
        else:
            # make decently smooth gradient, try to include end-point
            delta = (dMax-dMin)/127
            r_pts = arange(dMin, dMax+delta, delta)
            # sometimes forcing the end-point breaks
            if len(r_pts) > 128: r_pts = arange(dMin, dMax, delta)

            # set up tick marks
            delta = (r_pts[-1] - r_pts[0])/7
            eps = 0.1 * delta
            tx = arange(r_pts[0], r_pts[-1], delta)
            # if the last tick point is very far away from the end,
            # add one more at the end
            if (r_pts[-1] - tx[-1]) > .75*delta:
                #there MUST be an easier way!
                a = tx.tolist()
                a.append(r_pts[-1])
                tx = asarray(a)
            # else if the last tick point is misleadingly close,
            # replace it with the true endpoint
            elif (r_pts[-1] - tx[-1]) > eps: tx[-1] = r_pts[-1]

        data = outerproduct(ones(5),r_pts)
        # need to clear axes because axis Intervals weren't updating
        ax.clear()
        ax.imshow(data, interpolation="nearest",
              cmap=self.cmap, norm=norm, extent=(r_pts[0], r_pts[-1], 0, 1))
        ax.images[0].set_data(data)
        ax.xaxis.set_ticks(tx)
        self.data = data
        self.draw()


##############################################################################
class StatusBar (QFrame, LayoutWidgetMixin):

    #-------------------------------------------------------------------------
    def __init__(self, parent, range, cmap, *args):
        LayoutWidgetMixin.__init__(self, QHBoxLayout, (), QFrame, parent, *args)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)

        # colorbar
        self.colorbar = ColorBar(self, range, cmap=cmap)
        self.colorbar.setFixedSize(400,20)
        self.addWidget(self.colorbar)
 
        # pixel value
        self.label = QLabel(self)
        #self.label.set_alignment(2, 0.5)
        self.label.setFixedSize(140,20)
        #self.label.set_line_wrap(True)
        self.addWidget(self.label)

    #-------------------------------------------------------------------------
    def setLabel(self, text):
        self.label.setText(text)


##############################################################################
class ArrayView (QWidget, LayoutWidgetMixin):
    #mag_norm = normalize()
    #phs_norm = normalize(-pi, pi)
    _mouse_x = _mouse_y = None
    _dragging = False

    #-------------------------------------------------------------------------
    def __init__(self, data, dim_names=[], title="sliceview", cmap=cm.bone):
        LayoutWidgetMixin.__init__(self, QGridLayout, (3,2,5), QWidget)
        self.setCaption("Array Viewer")
        self.layout.setRowStretch(1,1)
        self.layout.setColStretch(1,1)
        self.data = asarray(data)

        # if data is complex, show the magnitude by default
        self.transform = iscomplex(data) and abs_xform or ident_xform

        # control panel
        self.control_panel = \
          ControlPanel(self, data.shape, dim_names, iscomplex(data))
        self.control_panel.connect(
            self.spinnerHandler,
            self.radioHandler,
            self.sliderHandler,
            self.contrastHandler)
        #self.control_panel.setMinimumSize(200, 200)
        self.addWidget(self.control_panel, 0, 0)

        # row plot
        self.rowplot = RowPlot(self, self.getRow())
        self.rowplot.setMinimumSize(400, 200)
        self.addWidget(self.rowplot, 0, 1)

        # column plot
        self.colplot = ColPlot(self, self.getCol())
        self.colplot.setMinimumSize(200, 400)
        self.addWidget(self.colplot, 1, 0)
        
        # Set up normalization BEFORE plotting images.
        # Contrast level of 1.0 gives default normalization (changed by
        # contrast slider).
        self.conLevel = 1.0
        self.norm = None
        self.setNorm()

        # slice image
        self.sliceplot = SlicePlot(self, self.getSlice(),
          self.control_panel.getRowIndex(),
          self.control_panel.getColIndex(),
          cmap=cmap, norm=self.norm)
        self.sliceplot.setMinimumSize(400, 400)
        #self.sliceplot.set_size_request(400, 400)
        self.sliceplot.mpl_connect(
          'motion_notify_event', self.sliceMouseMotionHandler)
        self.sliceplot.mpl_connect(
          'button_press_event', self.sliceMouseDownHandler)
        self.sliceplot.mpl_connect(
          'button_release_event', self.sliceMouseUpHandler)
        self.addWidget(self.sliceplot, 1, 1)

        # status
        self.status = StatusBar(self, self.sliceDataRange(), cmap)
        self.status.setMinimumSize(600,25)
        self.addWidget(self.status, 2, 2, 0, 1)

        self.updateDataRange()

        # main window
        #gtk.Window.__init__(self)
        #self.connect("destroy", lambda x: gtk.main_quit())
        #self.set_default_size(400,300)
        #self.set_title(title)
        #self.set_border_width(3)
        #self.add(table)
        #self.show_all()
        #show()

    #-------------------------------------------------------------------------
    def getRow(self):
        return self.getSlice()[self.control_panel.getRowIndex(),:]

    #-------------------------------------------------------------------------
    def getCol(self):
        return self.getSlice()[:,self.control_panel.getColIndex()]

    #-------------------------------------------------------------------------
    def getSlice(self):
        return self.transform(
          squeeze(self.data[self.control_panel.getIndexSlices()]))

    #-------------------------------------------------------------------------
    def updateRow(self):
        self.updateCrosshairs()
        self.rowplot.setData(self.getRow())

    #-------------------------------------------------------------------------
    def updateCol(self):
        self.updateCrosshairs()
        self.colplot.setData(self.getCol())

    #-------------------------------------------------------------------------
    def updateSlice(self):
        self.setNorm()
        self.sliceplot.setData(self.getSlice(), norm=self.norm)
        self.rowplot.setData(self.getRow())
        self.colplot.setData(self.getCol())
        self.status.colorbar.setRange(self.sliceDataRange(), norm=self.norm)

    #-------------------------------------------------------------------------
    def sliceDataRange(self):
        flatSlice = ravel(self.getSlice())
        return amin(flatSlice), amax(flatSlice)

    #------------------------------------------------------------------------- 
    def updateDataRange(self):
        flat_data = self.transform(self.data).flat
        data_min = amin(flat_data)
        data_max = amax(flat_data)
        self.rowplot.setDataRange(data_min, data_max)
        self.colplot.setDataRange(data_max, data_min)

    #-------------------------------------------------------------------------
    def spinnerHandler(self, adj):
        print "sliceview::spinnerHandler slice_dims", \
               self.control_panel.slice_dims

    #-------------------------------------------------------------------------
    def radioHandler(self, button, transform):
        if not button.get_active(): return
        self.transform = transform
        self.updateDataRange()
        self.updateSlice()

    #-------------------------------------------------------------------------
    def sliderHandler(self, slider):
        row_dim, col_dim= self.control_panel.slice_dims
        if slider.dim.index == row_dim: self.updateRow()
        elif slider.dim.index == col_dim: self.updateCol()
        else: self.updateSlice()

    #-------------------------------------------------------------------------
    def contrastHandler(self, slider):
        self.conLevel = self.control_panel.getContrastLevel()
        self.updateSlice()

    #-------------------------------------------------------------------------
    def sliceMouseDownHandler(self, event):
        y, x = self.sliceplot.getEventCoords(event)
        self._dragging = True
        # make sure this registers as a "new" position
        self._mouse_x = self._mouse_y = None
        self.updateCoords(y,x)

    #-------------------------------------------------------------------------
    def sliceMouseUpHandler(self, event):
        y, x = self.sliceplot.getEventCoords(event)
        self._dragging = False

    #-------------------------------------------------------------------------
    def sliceMouseMotionHandler(self, event):
        y, x = self.sliceplot.getEventCoords(event)
        self.updateCoords(y,x)

    #-------------------------------------------------------------------------
    def updateCoords(self, y, x):

        # do nothing if coords haven't changed
        if x == self._mouse_x and y == self._mouse_y: return
        self._mouse_x, self._mouse_y = x, y

        # update statusbar element value label
        self.updateStatusLabel(y, x)

        # update crosshairs and projection plots if button down
        if self._dragging: self.updateProjections(y,x)

    #------------------------------------------------------------------------- 
    def updateStatusLabel(self, y, x):
        if x != None and y != None:
            text = "[%d,%d] = %.4f"%(y, x, self.getSlice()[y,x])
        else: text = ""
        self.status.setLabel(text)

    #------------------------------------------------------------------------- 
    def updateProjections(self, y, x):
        "Update crosshairs and row and column plots."
        if x != None and y != None:
            self.control_panel.setRowIndex(y)
            self.control_panel.setColIndex(x)
            self.updateCrosshairs()

    #------------------------------------------------------------------------- 
    def updateCrosshairs(self):
        self.sliceplot.setCrosshairs(
          self.control_panel.getColIndex(),
          self.control_panel.getRowIndex())
        
    #------------------------------------------------------------------------- 
    def setNorm(self):
        scale = -0.75*(self.conLevel-1.0) + 1.0
        dMin, dMax = self.sliceDataRange()

        # only scale the minimum value if it is below zero (?)
        sdMin = dMin < 0 and dMin * scale or dMin

        # if the norm scalings haven't changed, don't change norm
        if self.norm and\
           (sdMin, dMin*scale) == (self.norm.vmin, self.norm.vmax): return

        # else set it to an appropriate scaling
        self.norm = self.transform == phs_xform and\
          normalize(-pi*scale, pi*scale) or normalize(sdMin, scale*dMax)
   

#----------------------------------------------------------------------------- 
def arrayview(data):
    viewer = ArrayView(data)
    viewer.show()
    qApp.setMainWidget(viewer)
    qApp.exec_loop()


##############################################################################
if __name__ == "__main__":
    from pylab import randn
    arrayview(randn(20,20))
