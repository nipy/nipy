#!/usr/bin/env python
import qt
from pylab import Figure, figaspect, gci, show, amax, amin, squeeze, asarray,\
    cm, angle, normalize, pi, arange, ravel, ones, outerproduct, floor,\
    fromfunction, zeros
from matplotlib.lines import Line2D
from matplotlib.image import AxesImage
from matplotlib.backends.backend_qtagg import \
  FigureCanvasQTAgg as FigureCanvas

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
class SliderTransform (object):
    def __init__(self, lower, stepsize):
        self.lower, self.stepsize = lower, stepsize
    def getValue(self, tick): return self.lower + self.stepsize*tick
    def getTick(self, value): return int((value - self.lower)/self.stepsize)


##############################################################################
class Slider (qt.QSlider):
    def __init__(self, parent, value, lower, upper, stepsize, pagesize, *args):
        qt.QSlider.__init__(self, parent, *args)
        self.xform = SliderTransform(lower, stepsize)
        self.setValue(value)
        self.connect(self, qt.SIGNAL("valueChanged(int)"), self.valueChanged)
    def getValue(self): 
        return self.xform.getValue(self.sliderPosition)
    def setValue(self, value):
        self.sliderPosition = self.xform.getTick(value)
    def valueChanged(self, tick):
        print "Slider.valueChanged called"
        self.emit(qt.PYSIGNAL("valueChanged"), (self,))


##############################################################################
class DimSpinner (qt.QSpinBox):
    def __init__(self, parent, name, value, start, end, handler, *args):
        qt.QSpinBox.__init__(self, *args)
        #adj = gtk.Adjustment(0, start, end, 1, 1)
        #adj.name = name
        #gtk.SpinButton.__init__(self, adj, 0, 0)
        #adj.connect("value-changed", handler)


##############################################################################
class DimSlider (Slider):
    def __init__(self, parent, dim, *args):
        Slider.__init__(self, parent, 0, 0, dim.size-1, 1, 8, *args)
        self.dim = dim
        #adj = gtk.Adjustment(0, 0, dim.size-1, 1, 1)
        #adj.dim = dim
        #gtk.HScale.__init__(self, adj)
        #self.set_digits(0)
        #self.set_value_pos(gtk.POS_RIGHT)


##############################################################################
class ContrastSlider (Slider):
    def __init__(self, parent, *args):
        Slider.__init__(self, parent, 1.0, 0.05, 2.0, 0.05, 0.4, *args)
        #gtk.HScale.__init__(self, gtk.Adjustment(1.0, 0.05, 2.0, 0.05, 1))
        #self.set_digits(2)
        #self.set_value_pos(gtk.POS_RIGHT)


##############################################################################
class ControlPanel (qt.QWidget):

    #-------------------------------------------------------------------------
    def __init__(self, parent, shape, dim_names=[], iscomplex=False, *args):
        qt.QWidget.__init__(self, parent, *args)
        self._init_dimensions(shape, dim_names)
        #gtk.Frame.__init__(self)
        #main_vbox = gtk.VBox()
        #main_vbox.set_border_width(2)

        # spinner for row dimension
        #spinner_box = gtk.HBox()
        self.row_spinner = DimSpinner(
            self, "row", len(shape)-2, 0, len(shape)-2, self.spinnerHandler, self)
        #spinner_box.add(gtk.Label("Row:"))
        #spinner_box.add(self.row_spinner)

        # spinner for column dimension
        self.col_spinner = DimSpinner(
            self, "col", len(shape)-1, 1, len(shape)-1, self.spinnerHandler, self)
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
        #    button = prev_button = qt.QRadioButton(prev_button, name)
        #    button.transform = xform_map[name]
        #    if name=="abs": button.set_active(True)
        #    self.radios.append(button)
        #    radio_box.add(button)
        #if iscomplex:
        #    main_vbox.pack_end(radio_box, False, False, 0)
        #    main_vbox.pack_end(gtk.HSeparator(), False, False, 0)

        # slider for each data dimension
        self.sliders = [DimSlider(self, dim) for dim in self.dimensions]
        #for slider, dim in zip(self.sliders, self.dimensions):
        #    self._add_slider(slider, "%s:"%dim.name, main_vbox)

        # start with the center row and column
        rowdim = self.getRowDim()
        #self.sliders[rowdim.index].set_value(rowdim.size/2)
        coldim = self.getColDim()
        #self.sliders[coldim.index].set_value(coldim.size/2)

        # slider for contrast adjustment
        self.contrast_slider = ContrastSlider(self)
        #self._add_slider(self.contrast_slider, "Contrast:", main_vbox)

        #self.add(main_vbox)

    #-------------------------------------------------------------------------
    def _add_slider(self, slider, label, vbox):pass
        #label = gtk.Label(label)
        #label.set_alignment(0, 0.5)
        #vbox.pack_start(label, False, False, 0)
        #vbox.pack_start(slider, False, False, 0)

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
        self.row_spinner.get_adjustment().connect(
          "value-changed", spinner_handler)
        self.col_spinner.get_adjustment().connect(
          "value-changed", spinner_handler)

        # connect radio buttons
        for r in self.radios: r.connect("toggled", radio_handler, r.transform)

        # connect slice position sliders
        for s in self.sliders:
            s.get_adjustment().connect("value_changed", slider_handler)

        # connect contrast slider
        self.contrast_slider.get_adjustment().connect(
          "value_changed", contrast_handler)

    #-------------------------------------------------------------------------
    def getContrastLevel(self):
        return self.contrast_slider.getValue()

    #-------------------------------------------------------------------------
    def getDimPosition(self, dnum):
        return int(self.sliders[dnum].getValue())

    #-------------------------------------------------------------------------
    def setDimPosition(self, dnum, index):
        return self.sliders[dnum].setValue(int(index))

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
class RowPlot (FigureCanvas):

    #-------------------------------------------------------------------------
    def __init__(self, parent, data):
        self.parent = parent
        fig = Figure(figsize=(3., 6.))
        ax  = fig.add_axes([0.05, 0.05, 0.85, 0.85])
        ax.xaxis.tick_top()
        ax.yaxis.tick_right()
        FigureCanvas.__init__(self, fig)
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
class ColPlot (FigureCanvas):

    #-------------------------------------------------------------------------
    def __init__(self, parent, data):
        self.parent = parent
        fig = Figure(figsize=(6., 3.))
        fig.add_axes([0.1, 0.1, 0.85, 0.85])
        FigureCanvas.__init__(self, fig)
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
class SlicePlot (FigureCanvas):

    #-------------------------------------------------------------------------
    def __init__(self, parent, data, x, y, cmap=cm.bone, norm=None):
        self.parent = parent
        self.norm = None
        fig = Figure(figsize=figaspect(data))
        ax  = fig.add_axes([0.05, 0.1, 0.85, 0.85])
        ax.yaxis.tick_right()
        ax.title.set_y(1.05) 
        FigureCanvas.__init__(self, fig)
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
class ColorBar (FigureCanvas):

    #-------------------------------------------------------------------------
    def __init__(self, parent, range, cmap=cm.bone, norm=None):
        self.parent = parent
        fig = Figure(figsize = (5,0.5))
        fig.add_axes((0.05, 0.55, 0.9, 0.3))
        FigureCanvas.__init__(self, fig)
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
class StatusBar (qt.QWidget):

    #-------------------------------------------------------------------------
    def __init__(self, parent, range, cmap, *args):
        qt.QWidget.__init__(self, parent, *args)
        #main_hbox = gtk.HBox()
        #main_hbox.set_border_width(0)

        # neighborhood size selection (eg '5x5', '3x4')
        # these numbers refer to "radii", not box size
        #self.entry = gtk.Entry(3)
        #self.entry.set_size_request(40,25)

        # pixel value
        #self.px_stat = gtk.Statusbar()
        #self.px_stat.set_has_resize_grip(False)
        #self.px_stat.set_size_request(160,25)

        # neighborhood avg
        #self.av_stat = gtk.Statusbar()
        #self.av_stat.set_has_resize_grip(False)
        #self.av_stat.set_size_request(160,25)

        # try to label entry box
        #label = gtk.Label("Radius")
        #label.set_alignment(0, 0.2)
        #label.set_size_request(10,25)
        #label.set_line_wrap(True)

        # colorbar
        self.cbar = ColorBar(self, range, cmap=cmap)
        #self.cbar.set_size_request(400,20)
        #main_hbox.add(self.cbar)
 
        # pixel value
        #self.label = gtk.Label()
        #self.label.set_alignment(2, 0.5)
        #self.label.set_size_request(140,20)
        #self.label.set_line_wrap(True)
        #main_hbox.add(self.label)
       
        #self.px_context = self.px_stat.get_context_id("Pixel Value")
        #self.av_context = self.av_stat.get_context_id("Neighborhood Avg")
        # default area to average
        #self.entry.set_text('3x3')
        #self.add(main_hbox)
        #self.show_all()

    #-------------------------------------------------------------------------
    def report(self, event, data):
        if not (event.xdata and event.ydata):
            avbuf = pxbuf = "  clicked outside axes"
        else:
            y, x = int(event.ydata), int(event.xdata)
            pxbuf = "  pix val: %f"%data[y, x]
            avbuf = "  %ix%i avg: %s"%self.squareAvg(y, x, data)
        
        self.pop_items()
        self.push_items(pxbuf, avbuf)

    #-------------------------------------------------------------------------
    def squareAvg(self, y, x, data):
        areaStr = self.getText()
        #box is defined +/-yLim rows and +/-xLim cols
        #if for some reason areaStr was entered wrong, default to (1, 1)
        yLim, xLim = len(areaStr)==3 and\
                     (int(areaStr[0]), int(areaStr[2])) or (1, 1)
        if y < yLim or x < xLim or\
           y+yLim >= data.shape[0] or\
           x+xLim >= data.shape[1]:
            return (yLim, xLim, "outOfRange")

        indices = fromfunction(lambda yi,xi: y+yi-yLim + 1.0j*(x + xi-xLim),
                               (yLim*2+1, xLim*2+1))
        scale = indices.shape[0]*indices.shape[1]
        av = sum(map(lambda zi: data[int(zi.real), int(zi.imag)]/scale,
                     indices.flat))
        
        #return box dimensions and 7 significant digits of average
        return (yLim, xLim, str(av)[0:8])

    #-------------------------------------------------------------------------
    def getText(self): return self.entry.get_text()

    #-------------------------------------------------------------------------
    def setLabel(self, text):
        self.label.set_text(text)

    #-------------------------------------------------------------------------    
    def pop_items(self):
        self.av_stat.pop(self.av_context)
        self.px_stat.pop(self.px_context)

    #-------------------------------------------------------------------------
    def push_items(self, pxbuf, avbuf):
        self.av_stat.push(self.av_context, avbuf)
        self.px_stat.push(self.px_context, pxbuf)


##############################################################################
#class arrayview (gtk.Window):
class ArrayView (qt.QMainWindow):
    #mag_norm = normalize()
    #phs_norm = normalize(-pi, pi)
    _mouse_x = _mouse_y = None
    _dragging = False

    #-------------------------------------------------------------------------
    def __init__(self, data, dim_names=[], title="sliceview", cmap=cm.bone):
        qt.QMainWindow.__init__(self)
        self.data = asarray(data)

        # if data is complex, show the magnitude by default
        self.transform = iscomplex(data) and abs_xform or ident_xform

        # widget layout table
        #table = gtk.Table(3, 2)
        table = qt.QGridLayout()
        #self.setLayout(table)

        # control panel
        self.control_panel = \
          ControlPanel(self, data.shape, dim_names, iscomplex(data))
        #self.control_panel.connect(
        #    self.spinnerHandler,
        #    self.radioHandler,
        #    self.sliderHandler,
        #    self.contrastHandler)
        #self.control_panel.set_size_request(200, 200)
        #table.attach(self.control_panel, 0, 1, 0, 1)

        # row plot
        self.rowplot = RowPlot(self, self.getRow())
        #self.rowplot.set_size_request(400, 200)
        #table.attach(self.rowplot, 1, 2, 0, 1)

        # column plot
        self.colplot = ColPlot(self, self.getCol())
        #self.colplot.set_size_request(200, 400)
        #table.attach(self.colplot, 0, 1, 1, 2)
        
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
        #self.sliceplot.set_size_request(400, 400)
        self.sliceplot.mpl_connect(
          'motion_notify_event', self.sliceMouseMotionHandler)
        self.sliceplot.mpl_connect(
          'button_press_event', self.sliceMouseDownHandler)
        self.sliceplot.mpl_connect(
          'button_release_event', self.sliceMouseUpHandler)
        #table.attach(self.sliceplot, 1, 2, 1, 2)
        #table.addWidget(self.sliceplot)

        # status
        self.status = StatusBar(self, self.sliceDataRange(), cmap)
        #self.status.set_size_request(200,30)
        #table.attach(self.status, 0, 2, 2, 3)

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
        self.status.cbar.setRange(self.sliceDataRange(), norm=self.norm)

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
    def sliderHandler(self, adj):
        row_dim, col_dim= self.control_panel.slice_dims
        if adj.dim.index == row_dim: self.updateRow()
        elif adj.dim.index == col_dim: self.updateCol()
        else: self.updateSlice()

    #-------------------------------------------------------------------------
    def contrastHandler(self, adj):
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
    main_win = ArrayView(data)
    qt.qApp.setMainWidget(main_win)
    main_win.show()
    qt.qApp.exec_loop()


##############################################################################
if __name__ == "__main__":
    from pylab import randn
    arrayview(randn(6,6))
