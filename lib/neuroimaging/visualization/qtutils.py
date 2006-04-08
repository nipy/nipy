"""
General helper classes for handling some QT boilerplate.
"""
from qt import *

##############################################################################
class LayoutWidgetMixin (object):
    """
    A QWidget that has a layout.  Child widgets may be added directly instead
    of referring to the layout.  Layout instantiation is handled internally.
    @param layout_class:  the class object of the desired layout.
    @param layout_args:  a tuple of positional parameters to be used when
      instantiating the layout class.
    """

    def __init__(self, layout_class, layout_args, widget_class, *args):
        widget_class.__init__(self, *args)
        self.layout = layout_class(self, *layout_args)

    def addWidget(self, widget, *args):
        if isinstance(self.layout, QGridLayout) and len(args)==4:
            self.layout.addMultiCellWidget(widget, *args)
        else:
            self.layout.addWidget(widget, *args)


##############################################################################
class HBox (QFrame, LayoutWidgetMixin):
    def __init__(self, *args):
        LayoutWidgetMixin.__init__(self, QHBoxLayout, (), QFrame, *args)


##############################################################################
class VBox (QFrame, LayoutWidgetMixin):
    def __init__(self, *args):
        LayoutWidgetMixin.__init__(self, QVBoxLayout, (), QFrame, *args)


##############################################################################
class RangeTransform (object):
    """
    Converts (from/to) a discrete range of floats (to/from) the integer tick
    numbers used by a QRangeControl.
    @param lower:  float lower bound.
    @param upper:  float upper bound.
    @param stepsize:  distance between float steps.

    >>> t=RangeTransform(-1,1,0.4)
    >>> values = [t.getValue(tick) for tick in range(0,6)]
    >>> values
    [-1.0, -0.59999999999999998, -0.19999999999999996, 0.20000000000000018, 0.60000000000000009, 1.0]
    >>> [t.getTick(val) for val in values]
    [0, 1, 2, 3, 4, 5]
    """
    def __init__(self, lower, upper, stepsize):
        self.lower = float(lower)
        self.upper = float(upper)
        self.stepsize = float(stepsize)

    def getValue(self, tick): return self.lower + self.stepsize*tick

    def getTick(self, value): return int((value - self.lower)/self.stepsize)

    def numTicks(self): return int((self.upper-self.lower)/self.stepsize)


##############################################################################
class RangeSlider (QSlider):
    """
    A QSlider that reports and receives its values in a discrete float range,
    rather than integer.  Uses a RangeTransform.
    @param parent:  parent QWidget.
    @param value:  initial value.
    @param lower:  float lower bound.
    @param upper:  float upper bound.
    @param stepsize:  distance between float steps.
    """
    def __init__(self, parent, value, lower, upper, stepsize, orientation, *args):
        self.transform = RangeTransform(lower, upper, stepsize)
        upper_tick = self.transform.numTicks()
        page_tick = int(upper_tick/6.)
        tick_value = self.transform.getTick(value)
        QSlider.__init__(self, 0, upper_tick, page_tick,
            tick_value, orientation, parent, *args)
        self.connect(self, SIGNAL("valueChanged(int)"), self.rangeValueChanged)
        self.connect(self, SIGNAL("sliderMoved(int)"), self.rangeValueChanged)

    def getRangeValue(self): 
        return self.transform.getValue(self.value())

    def setRangeValue(self, value, direct=False):
        if direct:
            self.directSetValue(self.transform.getTick(value))
        else:
            self.setValue(self.transform.getTick(value))

    def rangeValueChanged(self, tick):
        self.emit(PYSIGNAL("range-value-changed"), (self,))


#-----------------------------------------------------------------------------
if __name__ == "__main__":
    import doctest
    doctest.testmod()
