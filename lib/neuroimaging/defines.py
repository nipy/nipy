"""
This modules checks to see if particular modules are importable.
For now, it only checks if pylab and/or qt are available.
"""

def pylab_def():
    """
    Check to see if pylab/matplotlib is importable.
    """
    global PYLAB_DEF
    global pylab
    try:
        import pylab
        PYLAB_DEF = True
    except:
        PYLAB_DEF = False
        pylab = None
        pass
    return PYLAB_DEF, pylab

def qt_def():
    """
    Check to see if qt is importable.
    """
    global QT_DEF
    global qt
    try:
        import qt
        QT_DEF = True
    except:
        QT_DEF = False
        qt = None
        pass
    return QT_DEF, qt

def enthought_traits_def():
    """
    Check to see if enthought.traits is importable.
    """
    global ENTHOUGHT_TRAITS_DEF
    global traits
    try:
        import enthought.traits.api as traits
        ENTHOUGHT_TRAITS_DEF = True
    except:
        ENTHOUGHT_TRAITS_DEF = False
        try:
            import neuroimaging.utils.enthought.traits as traits
            ENTHOUGHT_TRAITS_DEF = True
        except Exception, e:
            traits = None
            print e
            pass
    return ENTHOUGHT_TRAITS_DEF, traits
