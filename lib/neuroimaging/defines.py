"""
Module checks to see if particular modules are importable.
For now, it only checks if pylab and/or qt are available.
"""

__docformat__ = 'restructuredtext'

def pylab_def():
    """
    Check to see if pylab/matplotlib is importable.
    """
#    global PYLAB_DEF
#    global pylab
    try:
        import pylab
    except (ImportError, RuntimeError):
        PYLAB_DEF = False
        pylab = None
        pass
    else:
        PYLAB_DEF = True

    return PYLAB_DEF, pylab

#def qt_def():
#    """
#    Check to see if qt is importable.
#    """
#    global QT_DEF
#    global qt
#    try:
#        import qt
#        QT_DEF = True
#    except:
#        QT_DEF = False
#        qt = None
#        pass
#    return QT_DEF, qt

def enthought_traits_def():
    """
    Check to see if enthought.traits is importable.
    """
#    global ENTHOUGHT_TRAITS_DEF
#    global traits
    try:
        import enthought.traits.api as traits
    except ImportError:
        try:
            import neuroimaging.utils.enthought.traits as traits
        except ImportError:
            ENTHOUGHT_TRAITS_DEF = False
            traits = None
            pass
        else:
            ENTHOUGHT_TRAITS_DEF = True
    else:
        ENTHOUGHT_TRAITS_DEF = True

    return ENTHOUGHT_TRAITS_DEF, traits
