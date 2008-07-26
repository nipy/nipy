"""
Module checks to see if particular modules are importable.
For now, it only checks if pylab and/or qt are available.
"""

__docformat__ = 'restructuredtext'

def enthought_traits_def():
    """
    Check to see if enthought.traits is importable.
    """
    try:
        import enthought.traits.api as traits
    except ImportError:
        try:
            import neuroimaging.externals.enthought.traits as traits
        except ImportError:
            ENTHOUGHT_TRAITS_DEF = False
            traits = None
        else:
            ENTHOUGHT_TRAITS_DEF = True
    else:
        ENTHOUGHT_TRAITS_DEF = True

    return ENTHOUGHT_TRAITS_DEF, traits
