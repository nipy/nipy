#------------------------------------------------------------------------------
#
#  Copyright (c) 2005, Enthought, Inc.
#  All rights reserved.
# 
#  Written by: David C. Morrill
#
#  Date: 12/06/2005
# 
#------------------------------------------------------------------------------
""" Pseudo-package for all of the core symbols from Traits and TraitsUI.
"""
from neuroimaging.externals.enthought.traits.api \
    import *

try:    
    from neuroimaging.externals.enthought.traits.ui.api \
        import *
except:
    pass
