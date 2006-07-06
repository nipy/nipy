#------------------------------------------------------------------------------
# Copyright (c) 2005, Enthought, Inc.
# All rights reserved.
# 
# This software is provided without warranty under the terms of the BSD
# license included in enthought/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.neuroimaging.extra.enthought.com/licenses/BSD.txt
# Thanks for using Enthought open source!
# 
# Author: Enthought, Inc.
# Description: <Enthought util package component>
#------------------------------------------------------------------------------
"""
===========================
Utility Functions 
=========================== 

:Copyright: 2003 Enthought, Inc.


"""

# Prints the caller's stack info:
def called_from ( levels = 1, context = 1 ):
    print '***** Deprecated.  Please use neuroimaging.extra.enthought.debug.called_from'
    from inspect import stack
    stk = stack( context )
    frame, file_name, line_num, func_name, lines, index = stk[1]
    print "'%s' called from:" % func_name
    for frame_rec in stk[ 2: 2 + levels ]:
        frame, file_name, line_num, func_name, lines, index = frame_rec
        print '   %s (%s: %d)' % ( func_name, file_name, line_num )
        if lines is not None:
            if len( lines ) == 1:
                print '      ' + lines[0].strip()[:73]
            else:
                for i, line in enumerate( lines ):
                    print '   %s  %s' % ( '|>'[ i == index ], line.rstrip() )

# command line version
def test(level=1,verbosity=1):
    import unittest
    runner = unittest.TextTestRunner(verbosity=verbosity)
    runner.run(test_suite(level))
    return runner
    
    
# returns a test suite for use elsewhere 
def test_suite(level=1):
    import neuroimaging.extra.enthought.util
    import neuroimaging.extra.enthought.util.testingx as testing
    return testing.harvest_test_suites(neuroimaging.extra.enthought.util,level=level)
