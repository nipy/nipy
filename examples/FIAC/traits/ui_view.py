from enthought.traits.ui import View, Group, Item
import enthought.traits as traits

import sys, os
sys.path.append(os.path.abspath('..'))

import run as FIACrun
import fiac
import numpy as N

from ui import RunUI

class RunUIView(RunUI):

    '''
    This class just has a "better" default traits view.
    '''

    traits_view = View(
        Group(
            Item(
                name   = 'subj',
                label  = 'Subject'
            ),
            Item(
                name   = 'run',
                label  = 'run'
            ),
            Item(
                name   = 'mask',
                label  = 'Use mask?'
            ),
            Item(
                name   = '_',
            ),
            Item(
                name   = 'fmrifile',
                label  = 'Data:',
                style  = 'readonly'
            ),
            Item(
                name   = 'maskfile',
                label  = 'Mask:',
                style  = 'readonly'
            ),
            label = 'Data'
        ),
        Group(
            Item(
                name   = 'drift_df',
                label  = 'Spline DF'
            ),
            Item(
                name   = 'knots',
                label  = 'Spline knots',
                style  = 'readonly'
            ),
            Item( name = '_' ),
            Item(
                name   = 'normalize',
                tooltip = 'Normalize frames to % bold?'
            ),
            Item(
                name   = 'norm_reg',
                label  = 'Mean regressor',
                tooltip = 'Use frame averages as a regressor?'
            ),
            label = 'Confounds'
        )
    ) 


if __name__ == '__main__':
    a = RunUIView()
    print a.fmrifile, 'fmri'
    a.configure_traits(kind='live')
    print a.subj, a.fmrifile
