from enthought.traits.ui import View, Group, Item, Handler
import enthought.traits as traits

import run as FIACrun
import fiac
import numpy as N

class RunUI(traits.HasTraits):

    base = 'http://kff.stanford.edu/FIAC/'

    # Subject/run traits

    run = traits.Range(low=1, high=5, value=3)
    subj = traits.Range(low=0, high=15, value=3)
    mask = traits.true

    fmrifile = traits.Str
    maskfile = traits.Str
    
    # Confound traits

    normalize = traits.true
    mean_reg = traits.true
    drift_df = traits.Int
    knots = traits.Array(shape=(None,))
    tmax = traits.Float(500)
    tmin = traits.Float(0)
    blah = traits.Event

    def __init__(self, subj=3, run=3, drift_df=7, **keywords):
        traits.HasTraits.__init__(self, subj=subj, run=run, drift_df=drift_df,
                                  **keywords)

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
    ) 

    def _drift_df_changed(self):
        self.knots = N.linspace(self.tmin, self.tmax, self.drift_df - 4)

    def _subj_changed(self):
        self.fmrifile = fiac.FIACpath('fsl/filtered_func_data.img', subj=self.subj, run=self.run)

        try:
            test = urllib.urlopen(self.fmrifile)
        except:
            self.fmrifile = 'URL "%s" not found' % self.fmrifile

        self.maskfile = fiac.FIACpath('fsl/mask.img', subj=self.subj,
                                      run=self.run)
        try:
            test = urllib.urlopen(self.maskfile)
        except:
            self.maskfile = 'URL "%s" not found' % self.maskfile

    def _run_changed(self):
        self._subj_changed()

    def do_confound(self):
        self.handler.configure_traits(view='confound_view', context={'RunUI':self})
        
class MyHandler(Handler):

    data_group = Group(
            Item(
                object = 'RunUI',
                name   = 'subj',
                label  = 'Subject'
            ),
            Item(
                object = 'RunUI',
                name   = 'run',
                label  = 'run'
            ),
            Item(
                object = 'RunUI',
                name   = 'mask',
                label  = 'Use mask?'
            ),
            Item(
                name   = '_',
            ),
            Item(
                object = 'RunUI',
                name   = 'fmrifile',
                label  = 'Data:',
                style  = 'readonly'
            ),
            Item(
                object = 'RunUI',
                name   = 'maskfile',
                label  = 'Mask:',
                style  = 'readonly'
            ),
            label = 'Data')

    confound_group = Group(
            Item(
                object = 'RunUI',
                name   = 'drift_df',
                label  = 'Spline DF'
            ),
            Item(
                object = 'RunUI',
                name   = 'knots',
                label  = 'Spline knots',
                style  = 'readonly'
            ),
            Item( name = '_' ),
            Item(
                object = 'RunUI',
                name   = 'normalize',
                tooltip = 'Normalize frames to % bold?'
            ),
            Item(
                object = 'RunUI',
                name   = 'mean_reg',
                label  = 'Mean regressor',
                tooltip = 'Use frame averages as a regressor?'
            ),
            label = 'Confounds')

    traits_view = View(data_group, confound_group)

    data_view = View(data_group)
    confound_view = View(confound_group)


if __name__ == '__main__':
    a = RunUI()
    handler = MyHandler()

    a.handler = handler
    handler.configure_traits(context={'RunUI':a})
    print 'run', a.run

    handler.configure_traits(view='data_view', context={'RunUI':a})
    print 'run', a.run

    a.do_confound()
    print 'df', a.drift_df
