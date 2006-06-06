from enthought.traits.ui import View, Group, Item
import enthought.traits as traits

import run as FIACrun
import fiac

class RunUI(traits.HasTraits):

    base = 'http://kff.stanford.edu/FIAC/'

    # Subject/run traits

    run = traits.Range(low=1, high=5, value=3)
    run = traits.Range(low=0, high=15, value=3)
    mask = traits.true

    fmrifile = traits.Str
    maskfile = traits.Str
    
    # Confound traits

    normalize = traits.true
    drift_df = traits.Int
    knots = traits.Array(shape=(None,))
    tmax = traits.Float(500)
    tmin = traits.Float(0)

    def __init__(self, subj=3, run=3, drift_df=7, **keywords):
        traits.HasTraits.__init__(self, **keywords)
        
    def _drift_df_changed(self):
        self.knots = N.linspace(self.tmin, self.tmax, self.drift_df - 4)

    def _subj_changed(self):
        self.fmrifile = fiac.FIACpath('fsl/filtered_func_data.img', subj=subj, run=run)

        try:
            test = urllib.urlopen(self.fmrifile)
        except:
            self.fmrifile = 'File not found'

        self.maskfile = fiac.FIACpath('fsl/mask.img', subj=self.subj,
                                      run=self.run)
        try:
            test = urllib.urlopen(self.maskfile)
        except:
            self.maskfile = 'File not found'

    def _run_changed(self):
        self._subj_changed()




