from enthought.traits.ui import View, Group, Item
import enthought.traits as traits

import sys, os, urllib2
sys.path.append(os.path.abspath('..'))

import run as FIACrun
import fiac

base = 'http://kff.stanford.edu/FIAC/'

class RunUI(traits.HasTraits):

    # Subject/run traits

    subj = traits.Range(low=0, high=15, value=3) # Which of the 16 FIAC subjects?
    run = traits.Range(low=1, high=5, value=3) # Which run?

    mask = traits.true # Use a mask for the analysis?

    fmrifile = traits.Str # Location of fMRI data -- it is a trait because
                          # we want to display it in the view
    maskfile = traits.Str # Same as above
    
    # Confound traits

    drift_df = traits.Int # How many degrees of freedom should be used in the
                          # natural spline drift confound?

    knots = traits.Array(shape=(None,)) # Knots for the natural splines

    tmax = traits.Float(500) # upper and lower limits for knots
    tmin = traits.Float(0)

    normalize = traits.true # Perform frame by frame normalization? i.e.
                            # divide each frame by its (possibly masked)
                            # mean?
    norm_reg = traits.true  # Use the (possibly masked) frame means as a regressor in the design?

    def __init__(self, **keywords):
        traits.HasTraits.__init__(self, **keywords)
        self._subj_changed()
        self._drift_df_changed()
        
    def _drift_df_changed(self):
        self.knots = N.linspace(self.tmin, self.tmax, self.drift_df - 2)[1:-1]

    def _subj_changed(self):
        self.maskfile = fiac.FIACpath('fsl/mask.img', subj=self.subj, run=self.run, base=base)
        if not self.validate():
            self.fmrifile = 'URL "%s" not found' % self.fmrifile
            self.maskfile = 'URL "%s" not found' % self.maskfile

    def validate(self):
        self.fmrifile = fiac.FIACpath('fsl/filtered_func_data.img', subj=self.subj, run=self.run, base=base)
        try:
            test = urllib2.urlopen(self.fmrifile)
            return True
        except:
            return False

    def _run_changed(self):
        self._subj_changed()



