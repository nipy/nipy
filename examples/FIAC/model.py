import os, time, glob

from neuroimaging import traits
import numpy as np
from neuroimaging.fixes.scipy.stats_models import contrast

from neuroimaging.modalities.fmri import protocol, functions
from neuroimaging.modalities.fmri.fmristat import delay
from neuroimaging.modalities.fmri.filters import Filter
from neuroimaging.modalities.fmri.hrf import canonical
from neuroimaging.core.api import Image, create_outfile
import neuroimaging.modalities.fmri.fmristat.utils as fmristat

import fiac
import io

from readonly import ReadOnlyValidate, HasReadOnlyTraits

#-----------------------------------------------------------------------------#
#
# Canonical drift
#
#-----------------------------------------------------------------------------#

def drift_fn(time):
    """
    Drift function defined by fmristat
    """
    _t = (time - 1.25) / 2.5 - 191/2. # return to time index, centered at numframes/ 2
    v = np.asarray([np.ones(_t.shape[0]), _t, _t**2, _t**3, np.greater(_t, 0) * _t**3])
    for i in range(5):
        v[i] /= v[i].max()
    return v

canonical_drift = protocol.ExperimentalQuantitative('drift', drift_fn)

#-----------------------------------------------------------------------------#
#
# Delay HRF
#
#-----------------------------------------------------------------------------#

delay_hrf = delay.DelayHRF()

#-----------------------------------------------------------------------------#
#
# Generic Model
#
#-----------------------------------------------------------------------------#

class Model(HasReadOnlyTraits):

    drift = ReadOnlyValidate(traits.Instance(protocol.ExperimentalQuantitative), desc='Model for drift.')

    normalize = ReadOnlyValidate(traits.true,
                                 desc='Use frame averages to normalize to % BOLD?')

    
    normalize_reg = ReadOnlyValidate(traits.true,
                                     desc='Use frame averages as a regressor in model?')
                          
    frameavg = ReadOnlyValidate(traits.Instance(protocol.ExperimentalQuantitative), desc='Frame averages, used if image is normalized or as a regressor.')

    hrf = traits.Instance(Filter, desc='Hemodynamic response function.')

    formula = traits.Instance(protocol.Formula, desc='Model formula.')

    shift = traits.Float(1.25, desc='Global offset time from images recorded frametimes -- note that FSL has slice-timed data to the midpoint of the TR.')

    plot = ReadOnlyValidate(traits.Instance(MultiPlot),
                            desc='Rudimentary multi-line plotter for design')

    def __init__(self, drift=None, hrf=None, normalize=True,
                 normalize_reg=True, **keywords):
        self.normalize = normalize
        self.normalize_reg = normalize_reg
        self.drift = drift or canonical_drift
        self.hrf = hrf or delay_hrf
        HasReadOnlyTraits.__init__(self, **keywords)
        self.formula = protocol.Formula(self.drift)

    def __repr__(self):
        return '<FIAC drift model>'

    def view(self, time=np.linspace(0,191*2.5,3000)):
        """
        View a multiline display of the formula for a given model.
        """
        self.plot = MultiPlot(self.formula(time=time),
                              time=time,
                              title='Design for %s' % `self`)
        self.plot.draw()

#-----------------------------------------------------------------------------#
#
# Study level model
#
#-----------------------------------------------------------------------------#

class Study(Model, fiac.Study):

    def __init__(self, root=None, drift=None, hrf=None, **keywords):
        fiac.Study.__init__(self, root=root)
        Model.__init__(self, drift=drift, hrf=hrf, **keywords)

    def __repr__(self):
        return fiac.Study.__repr__(self)

local_study = Study(root=io.data_path)
www_study = Study(root='http://kff.stanford.edu/FIAC')

#-----------------------------------------------------------------------------#
#
# Subject level model 
#
#-----------------------------------------------------------------------------#

class Subject(Model, fiac.Subject):
    """
    Perhaps each subject has a different HRF -- this code
    can be put in a subclass of this class.
    """

    study = ReadOnlyValidate(traits.Instance(Model), desc='Study level model.')

    def __init__(self, id, drift=None, hrf=None,
                 study=local_study, **keywords):
        self.study = study
        fiac.Subject.__init__(self, id, study=study)
        Model.__init__(self, drift=drift, hrf=hrf, **keywords)
        self.formula = self.study.formula

    def __repr__(self):
        return fiac.Subject.__repr__(self)

#-----------------------------------------------------------------------------#
#
# Run level model 
#
#-----------------------------------------------------------------------------#

class Run(Model, fiac.Run):

    """
    Run specific model. Model formula is returned by formula method.
    """

    subject = ReadOnlyValidate(traits.Instance(Subject), desc='Subject level model.')

    resultdir = ReadOnlyValidate(traits.Str, desc='Directory where results are stored.')

    def __init__(self, subject, id, drift=None, hrf=None,
                 resultdir=os.path.join("fsl", "fmristat_run"),
                 **keywords):
        fiac.Run.__init__(self, subject, id=id)
        Model.__init__(self, drift=drift, hrf=hrf, **keywords)
        self.resultdir = self.joinpath(resultdir)
        self.subject = subject

        self.begin.convolve(canonical)
        self.experiment.convolve(self.hrf)
       
        self.formula = self.subject.formula + self.begin + self.experiment
        if self.normalize or self.normalize_reg:

            self.load()
            brainavg = fmristat.WholeBrainNormalize(self.fmri, mask=self.mask)
            if self.normalize_reg:
                brainavg_fn = \
                            functions.InterpolatedConfound(values=brainavg.avg,
                                                           times=self.shift +
                                                           self.fmri.frametimes)

                self.frameavg = protocol.ExperimentalQuantitative('frameavg',
                                                               brainavg_fn)
            if self.normalize:
                self.brainavg = brainavg
                
            self.clear()
            self.formula += self.frameavg

    def __repr__(self):
        return fiac.Run.__repr__(self)

    def result(self, which='contrasts', contrast='speaker', stat='t'):
        """
        Retrieve result of a specific which/contrast/stat
        """
        
        resultfile = os.path.join(self.resultdir, which, contrast,
                                  "%s.nii" % stat)
        return Image(resultfile)

    def cleanup(self):
        """
        Remove uncompressed .nii files from results directories.
        """
        for which in ['contrasts', 'delays']:
            for contrast in ['average', 'interaction', 'speaker', 'sentence']:
                resultpath = os.path.join(self.resultdir, which, contrast)
                [os.remove(imfile) for imfile in glob.glob(os.path.join(resultpath, "*nii"))]

    def _setup_contrasts(self):
        """
        This function is specific to the FIAC data, and defines
        the contrasts of interest.
        """

        p = self.experiment
        irf = self.hrf[0] 
        f = self.formula

        self.overallF = Contrast(p, f, name='overallF')

        SSt_SSp = p['SSt_SSp'].astimefn()
        DSt_SSp = p['DSt_SSp'].astimefn()
        SSt_DSp = p['SSt_DSp'].astimefn()
        DSt_DSp = p['DSt_DSp'].astimefn()

        # Average effect

        # important: average is NOT convolved with HRF even though p was!!!
        # same follows for other contrasts below
        self.average = (SSt_SSp + SSt_DSp + DSt_SSp + DSt_DSp) * 0.25        
        self.average = irf.convolve(self.average)
        self.average = Contrast(self.average, f, name='average')
        
        # Speaker effect

        self.speaker = (SSt_DSp - SSt_SSp + DSt_DSp - DSt_SSp) * 0.5
        self.speaker = irf.convolve(self.speaker)
        self.speaker = Contrast(self.speaker, f, name='speaker')
        
        # Sentence effect

        self.sentence = (DSt_SSp + DSt_DSp - SSt_SSp - SSt_DSp) * 0.5
        self.sentence = irf.convolve(self.sentence)
        self.sentence = Contrast(self.sentence, f, name='sentence')
        
        # Interaction effect

        self.interaction = SSt_SSp - SSt_DSp - DSt_SSp + DSt_DSp
        self.interaction = irf.convolve(self.interaction)
        self.interaction = Contrast(self.interaction, f, name='interaction')
        
        # delay -- this presumes
        # that the HRF used is a subclass of delay.DelayIRF
    
        # eventdict:{1:'SSt_SSp', 2:'SSt_DSp', 3:'DSt_SSp', 4:'DSt_DSp'}

        self.delays = fmristat.DelayContrast([SSt_SSp, SSt_DSp, DSt_SSp, DSt_DSp],
                                             [[-0.5,-0.5,0.5,0.5],
                                              [-0.5,0.5,-0.5,0.5],
                                              [1,-1,-1,1],
                                              [0.25,0.25,0.25,0.25]],
                                             f,
                                             name='task',
                                             rownames=['sentence',
                                                       'speaker',
                                                       'interaction',
                                                       'average'],
                                             IRF=self.hrf)

    def OLS(self, **OLSopts):
        """
        OLS pass through data.
        """
        
        self.load()
        if self.normalize:
            OLSopts['normalize'] = self.brainavg

        self.OLSmodel = fmristat.FmriStatOLS(self.fmri,
                                             formula=self.formula,
                                             mask=self.mask,
                                             tshift=self.shift, 
                                             path=self.resultdir,
                                             **OLSopts)

        self._setup_contrasts()
        self.OLSmodel.reference = self.average
        
        toc = time.time()
        self.OLSmodel.fit()
        tic = time.time()
        
        print 'OLS time', `tic-toc`
        
        rho = self.OLSmodel.rho_estimator.img
        rho.tofile("%s/rho.nii" % self.OLSmodel.path, clobber=True)
        
        self.clear()

    def AR(self, **ARopts):

        self.load()

        toc = time.time()
        self.ARmodel = fmristat.FmriStatAR(self.OLSmodel,
                                           contrasts=[self.overallF,
                                                      self.average,
                                                      self.speaker,
                                                      self.sentence,
                                                      self.interaction,
                                                      self.delays],
                                           tshift=self.shift,
                                           **ARopts)
        self.ARmodel.fit()
        tic = time.time()
        
        self.clear()
        print 'AR time', `tic-toc`

##         # if we output the AR whitened residuals, we might as
##         # well output the FWHM, too

##         if output_fwhm:
##             resid = neuroimaging.modalities.fmri.api.FmriImage(FIACpath('fsl/fmristat_run/ARresid.img', subj=subj, run=run))
##             fwhmest = fastFWHM(resid, fwhm=FIACpath('fsl/fmristat_run/fwhm.img'), clobber=True)
##             fwhmest()

##         del(OLS); del(AR); gc.collect()
##         return formula


#-----------------------------------------------------------------------------#
#
# Contrasts, with a "view" method
#
#-----------------------------------------------------------------------------#

class Contrast(contrast.Contrast, HasReadOnlyTraits):
    
    plot = ReadOnlyValidate(traits.Instance(MultiPlot),
                            desc='Rudimentary multi-line plotter for design')

    def view(self, time=np.linspace(0, 191*2.5, 3000)):
        self.plot = MultiPlot(self.term(time=time),
                              time=time,
                              title='Column space for %s' % `self`)
        self.plot.draw()

    def __repr__(self):
        return '<contrast: %s>' % self.name
        
def run(subj=3, run=3):
    """
    Run through a fit of FIAC data.
    """
    study = Study(root=io.data_path)
    subject = Subject(subj, study=study)
    runmodel = Run(subject, run)
    runmodel.OLS(clobber=True)
    runmodel.AR(clobber=True)

