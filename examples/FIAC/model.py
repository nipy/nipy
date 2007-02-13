import os, time

from neuroimaging import traits
import numpy as N
from scipy.sandbox.models import contrast

from neuroimaging.modalities.fmri.pca import MultiPlot
from neuroimaging.modalities.fmri import protocol, functions
from neuroimaging.modalities.fmri.fmristat import delay
from neuroimaging.modalities.fmri.filters import Filter
from neuroimaging.core.image.image import Image
import neuroimaging.modalities.fmri.fmristat.utils as fmristat

from fiac import Run, Subject, Study
import io

from readonly import ReadOnlyValidate, HasReadOnlyTraits

#-----------------------------------------------------------------------------#
#
# Canonical drift
#
#-----------------------------------------------------------------------------#

tmax = 190*2.5+1.25
tmin = 1.25
drift_fn = functions.SplineConfound(window=[tmin, tmax], df=5)
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

    formula = traits.Instance(protocol.formula, desc='Model formula.')

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
        self.formula = protocol.formula(self.drift)

    def __repr__(self):
        return '<FIAC drift model>'

    def view(self, time=N.linspace(0,191*2.5,3000)):
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

class StudyModel(Model, Study):

    def __init__(self, root=None, drift=None, hrf=None, **keywords):
        Study.__init__(self, root=root)
        Model.__init__(self, drift=drift, hrf=hrf, **keywords)

    def __repr__(self):
        return Study.__repr__(self)

local_study = StudyModel(root=io.data_path)
www_study = StudyModel(root='http://kff.stanford.edu/FIAC')

#-----------------------------------------------------------------------------#
#
# Subject level model 
#
#-----------------------------------------------------------------------------#

class SubjectModel(Model, Subject):
    """
    Perhaps each subject has a different HRF -- this code
    can be put in a subclass of this class.
    """

    study = ReadOnlyValidate(traits.Instance(StudyModel), desc='Study level model.')

    def __init__(self, id, drift=None, hrf=None,
                 study=local_study, **keywords):
        self.study = study
        Subject.__init__(self, id, study=study)
        Model.__init__(self, drift=drift, hrf=hrf, **keywords)
        self.formula = self.study.formula

    def __repr__(self):
        return Subject.__repr__(self)

#-----------------------------------------------------------------------------#
#
# Run level model 
#
#-----------------------------------------------------------------------------#

class RunModel(Model, Run):

    """
    Run specific model. Model formula is returned by formula method.
    """

    subject = ReadOnlyValidate(traits.Instance(SubjectModel), desc='Subject level model.')

    resultdir = ReadOnlyValidate(traits.Str, desc='Directory where results are stored.')

    def __init__(self, subject, id, drift=None, hrf=None,
                 resultdir=os.path.join("fsl", "fmristat_run"),
                 **keywords):
        Run.__init__(self, subject, id=id)
        Model.__init__(self, drift=drift, hrf=hrf, **keywords)
        self.resultdir = self.joinpath(resultdir)
        self.subject = subject

        self.begin.convolve(self.hrf[0])
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
        return Run.__repr__(self)


    def result(self, which='contrasts', contrast='speaker', stat='t'):
        resultfile = os.path.join(self.resultdir, which, contrast,
                                  "%s.img" % stat)
        return Image(resultfile)

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
        self.average = (SSt_SSp + DSt_SSp + SSt_DSp + DSt_DSp) * 0.25        
        self.average = irf.convolve(self.average)
        self.average = Contrast(self.average, f, name='average')
        
        # Sentence effect

        self.sentence = (DSt_SSp + DSt_DSp) * 0.5 - (SSt_SSp + SSt_DSp) * 0.5
        self.sentence = irf.convolve(self.sentence)
        self.sentence = Contrast(self.sentence, f, name='sentence')
        
        # Speaker effect

        self.speaker =  (SSt_DSp + DSt_DSp) / 2. - (SSt_SSp + DSt_SSp) / 2.
        self.speaker = irf.convolve(self.speaker)
        self.speaker = Contrast(self.speaker, f, name='speaker')
        
        # Interaction effect

        self.interaction = SSt_SSp - SSt_DSp - DSt_SSp + DSt_DSp
        self.interaction = irf.convolve(self.interaction)
        self.interaction = Contrast(self.interaction, f, name='interaction')
        
        # delay -- this presumes
        # that the HRF used is a subclass of delay.DelayIRF
    
        self.delays = fmristat.DelayContrast([SSt_DSp, DSt_DSp, SSt_SSp, DSt_SSp],
                                             [[0.5,0.5,-0.5,-0.5],
                                              [-0.5,0.5,-0.5,0.5],
                                              [-1,1,1,-1],
                                              [0.25,0.25,0.25,0.25]],
                                             f,
                                             name='task',
                                             rownames=['speaker',
                                                       'sentence',
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

        self.OLSmodel = fmristat.fMRIStatOLS(self.fmri,
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
        rho.tofile("%s/rho.img" % self.OLSmodel.path, clobber=True)
        
        self.clear()

    def AR(self, **ARopts):

        self.load()

        toc = time.time()
        self.ARmodel = fmristat.fMRIStatAR(self.OLSmodel,
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
##             resid = neuroimaging.modalities.fmri.fMRIImage(FIACpath('fsl/fmristat_run/ARresid.img', subj=subj, run=run))
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

    def view(self, time=N.linspace(0, 191*2.5, 3000)):
        self.plot = MultiPlot(self.term(time=time),
                              time=time,
                              title='Column space for %s' % `self`)
        self.plot.draw()

    def __repr__(self):
        return '<contrast: %s>' % self.name
        

if __name__ == '__main__':

    import sys
    if len(sys.argv) == 3:
        subj, run = map(int, sys.argv[1:])
    else:
        subj, run = (3, 3)

    study = StudyModel(root=io.data_path)
    subject = SubjectModel(subj, study=study)
    runmodel = RunModel(subject, run)
    runmodel.OLS(clobber=True)
    runmodel.AR(clobber=True)

##    run.view()

##     c = contrasts(run)
##     for i in range(len(c)-1): # don't plot delay
##         pylab.figure()
##         c[i].view()

##    pylab.show()
