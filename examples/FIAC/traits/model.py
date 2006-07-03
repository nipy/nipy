import os
import pylab
from neuroimaging import traits
import numpy as N

from neuroimaging.fmri.pca import PCA, MultiPlot
from neuroimaging.fmri import protocol
from neuroimaging.fmri.fmristat import delay
from neuroimaging.fmri.filters import Filter
from neuroimaging.image import Image
import neuroimaging.fmri.fmristat as fmristat

import neuroimaging.statistics.contrast as contrast

from fiac import Run, Subject, Study
from montage import Montage

from readonly import ReadOnlyValidate, HasReadOnlyTraits

#-----------------------------------------------------------------------------#
#
# Canonical drift
#
#-----------------------------------------------------------------------------#

tmax = 190*2.5+1.25
tmin = 0.
drift_fn = protocol.SplineConfound(window=[tmin, tmax], df=5)
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

    def __init__(self, drift=None, hrf=None, **keywords):
        self.drift = drift or canonical_drift
        self.hrf = hrf or delay_hrf
        HasReadOnlyTraits.__init__(self, **keywords)
        self.formula = protocol.Formula(self.drift)

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

local_study = StudyModel(root='/home/analysis/FIAC')
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
                 **keywords):
        Run.__init__(self, subject, id=id)
        Model.__init__(self, drift=drift, hrf=hrf, **keywords)
        self.resultdir = os.path.join(self.root, 'fsl', 'fmristat_run')
        self.subject = subject

        self.begin.convolve(self.hrf[0])
        self.experiment.convolve(self.hrf)
       
        self.formula = self.subject.formula + self.begin + self.experiment
        if self.normalize or self.normalize_reg:

            self.load()
            brainavg = fmristat.WholeBrainNormalize(self.fmri, mask=self.mask)
            if self.normalize_reg:
                brainavg_fn = \
                            protocol.InterpolatedConfound(values=brainavg.avg,
                                                          times=self.shift +
                                                          self.fmri.frametimes)

                self.frameavg = protocol.ExperimentalQuantitative('frameavg',
                                                               brainavg_fn)
            self.clear()
            self.formula += self.frameavg

    def __repr__(self):
        return Run.__repr__(self)

#-----------------------------------------------------------------------------#
#
# Contrasts, with a "view"
#
#-----------------------------------------------------------------------------#

class Contrast(contrast.Contrast, HasReadOnlyTraits):
    
    plot = ReadOnlyValidate(traits.Instance(MultiPlot),
                            desc='Rudimentary multi-line plotter for design')

    def view(self, time=N.linspace(0,191*2.5, 3000)):
        self.plot = MultiPlot(self.term(time=time),
                              time=time,
                              title='Column space for %s' % `self`)
        self.plot.draw()

    def __repr__(self):
        return '<Contrast: %s>' % self.name
        

#-----------------------------------------------------------------------------#
#
# FIAC contrasts of interest
#
#-----------------------------------------------------------------------------#

def contrasts(model):
    """
    This function is specific to the FIAC data, and defines
    the contrasts of interest.
    """

    p = model.experiment
    irf = model.hrf[0] 
    formula = model.formula

    overallF = Contrast(p, formula, name='overallF')

    SSt_SSp = p['SSt_SSp'].astimefn()
    DSt_SSp = p['DSt_SSp'].astimefn()
    SSt_DSp = p['SSt_DSp'].astimefn()
    DSt_DSp = p['DSt_DSp'].astimefn()

    # Average effect

    average = (SSt_SSp + DSt_SSp + SSt_DSp + DSt_DSp) * 0.25

    # important: average is NOT convolved with HRF even though p was!!!
    # same follows for other contrasts below
        
    average = irf.convolve(average)
    average = Contrast(average,
                       formula,
                       name='average')

    # Sentence effect

    sentence = (DSt_SSp + DSt_DSp) * 0.5 - (SSt_SSp + SSt_DSp) * 0.5
    sentence = irf.convolve(sentence)
    sentence = Contrast(sentence, formula, name='sentence')
        
    # Speaker effect

    speaker =  (SSt_DSp + DSt_DSp) / 2. - (SSt_SSp + DSt_SSp) / 2.
    speaker = irf.convolve(speaker)
    speaker = Contrast(speaker, formula, name='speaker')
        
    # Interaction effect

    interaction = SSt_SSp - SSt_DSp - DSt_SSp + DSt_DSp
    interaction = irf.convolve(interaction)
    interaction = Contrast(interaction, formula, name='interaction')
        
    # delay -- this presumes
    # that the HRF used is a subclass of delay.DelayIRF
    
    delays = fmristat.DelayContrast([SSt_DSp, DSt_DSp, SSt_SSp, DSt_SSp],
                                    [[0.5,0.5,-0.5,-0.5],
                                     [-0.5,0.5,-0.5,0.5],
                                     [-1,1,1,-1],
                                     [0.25,0.25,0.25,0.25]],
                                    model.formula,
                                    name='task',
                                    rownames=['speaker',
                                              'sentence',
                                              'interaction',
                                              'overall'],
                                    IRF=model.hrf)

    return [overallF, average, sentence, speaker, interaction, delays]
       

if __name__ == '__main__':

    import pylab
    study = StudyModel(root='/home/analysis/FIAC')
    subject = SubjectModel(0, study=study)
    run = RunModel(subject, 1)
    run.view()

    c = contrasts(run)
    for i in range(len(c)-1): # don't plot delay
        pylab.figure()
        c[i].view()

    pylab.show()
