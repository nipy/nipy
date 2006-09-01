import string, os, gc, time, urllib

import numpy as N
import pylab

import enthought.traits.ui
from neuroimaging import traits

from neuroimaging.modalities.fmri import fMRIImage
from neuroimaging.modalities.fmri.protocol import ExperimentalFactor
from neuroimaging.core.image import Image

from protocol import event_protocol, block_protocol
from io import urlexists
from readonly import ReadOnlyValidate, HasReadOnlyTraits

#-----------------------------------------------------------------------------#

class Study(HasReadOnlyTraits):

    root = ReadOnlyValidate(traits.Str, desc='Root FIAC directory.')

local_study = Study(root='/home/analysis/FIAC')
www_study = Study(root='http://kff.stanford.edu/FIAC')

#-----------------------------------------------------------------------------#

class Subject(HasReadOnlyTraits):

    id = ReadOnlyValidate(traits.Range(low=0,high=15),
                          desc='Which FIAC subject? Ranges from 0 to 15.')

    study = ReadOnlyValidate(Study)
    root = ReadOnlyValidate(traits.Str,
                            desc='Root directory for this subject.')

    event = traits.ReadOnly(desc='Which runs are event-related runs?')
    block = traits.ReadOnly(desc='Which runs are block runs?')

    def __init__(self, id, study=local_study):

        self.study = study
        self.id = id
        self.root = os.path.join(self.study.root, 'fiac%d' % self.id)
        self._getruns()

    def __repr__(self):
        return '<FIAC subject %d>' % self.id

    def _getruns(self):

        _path = lambda x: os.path.join(self.root, x)
        event = []
        block = []

        for run in range(1,5):
            if urlexists(_path('subj%d_bloc_fonc%d.txt' % (self.id, run))):
                block.append(run)
            elif urlexists(_path('subj%d_evt_fonc%d.txt' % (self.id, run))):
                event.append(run)

        self.event = event
        self.block = block
        

#-----------------------------------------------------------------------------#

class Run(HasReadOnlyTraits):
    
    subject = ReadOnlyValidate(Subject)
    id = ReadOnlyValidate(traits.Range(low=1, high=5, desc='Which run? Ranges from 1 to 5. '))

    root = ReadOnlyValidate(traits.Str,
                            desc='Directory containing data for this run.')

    fmrifile = ReadOnlyValidate(traits.Str, desc="Path for fMRI file.", label='fMRI data')
    maskfile = ReadOnlyValidate(traits.Str, desc="Path for mask.", label='Mask')
    anatfile = ReadOnlyValidate(traits.Str, desc="Path for hires anatomy.",
                                label='Hi-res anatomy')

    fmri = traits.Instance(fMRIImage)
    mask = traits.Instance(Image)
    anat = traits.Instance(Image)


    type = ReadOnlyValidate(traits.Trait(['block', 'event'],
                                         desc='What type of run?'))

    begin = traits.Instance(ExperimentalFactor)
    experiment = traits.Instance(ExperimentalFactor)

    def __init__(self, subject, id, **keywords):
        self.subject = subject
        self.id = id
        traits.HasTraits.__init__(self, **keywords)

        if self.id in self.subject.block:
            self.type = 'block'
        elif self.id in self.subject.event:
            self.type = 'event'
        else:
            raise ValueError, 'run %d not found for %s' % (self.id, `self.subject`)

        self.root = os.path.join(self.subject.root, 'fonc%d' % self.id)
        self._getimages()
        self._getprotocol()

    def __repr__(self):
        return '<FIAC subject %d, run %d>' % (self.subject.id, self.id)

    def _getimages(self):
        """
        Find mask and anatomy image for given subject/run.
        """

        _path = lambda x: os.path.join(self.root, x)
        self.fmrifile = _path('fsl/filtered_func_data.img')
        self.maskfile = _path('fsl/mask.img')
        self.anatfile = _path('fsl/highres2standard.img')

    def load(self):
        if urlexists(self.fmrifile):
            self.fmri = fMRIImage(self.fmrifile)
            
        if urlexists(self.maskfile):
            self.mask = Image(self.maskfile)
            
        if urlexists(self.anatfile):
            self.anat = Image(self.anatfile, usematfile=False)

    def clear(self):
        del(self.fmri); del(self.mask); del(self.anat)
        
    def _getprotocol(self):
        _path = lambda x: os.path.join(self.subject.root,
                                       x)
        if self.id in self.subject.block: 
            self.begin, self.experiment = block_protocol(_path('subj%d_bloc_fonc%d.txt' % (self.subject.id, self.id)))
        elif self.id in self.subject.event:
            self.begin, self.experiment = event_protocol(_path('subj%d_evt_fonc%d.txt' % (self.subject.id, self.id)))

if __name__ == '__main__':
    study = Study(root='/home/analysis/FIAC')
    subjects = [Subject(i) for i in range(16)]

    runs = []
    for i in range(16):
        subject = subjects[i]
        subj_runs = []
        for run in subject.event + subject.block:
            cur_run = Run(subject, id=run)
            subj_runs.append(cur_run)
        runs.append(subj_runs)

