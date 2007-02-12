import os

from neuroimaging import traits

from neuroimaging.modalities.fmri import fMRIImage
from neuroimaging.modalities.fmri.protocol import ExperimentalFactor
from neuroimaging.core.api import Image

from protocol import event_protocol, block_protocol
from io import urlexists, data_path
from readonly import ReadOnlyValidate, HasReadOnlyTraits

class Study(HasReadOnlyTraits):

    root = ReadOnlyValidate(traits.Str, desc='Root FIAC directory.')

    def joinpath(self, x):
        return os.path.join(self.root, x)

local_study = Study(root=data_path)
www_study = Study(root='http://kff.stanford.edu/FIAC')



class Subject(HasReadOnlyTraits):

    id = ReadOnlyValidate(traits.Range(low=0,high=15),
                          desc='Which FIAC subject? Ranges from 0 to 15.')

    study = ReadOnlyValidate(traits.Instance(Study))
    root = ReadOnlyValidate(traits.Str,
                            desc='Root directory for this subject.')

    event = traits.ReadOnly(desc='Which runs are event-related runs?')
    block = traits.ReadOnly(desc='Which runs are block runs?')

    def __init__(self, id, study=local_study):

        self.study = study
        self.id = id
        self.root = self.study.joinpath('fiac%d' % self.id)
        self.event, self.block = self._getruns()

    def __repr__(self):
        return '<FIAC subject %d>' % self.id

    def joinpath(self, x):
        return os.path.join(self.root, x)

    def _getruns(self):

        event = []
        block = []
        for run in range(1, 5):
            if urlexists(self.joinpath('subj%d_bloc_fonc%d.txt' % (self.id, run))):
                block.append(run)
            elif urlexists(self.joinpath('subj%d_evt_fonc%d.txt' % (self.id, run))):
                event.append(run)

        return event, block
        

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

    design_type = ReadOnlyValidate(traits.Trait(['block', 'event'],
                                           desc='What type of run?'))

    begin = traits.Instance(ExperimentalFactor)
    experiment = traits.Instance(ExperimentalFactor)

    def __init__(self, subject, id, **keywords):
        self.subject = subject
        self.id = id
        traits.HasTraits.__init__(self, **keywords)

        if self.id in self.subject.block:
            self.design_type = 'block'
        elif self.id in self.subject.event:
            self.design_type = 'event'
        else:
            raise ValueError, 'run %d not found for %s' % (self.id, `self.subject`)

        self.root = self.subject.joinpath('fonc%d' % self.id)
        self._getimages()
        self._getprotocol()

    def __repr__(self):
        return '<FIAC subject %d, run %d>' % (self.subject.id, self.id)

    def joinpath(self, x):
        return os.path.join(self.root, x)

    def _getimages(self):
        """
        Find mask and anatomy image for given subject/run.
        """

        self.fmrifile = self.joinpath('fsl/filtered_func_data.img')
        self.maskfile = self.joinpath('fsl/mask.img')
        self.anatfile = self.joinpath('fsl/highres2standard.img')

    def load(self):
        """
        Open each of the files associated with this run.
        """

        if urlexists(self.fmrifile):
            self.fmri = fMRIImage(self.fmrifile)
            
        if urlexists(self.maskfile):
            self.mask = Image(self.maskfile)
            
        if urlexists(self.anatfile):
            self.anat = Image(self.anatfile, usemat=False)

    def clear(self):
        """
        Close each of the files associated with this run.
        """
        del(self.fmri)
        del(self.mask)
        del(self.anat)
        
    def _getprotocol(self):
        if self.id in self.subject.block: 
            self.begin, self.experiment = block_protocol(self.subject.joinpath('subj%d_bloc_fonc%d.txt' % (self.subject.id, self.id)))
        elif self.id in self.subject.event:
            self.begin, self.experiment = event_protocol(self.subject.joinpath('subj%d_evt_fonc%d.txt' % (self.subject.id, self.id)))

subjects = [0,1,3,4,6,7,8,9,10,11,12,13,14,15]

