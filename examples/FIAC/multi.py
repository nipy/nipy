import glob

from os.path import join, exists
import os

from neuroimaging.core.api import Image
from neuroimaging.algorithms.onesample import ImageOneSample

import fixed, fmristat, fiac

class Multi(fixed.Fixed):

    def __init__(self, **args):
        fixed.Fixed.__init__(self, **args)
        self.effects = [Image(fixed.Fixed.resultpath(self, join('fiac%d' % s, 'effect.nii')))
                       for s in fiac.subjects]
        self.sds = [Image(fixed.Fixed.resultpath(self, join('fiac%d' % s, 'sd.nii')))
                    for s in fiac.subjects]
        
    def resultpath(self, path):
        return join(self.root, 'multi', self.design, self.which, self.contrast, path)

    def fit(self, clobber=True):

        if not exists(self.resultpath("")):
            os.makedirs(self.resultpath(""))
            
        input = [(self.effects[i], self.sds[i]) for i in range(len(self.effects))]

        fitter = ImageOneSample(input, path=self.resultpath(""), clobber=clobber, which='varatio')
        fitter.fit()

        fitter = ImageOneSample(input, path=self.resultpath(""), clobber=clobber, which='mean')
        fitter.fit()

        [os.remove(fixed.Fixed.resultpath(self, join('fiac%d' % s, 'effect.nii'))) for s in fiac.subjects]
        [os.remove(fixed.Fixed.resultpath(self, join('fiac%d' % s, 'sd.nii'))) for s in fiac.subjects]
