__docformat__ = 'restructuredtext'

import numpy as N

from neuroimaging.core.image.image import ImageSequenceIterator
from neuroimaging.algorithms.statistics import onesample
from neuroimaging.algorithms.statistics.regression import RegressionOutput

class ImageOneSample(onesample.OneSampleIterator):
    
    """
    Fit a one sample t to a sequence of images. Input should be either a
    sequence of images (in which case variances are treated as equal) or a
    sequence of pairs of images and weights (in which case the variance of each
    image is a function of the 'weight' image). The 'weight' image can be a
    'std', 'var', or 'weight' -- the appropriate transform will be applied.
    """

    def __init__(self, input, outputs=None, path='onesample', ext='.hdr',
                 t=True, sd=True, mean=True, clobber=False, which='mean',
                 varfiximg=None, varatioimg=None, est_varatio=True,
                 est_varfix=True):

        if outputs is None:
            outputs = []
        self.which = which
        self.varfiximg = varfiximg
        self.varatioimg = varatioimg

        if isinstance(input[0], (list, tuple)):
            self.haveW = True
            imgs = [val[0] for val in input]
            wimgs = [val[1] for val in input]
            self.iterator = ImageSequenceIterator(imgs)
            self.witerator = ImageSequenceIterator(wimgs,
                                                   grid=self.iterator.grid)
        else:
            self.haveW = False
            self.iterator = ImageSequenceIterator(input)

        onesample.OneSampleIterator.__init__(self, self.iterator,
                                             outputs=outputs)

        if self.which == 'mean':
            if t:
                self.outputs.append(TOutput(self.iterator.grid, path=path,
                                            clobber=clobber, ext=ext))
            if sd:
                self.outputs.append(SdOutput(self.iterator.grid, path=path,
                                             clobber=clobber, ext=ext))
            if mean:
                self.outputs.append(MeanOutput(self.iterator.grid, path=path,
                                               clobber=clobber, ext=ext))
        else:
            if est_varatio:
                self.outputs.append(VaratioOutput(self.iterator.grid, path=path,
                                                  clobber=clobber, ext=ext))

            if est_varfix:
                self.outputs.append(VarfixOutput(self.iterator.grid, path=path,
                                                 clobber=clobber, ext=ext))

    def weights(self):
        if self.haveW:
            return self.witerator.next()
        else:
            return 1.
            
    def _getinputs(self):
        if self.varatioimg is not None:
            self.varatio = self.varatioimg.next()            
        else:
            self.varatio = 1.
        
        if self.varfiximg is not None:
            self.varfix = self.varfiximg.next()
        else:
            self.varfix = 0.

    def fit(self):
        return onesample.OneSampleIterator.fit(self, which=self.which)

class ImageOneSampleOutput(RegressionOutput):
    """
    A class to output things a one sample T passes through data. It
    uses the image\'s iterator values to output to an image.

    """

    def __init__(self, grid, nout=1, basename="", clobber=False,
                 path='onesample', ext='.img'):
        RegressionOutput.__init__(self, grid, nout)
        self.img, self.it = self._setup_img(clobber, path, ext, basename)


class TOutput(ImageOneSampleOutput):

    def __init__(self, grid, Tmax=100, Tmin=-100, **keywords):
        ImageOneSampleOutput.__init__(self, grid, basename='t', **keywords)
        self.Tmax = Tmax
        self.Tmin = Tmin

    def extract(self, results):
        return N.clip(results['mean']['t'], self.Tmin, self.Tmax)

class SdOutput(ImageOneSampleOutput):

    def __init__(self, grid, **keywords):
        ImageOneSampleOutput.__init__(self, grid, basename='sd', **keywords)

    def extract(self, results):
        return results['mean']['sd']

class MeanOutput(ImageOneSampleOutput):

    def __init__(self, grid, **keywords):
        ImageOneSampleOutput.__init__(self, grid, basename='effect', **keywords)

    def extract(self, results):
        return results['mean']['mu']

class VaratioOutput(ImageOneSampleOutput):

    def __init__(self, grid, **keywords):
        ImageOneSampleOutput.__init__(self, grid, basename='varatio', **keywords)

    def extract(self, results):
        return results['varatio']['varatio']

class VarfixOutput(ImageOneSampleOutput):

    def __init__(self, grid, **keywords):
        ImageOneSampleOutput.__init__(self, grid, basename='varfix', **keywords)

    def extract(self, results):
        return results['varatio']['varfix']
