"""
TODO
"""
__docformat__ = 'restructuredtext'

import numpy as np

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
        """
        :Parameters:
            input : TODO
                TODO
            outputs : TODO
                TODO
            path : ``string``
                TODO
            ext : ``string``
                TODO
            t : ``bool``
                TODO
            sd : ``bool``
                TODO
            mean : ``bool``
                TODO
            clobber : ``bool``
                TODO
            which : ``string``
                TODO
            varfiximg : TODO
                TODO
            varatioimg : TODO
                TODO
            est_varatio : ``bool``
                TODO
            est_varfix : ``bool``
        """

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
            self.witerator = ImageSequenceIterator(wimgs)
        else:
            self.haveW = False
            self.iterator = ImageSequenceIterator(input)

        onesample.OneSampleIterator.__init__(self, self.iterator,
                                             outputs=outputs)

        grid = self.iterator.imgs[0].grid
        if self.which == 'mean':
            if t:
                self.outputs.append(TOutput(grid, path=path,
                                            clobber=clobber, ext=ext))
            if sd:
                self.outputs.append(SdOutput(grid, path=path,
                                             clobber=clobber, ext=ext))
            if mean:
                self.outputs.append(MeanOutput(grid, path=path,
                                               clobber=clobber, ext=ext))
        else:
            if est_varatio:
                self.outputs.append(VaratioOutput(grid, path=path,
                                                  clobber=clobber, ext=ext))

            if est_varfix:
                self.outputs.append(VarfixOutput(grid, path=path,
                                                 clobber=clobber, ext=ext))

    def weights(self):
        """
        :Returns: TODO
        """
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
        """
        :Returns: TODO
        """
        return onesample.OneSampleIterator.fit(self, which=self.which)

class ImageOneSampleOutput(RegressionOutput):
    """
    A class to output things a one sample T passes through data. It
    uses the image\'s iterator values to output to an image.

    """

    def __init__(self, grid, nout=1, basename="", clobber=False,
                 path='onesample', ext='.img'):
        """
        :Parameters:
            grid : TODO
                TODO
            nout : ``int``
                TODO
            basename : ``string``
                TODO
            clobber : ``bool``
                TODO
            path : ``string``
                TODO
            ext : ``string``
                TODO
        """
        RegressionOutput.__init__(self, grid, nout)
        self.img, self.it = self._setup_img(clobber, path, ext, basename)


class TOutput(ImageOneSampleOutput):
    """
    TODO
    """

    def __init__(self, grid, Tmax=100, Tmin=-100, **keywords):
        """
        :Parameters:
            grid : TODO
                TODO
            Tmax : TODO
                TODO
            Tmin : TODO
                TODO
            keywords : ``dict``
                Keyword arguments passed to `ImageOneSampleOutput.__init__`
        """
        ImageOneSampleOutput.__init__(self, grid, basename='t', **keywords)
        self.Tmax = Tmax
        self.Tmin = Tmin

    def extract(self, results):
        return np.clip(results['mean']['t'], self.Tmin, self.Tmax)

class SdOutput(ImageOneSampleOutput):
    """
    TODO
    """

    def __init__(self, grid, **keywords):
        """
        :Parameters:
            grid : TODO
                TODO
            keywords : ``dict``
                Keyword arguments passed to `ImageOneSampleOutput.__init__`
        """
        ImageOneSampleOutput.__init__(self, grid, basename='sd', **keywords)

    def extract(self, results):
        """
        :Parameters:
            results : TODO
                TODO

        :Returns: TODO
        """
        return results['mean']['sd']

class MeanOutput(ImageOneSampleOutput):
    """
    TODO
    """

    def __init__(self, grid, **keywords):
        """
        :Parameters:
            grid : TODO
                TODO
            keywords : ``dict``
                Keyword arguments passed to `ImageOneSampleOutput.__init__`
        """
        ImageOneSampleOutput.__init__(self, grid, basename='effect', **keywords)

    def extract(self, results):
        """
        :Parameters:
            results : TODO
                TODO

        :Returns: TODO
        """
        return results['mean']['mu']

class VaratioOutput(ImageOneSampleOutput):
    """
    TODO
    """

    def __init__(self, grid, **keywords):
        """
        :Parameters:
            grid : TODO
                TODO
            keywords : ``dict``
                Keyword arguments passed to `ImageOneSampleOutput.__init__`
        """
        ImageOneSampleOutput.__init__(self, grid, basename='varatio', **keywords)

    def extract(self, results):
        """
        :Parameters:
            results : TODO
                TODO

        :Returns: TODO
        """
        return results['varatio']['varatio']

class VarfixOutput(ImageOneSampleOutput):
    """
    TODO
    """

    def __init__(self, grid, **keywords):
        """
        :Parameters:
            grid : TODO
                TODO
            keywords : ``dict``
                Keyword arguments passed to `ImageOneSampleOutput.__init__`
        """
        ImageOneSampleOutput.__init__(self, grid, basename='varfix', **keywords)

    def extract(self, results):
        """
        :Parameters:
            results : TODO
                TODO

        :Returns: TODO
        """
        return results['varatio']['varfix']
