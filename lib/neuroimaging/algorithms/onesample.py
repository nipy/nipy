import os, types

import numpy as N

from neuroimaging.core.image.image import Image, ImageSequenceIterator
from neuroimaging.algorithms.statistics import onesample
from neuroimaging.algorithms.statistics.regression import RegressionOutput

class ImageOneSample(onesample.OneSampleIterator):
    
    """
    Fit a one sample t to a sequence of images. Input should be either a
    sequence of images (in which case variances are treated as equal) or a
    sequence of pairs of images and weights (in which case the variance of each
    image is a function of the \'weight\' image). The \'weight\' image can be a
    'std', 'var', or 'weight' -- the appropriate transform will be applied.
    """

    def __init__(self, input, outputs=[], path='onesample', ext='.hdr',
                 t=True, sd=True, mean=True, clobber=False, which='mean',
                 varfiximg=None, varatioimg=None, est_varatio=True, est_varfix=True):

        self.which = which
        self.varfiximg = varfiximg
        self.varatioimg = varatioimg

        if type(input[0]) in [types.ListType, types.TupleType]:
            self.haveW = True
            imgs = [val[0] for val in input]
            wimgs = [val[1] for val in input]
            self.iterator = ImageSequenceIterator(imgs)
            self.witerator = ImageSequenceIterator(wimgs, grid=self.iterator.grid)
        else:
            self.haveW = False
            self.iterator = ImageSequenceIterator(input)


        onesample.OneSampleIterator.__init__(self, self.iterator, outputs=outputs)

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
        ## TO DO: rename this methods, something like "getinput"
        if self.haveW:
            w = self.witerator.next(value=self.iterator.grid.itervalue())
        else:
            return 1.

        if self.varatioimg is not None:
            value = self.iterator.grid.itervalue()
            self.varatio = self.varatioimg.next(value=value)
        else:
            self.varatio = 1.
        
        if self.varfiximg is not None:
            value = self.iterator.grid.itervalue()
            self.varfix = self.varfiximg.next(value=value)
        else:
            self.varfix = 0.
            
        return w


    def fit(self):
        return onesample.OneSampleIterator.fit(self, which=self.which)

class ImageOneSampleOutput(RegressionOutput):
    """
    A class to output things a one sample T passes through data. It
    uses the image\'s iterator values to output to an image.

    """

    def __init__(self, grid, basename="", clobber=False, nout=1, path='onesample',
                 ext='.img', **keywords):
        RegressionOutput.__init__(self)
        self.basename = basename
        self.grid = grid
        self.clobber = clobber
        self.nout = nout
        if not os.path.exists(path):
            os.makedirs(path)

        self.img = iter(Image('%s/%s%s' % (path, self.basename, ext),
                                    mode='w', clobber=self.clobber, grid=grid))

    def sync_grid(self, img=None):
        """
        Synchronize an image's grid iterator to self.grid's iterator.
        """
        if img is None:
            img = self.img
        img.grid._iterguy = self.grid._iterguy
        iter(img)
        
    def __iter__(self):
        return self

    def next(self, data=None):
        value = self.grid.itervalue()
        self.img.next(data=data, value=value)

    def extract(self, results):
        raise NotImplementedError

class TOutput(ImageOneSampleOutput):

    Tmax = 100.
    Tmin = -100.

    def __init__(self, grid, **keywords):
        ImageOneSampleOutput.__init__(self, grid, 't', **keywords)

    def extract(self, results):
        return N.clip(results['mean']['t'], self.Tmin, self.Tmax)

class SdOutput(ImageOneSampleOutput):

    def __init__(self, grid, **keywords):
        ImageOneSampleOutput.__init__(self, grid, 'sd', **keywords)

    def extract(self, results):
        return results['mean']['sd']

class MeanOutput(ImageOneSampleOutput):


    def __init__(self, grid, **keywords):
        ImageOneSampleOutput.__init__(self, grid, 'effect', **keywords)

    def extract(self, results):
        return results['mean']['mu']

class VaratioOutput(ImageOneSampleOutput):

    def __init__(self, grid, **keywords):
        ImageOneSampleOutput.__init__(self, grid, 'varatio', **keywords)

    def extract(self, results):
        return results['varatio']['varatio']

class VarfixOutput(ImageOneSampleOutput):

    def __init__(self, grid, **keywords):
        ImageOneSampleOutput.__init__(self, grid, 'varfix', **keywords)

    def extract(self, results):
        return results['varatio']['varfix']
