import os, types

import numpy as N
from neuroimaging import traits

from neuroimaging.core.image.image import Image, ImageSequenceIterator
from neuroimaging.algorithms.statistics import onesample
from neuroimaging.algorithms.statistics.regression import RegressionOutput

class ImageOneSample(onesample.OneSampleIterator):
    
    """
    Fit a one sample t to a sequence of images. Input should be either a sequence of images (in which
    case variances are treated as equal) or a sequence of pairs of images and weights (in which case
    the variance of each image is a function of the \'weight\' image). The \'weight\' image
    can be a 'std', 'var', or 'weight' -- the appropriate transform will be applied.
    """

    all = traits.false
    haveW = traits.false
    t = traits.true
    sd = traits.true
    mean = traits.true
    clobber = traits.false
    path = traits.Str('onesample')
    basename = traits.Str()
    ext = traits.Str('.hdr')
    varatioimg = traits.Any()
    est_varatio = traits.true
    varfiximg = traits.Any()
    est_varfix = traits.true
    which = traits.Trait('mean', 'varatio')

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

    def __init__(self, input, outputs=[], **keywords):

        traits.HasTraits.__init__(self, **keywords)


        if type(input[0]) in [types.ListType, types.TupleType]:
            self.haveW = True
            imgs = [val[0] for val in input]
            wimgs = [val[1] for val in input]
            self.iterator = ImageSequenceIterator(imgs)

            ## don't know if this should go here....
            #if self.all:
            #    self.iterator.grid.itertype = 'all'
            #    self.iterator.grid = iter(self.iterator.grid)

            self.witerator = ImageSequenceIterator(wimgs, grid=self.iterator.grid)
        else:
            self.iterator = ImageSequenceIterator(input)

        onesample.OneSampleIterator.__init__(self, self.iterator, outputs=outputs, **keywords)

        self.outputs = outputs
        if self.which == 'mean':
            if self.t:
                self.outputs.append(TOutput(self.iterator.grid, path=self.path,
                                            clobber=self.clobber, ext=self.ext))
            if self.sd:
                self.outputs.append(SdOutput(self.iterator.grid, path=self.path,
                                             clobber=self.clobber, ext=self.ext))
            if self.mean:
                self.outputs.append(MeanOutput(self.iterator.grid, path=self.path,
                                               clobber=self.clobber, ext=self.ext))
        else:
            if self.est_varatio:
                self.outputs.append(VaratioOutput(self.iterator.grid, path=self.path,
                                                  clobber=self.clobber, ext=self.ext))

            if self.est_varfix:
                self.outputs.append(VarfixOutput(self.iterator.grid, path=self.path,
                                                 clobber=self.clobber, ext=self.ext))

    def fit(self):
        onesample.OneSampleIterator.fit(self, which=self.which)

class ImageOneSampleOutput(RegressionOutput):
    """
    A class to output things a one sample T passes through data. It
    uses the image\'s iterator values to output to an image.

    """

    nout = traits.Int(1)
    clobber = traits.false
    path = traits.Str('onesample')
    #basename = traits.Str()
    ext = traits.Str('.img')

    def __init__(self, grid, basename="", **keywords):
        RegressionOutput.__init__(self, **keywords)
        self.basename = basename
        self.grid = grid
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.img = iter(Image('%s/%s%s' % (self.path, self.basename, self.ext),
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
        return 0.

class TOutput(ImageOneSampleOutput):

    Tmax = 100.
    Tmin = -100.

    def __init__(self, grid, **keywords):
        ImageOneSampleOutput(self, grid, 't', **keywords)

    def extract(self, results):
        return N.clip(results.t, self.Tmin, self.Tmax)

class SdOutput(ImageOneSampleOutput):

    def __init__(self, grid, **keywords):
        ImageOneSampleOutput(self, grid, 'sd', **keywords)

    def extract(self, results):
        return results.sd

class MeanOutput(ImageOneSampleOutput):


    def __init__(self, grid, **keywords):
        ImageOneSampleOutput(self, grid, 'effect')


    def extract(self, results):
        return results.mu

class VaratioOutput(ImageOneSampleOutput):

    def __init__(self, grid, **keywords):
        ImageOneSampleOutput(self, grid, 'varatio')

    def extract(self, results):
        return results.varatio

class VarfixOutput(ImageOneSampleOutput):

    def __init__(self, grid, **keywords):
        ImageOneSampleOutput(self, frid, 'varfix')

    def extract(self, results):
        return results.varfix
