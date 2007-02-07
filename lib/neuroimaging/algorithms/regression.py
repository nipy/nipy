import os, csv, fpformat

import numpy as N

from neuroimaging.core.image.image import Image
from neuroimaging.algorithms.statistics.regression import RegressionOutput

class ImageRegressionOutput(RegressionOutput):
    """
    A class to output things in GLM passes through Image data. It
    uses the image's iterator values to output to an image.
    """

    def __init__(self, grid, nout=1, outgrid=None, it=None):
        RegressionOutput.__init__(self, grid, nout, outgrid)

        if self.nout > 1:
            self.grid = self.grid.replicate(self.nout)

        self.img = Image(N.zeros(outgrid.shape), grid=outgrid)
        if it is None:
            self.it = self.img.slice_iterator(mode='w')
        else:
            self.it = it


class TContrastOutput(ImageRegressionOutput):

    def __init__(self, grid, contrast, path='.', subpath='contrasts', ext=".img",
                 effect=True, sd=True, t=True, nout=1, outgrid=None,
                 clobber=False):
        ImageRegressionOutput.__init__(self, grid, nout, outgrid)
        self.contrast = contrast
        self.effect = effect
        self.sd = sd
        self.t = t
        self._setup_contrast()
        self._setup_output(clobber, path, subpath, ext)

    def _setup_contrast(self, **extra):
        self.contrast.getmatrix(**extra)

    def _setup_output(self, clobber, path, subpath, ext):
        outdir = os.path.join(path, subpath, self.contrast.name)
        self.timg, self.timg_it = self._setup_img(clobber, outdir, ext, 't')

        if self.effect:
            self.effectimg, self.effectimg_it = self._setup_img(clobber, outdir, ext, 'effect')
        if self.sd:
            self.sdimg, self.sdimg_it = self._setup_img(clobber, outdir, ext, 'sd')

        outname = os.path.join(outdir, 'matrix.csv')
        outfile = file(outname, 'w')
        outfile.write(','.join(fpformat.fix(x,4) for x in self.contrast.matrix) + '\n')
        outfile.close()

        outname = os.path.join(outdir, 'matrix.bin')
        outfile = file(outname, 'w')
        self.contrast.matrix = self.contrast.matrix.astype('<f8')
        self.contrast.matrix.tofile(outfile)
        outfile.close()

    def extract(self, results):
        return results.Tcontrast(self.contrast.matrix, sd=self.sd, t=self.t)

    def set_next(self, data):
        self.timg_it.next().set(data.t)
        if self.effect:
            self.effectimg_it.next().set(data.effect)
        if self.sd:
            self.sdimg_it.next().set(data.effect)


class FContrastOutput(ImageRegressionOutput):

    def __init__(self, grid, contrast, path='.', clobber=False,
                 subpath='contrasts', ext='.img', nout=1, outgrid=None):
        ImageRegressionOutput.__init__(self, grid, nout, outgrid)
        self.contrast = contrast
        self._setup_contrast()
        self._setup_output(clobber, path, subpath, ext)

    def _setup_contrast(self, **extra):
        self.contrast.getmatrix(**extra)

    def _setup_output(self, clobber, path, subpath, ext):
        outdir = os.path.join(path, subpath, self.contrast.name)
        self.img, self.it = self._setup_img(clobber, outdir, ext, 'F')

        outname = os.path.join(outdir, 'matrix.csv')
        outfile = file(outname, 'w')
        writer = csv.writer(outfile)
        for row in self.contrast.matrix:
            writer.writerow([fpformat.fix(x, 4) for x in row])
        outfile.close()

        outname = os.path.join(outdir, 'matrix.bin')
        outfile = file(outname, 'w')
        self.contrast.matrix = self.contrast.matrix.astype('<f8')
        self.contrast.matrix.tofile(outfile)
        outfile.close()

    def extract(self, results):
        return results.Fcontrast(self.contrast.matrix).F


class ResidOutput(ImageRegressionOutput):

    def __init__(self, grid, path='.', nout=1, clobber=False, basename='resid',
                 ext='.img', outgrid=None):
        ImageRegressionOutput.__init__(self, grid, nout, outgrid)
        outdir = os.path.join(path)

        self.img, self.it = self._setup_img(clobber, outdir, ext, basename)
        self.nout = self.grid.shape[0]

    def extract(self, results):
        return results.resid
    
