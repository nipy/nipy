import urllib, gc
import numpy as N

from neuroimaging import traits

from neuroimaging.core.api import Image
from neuroimaging.algorithms.interpolation import ImageInterpolator
from neuroimaging.core.reference.mapping import Affine
from neuroimaging.core.reference.grid import SamplingGrid

from fiac import Run

standard = Image('http://kff.stanford.edu/FIAC/avg152T1_brain.img',
                 ignore_origin=True,
                 abs_pixdim=True)
mni_grid = Image('http://kff.stanford.edu/FIAC/avg152T1_brain.img').grid

class Resampler(Run):

    clobber = traits.true(desc='Clobber existing output images?')
    input_grid = traits.Instance(SamplingGrid,
                                 desc='Input grid used for resampling -- note it uses FSL coordinates.')
    output_grid = traits.Instance(SamplingGrid,
                                  desc='Output grid of MNI template -- note it uses FSL coordinates.')

    def __init__(self, *args, **keywords):
        Run.__init__(self, *args, **keywords)
        self.input_grid, self.output_grid = self._get_grid()

    def _get_grid(self):
        fslmatstr = urllib.urlopen(self.joinpath('fsl/example_func2standard.mat')).read().split()
        fslmat =  N.array(fslmatstr).astype(N.float64)
        fslmat.shape = (4, 4)

        M = [[0,0,1,0],
             [0,1,0,0],
             [1,0,0,0],
             [0,0,0,1]]

        fslmat = N.dot(M, N.dot(fslmat, M))
        
        tmpimage = Image(self.maskfile, ignore_origin=True, abs_pixdim=True)
                         
        input_grid = tmpimage.grid
        
        input_coords = input_grid.output_coords
        output_coords = standard.grid.output_coords
        fworld2sworld = Affine(fslmat)
        svoxel2fworld = fworld2sworld.inverse() * standard.grid.mapping

        output_grid = SamplingGrid(standard.grid.shape, svoxel2fworld, 
                                   input_coords, output_coords)
        return input_grid, output_grid

    def __repr__(self):
        return '< Resampler for FIAC subject %d, run %d>' % (self.subject.id, self.id)

    def __str__(self):
        return Run.__repr__(self)

    def resample(self, image, clobber=None, outfile=None):
        if clobber is None:
            clobber = self.clobber

        image = Image(image)
        image.grid = self.input_grid # presumes data is same shape
                                       # as self.maskfile, but ensures
                                       # that abs(pixdim) is used and
                                       # origin is ignored

        interp = ImageInterpolator(image)
        interp_data = interp(self.output_grid.range())

        new = Image(interp_data, grid=mni_grid)
        if outfile is not None:
            new.tofile(outfile, clobber=clobber)
            return Image(outfile)
        else:
            return new

        del(interp)
        del(interp_data)
        del(new)
        gc.collect()

