import os, gc, urllib, glob
import numpy as N

from neuroimaging.core.api import Image
from neuroimaging.algorithms.interpolation import ImageInterpolator
from neuroimaging.algorithms.onesample import ImageOneSample
from neuroimaging.core.reference.mapping import Affine
from neuroimaging.core.reference.grid import SamplingGrid

from neuroimaging import traits

from readonly import ReadOnlyValidate

import model, io, fmristat

class Fixed(model.Study):

    design = ReadOnlyValidate(['block', 'event'], desc='Block or event?')
    which = ReadOnlyValidate(['contrasts', 'delays'], desc='Contrast or delays?')
    contrast = ReadOnlyValidate(['average', 'sentence', 'speaker', 'interaction'], desc='Which contrast?')

    def __init__(self, design='block', which='contrasts', root='.', contrast='average'):
        model.Study.__init__(self, root=root)
        self.design = design
        self.which = which
        self.contrast = contrast
        
    def resultpath(self, path):
        return os.path.join(self.root, 'fixed', self.design, self.which, self.contrast, path)

class Subject(model.Subject):

    runs = ReadOnlyValidate(traits.List, desc='Runs for this subject')
    contrast = ReadOnlyValidate(traits.Instance(Fixed), desc='Contrast in question')

    def __init__(self, id, contrast):
        self.id = id
        model.Subject.__init__(self, id, study=contrast)
        runs = []
 
        for run in getattr(self, self.study.design):
            runmodel = model.Run(self, run)
            runs.append(runmodel)
        self.runs = runs
        if not os.path.exists(self.resultpath("")):
            os.makedirs(self.resultpath(""))
        print 'runs', runs

    def resultpath(self, path):
        return os.path.join(self.study.resultpath(""), "fiac%d" % self.id, path)
    
    def fmristat_result(self, stat='t'):
        return fmristat.fixed(subject=self.id,
                           which=self.study.which,
                           contrast=self.study.contrast,
                           stat=stat,
                           design=self.study.design)

    def result(self, run, stat='effect', resampled=True):

        if not resampled:
            return run.result(which=self.study.which, contrast=self.study.contrast, stat=stat)
        else:
            return Image(self.resultpath('%s_%d.nii' % (stat, run.id)))

    def resample(self, run, stat='effect', clobber=False):
        """
        Resample the results of a given run using output of FSL
        """

        image = self.result(run, stat=stat, resampled=False)
        outname = self.resultpath("%s_%d.nii" % (stat, run.id))
        matfile = model.Run.joinpath(run, 'fsl/example_func2standard.mat')
        fslmat = N.array([float(e) for e in
                          urllib.urlopen(matfile).read().split()])
        fslmat.shape = (4,4)
        M = [[0,0,1,0],
             [0,1,0,0],
             [1,0,0,0],
             [0,0,0,1]]

        fslmat = N.dot(M, N.dot(fslmat, M))
    
        # standard = Image('http://kff.stanford.edu/FIAC/avg152T1_brain.img')
        standard = Image(self.study.joinpath('avg152T1_brain.img'))

        # force ignore of origin, absolute value of pixdims

        for im in [image, standard]:
            im.grid.mapping.transform[0:3,-1] = 0.
            im.grid.mapping.transform[0:3,0:3] = N.fabs(im.grid.mapping.transform[0:3,0:3])
            
        input_coords = image.grid.output_coords
        output_coords = standard.grid.output_coords

        fworld2sworld = Affine(fslmat)
        voxel2fworld = fworld2sworld.inverse() * standard.grid.mapping

        output_grid = SamplingGrid(standard.grid.shape, voxel2fworld, input_coords, output_coords)

        interp = ImageInterpolator(image)
        interp_data = interp(output_grid.range())

        new = Image(interp_data, grid=standard.grid)
        new.tofile(outname, clobber=True)
        del(interp); del(interp_data) ; del(new) ; gc.collect()

        outimg = Image(outname)
        return outimg

    def fit(self):
        """
        One sample fixed effects analysis
        """
        
        [self.resample(run, stat='effect', clobber=True) for run in self.runs]
        [self.resample(run, stat='sd', clobber=True) for run in self.runs]

        effects = [self.result(run, stat='effect') for run in self.runs]
        sds = [self.result(run, stat='sd') for run in self.runs]
        effect = Image(N.zeros(effects[0].grid.shape), grid=effects[0].grid)
        sd = Image(N.zeros(effects[0].grid.shape), grid=effects[0].grid)
        d = 0
        v = 0
        for i in range(len(effects)):
            w = N.nan_to_num(1. / sds[i][:]**2)
            effect[:] += effects[i][:] * w
            d += w

        effect[:] /= d
        
        sd[:] = N.sqrt(N.nan_to_num(1./d))
        _sd = sd.tofile(self.resultpath("sd.nii"), clobber=True)
        _effect = effect.tofile(self.resultpath("effect.nii"), clobber=True)

        [os.remove(sd._source.filename) for sd in sds]
        [os.remove(effect._source.filename) for effect in effects]
        [run.cleanup() for run in self.runs]

        return (_effect, _sd) 

    def cleanup(self):
        """
        Remove uncompressed .nii files from results directory.
        """
        [os.remove(imfile) for imfile in glob.glob(self.resultpath("*nii"))]

def _fix_origin(img):
    
    """
    FSL does not really deal with the origin and the signs of the pixdim.

    This function ensures that all resampled outputs have the same
    origin and pixdim
    """

    standard = Image('/home/stow/fsl/etc/standard/avg152T1_brain.img')
    img._source.header['pixdim'][1:] = standard._source.header['pixdim'][1:]

    hdrfile = file(img._source.header_file, 'rb+')
    img._source.write_header(hdrfile)
    hdrfile.close()
    return Image(img._source.filename)

def _cor_result(im1, im2):
    x = N.nan_to_num(im1[:])
    y = N.nan_to_num(im2[:])
    x.shape = N.product(x.shape)
    y.shape = N.product(y.shape)
    return N.corrcoef(x, y)[0,1]


def run(root=io.data_path, subj=3, resample=True, fit=True):

    for contrast in ['average', 'interaction', 'speaker', 'sentence']:
        for which in ['contrasts', 'delays']:
            for design in ['event', 'block']:
                fixed = Fixed(root=io.data_path, which=which, contrast=contrast, design=design)
                subject = Subject(subj, fixed)

                if fit:
                    effect, sd = subject.fit()


def compare(subj=3, which='delays', contrast='speaker', design='event'):
    fixed = Fixed(root=io.data_path, which=which, contrast=contrast, design=design)
    subject = Subject(subj, fixed)
    
    effect, sd = subject.fit()
    keffect, ksd = [subject.fmristat_result(stat=stat)
                    for stat in ['effect', 'sd']]
    
    print _cor_result(effect, keffect)
    print _cor_result(sd, ksd)

##     from neuroimaging.ui.visualization.viewer import BoxViewer

##     u1 = BoxViewer(sd, colormap='spectral'); u1.M=2.2; u1.draw()
##     u2 = BoxViewer(ksd, colormap='spectral'); u2.M=2.2; u2.draw()

##     v1 = BoxViewer(effect, colormap='spectral'); v1.m = -1; v1.M=1.6; v1.draw()
##     v2 = BoxViewer(keffect, colormap='spectral'); v2.m = -1; v2.M=1.6; v2.draw()

##     import pylab
##     pylab.show()

