import os, shutil, gc, time, urllib
import numpy as N

from neuroimaging.image import Image
from neuroimaging.image.interpolation import ImageInterpolator
from neuroimaging.image.onesample import ImageOneSample
from neuroimaging.reference.mapping import Affine
from neuroimaging.reference.grid import SamplingGrid

from fiac import FIACprotocol, FIACblock, FIACevent, FIACpath

def FIACfixed(contrast, which='block', subj=3, base='contrasts', clobber=False):
    keep = []

    pdir = {'block':FIACblock,
            'event':FIACevent}

    outdir = FIACpath('fixed/%s/%s/%s' % (which, base, contrast), run=-1, subj=subj)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for run in range(1, 5):
        try:
            p = pdir[which](subj=subj, run=run)
            keep.append(run)
        except:
            pass

    var = 0.

    input = []

    for run in keep:
        
        matfile = FIACpath('fsl/example_func2standard.mat', subj=subj, run=run)
        if os.path.exists(matfile):

            sdfile = FIACpath('fsl/fmristat_run/%s/%s/sd.img' % (base, contrast), subj=subj, run=run)
            outfile = os.path.join(outdir, 'sd_%d.img' % run)
            FIACresample(sdfile, outfile, subj=subj, run=run)
            tic = time.time()
            sdimg = fix_origin(Image(outfile))

            efffile = FIACpath('fsl/fmristat_run/%s/%s/effect.img' % (base, contrast), subj=subj, run=run)
            outfile = os.path.join(outdir, 'effect_%d.hdr' % run)
            FIACresample(efffile, outfile, subj=subj, run=run)
            effimg = fix_origin(Image(outfile))
            input.append((effimg, sdimg))

    if input:
        fitter = ImageOneSample(input, path=outdir, clobber=clobber, all=True, use_scale=False, weight_type='sd')
        fitter.fit()

def fix_origin(img):
    
    atlas = Image('/home/stow/fsl/etc/standard/avg152T1_brain.img')
    img.image.origin = atlas.image.origin
    img.image.pixdim = atlas.image.pixdim

    hdrfile = file(img.image.hdrfilename(), 'w')
    img.image.writeheader(hdrfile)
    hdrfile.close()
    return Image(img.image.filename)

def FIACresample(infile, outfile, subj=3, run=3, **other):

    inimage = Image(infile,
                                       ignore_origin=True,
                                       abs_pixdim=True)

    fslmatstr = urllib.urlopen('file://%s' % FIACpath('fsl/example_func2standard.mat',
                                                       subj=subj, run=run)).read().split()
    fslmat =  N.array(map(float, fslmatstr))
    fslmat.shape = (4,4)

    M = [[0,0,1,0],
         [0,1,0,0],
         [1,0,0,0],
         [0,0,0,1]]

    fslmat = N.dot(M, N.dot(fslmat, M))

    standard = Image('/home/analysis/FIAC/avg152T1_brain.img',
                                        ignore_origin=True,
                                        abs_pixdim=True)

    input_coords = inimage.grid.mapping.output_coords
    output_coords = standard.grid.mapping.output_coords
    fworld2sworld = Affine(input_coords,
                                             output_coords,
                                             fslmat)
    svoxel2fworld = fworld2sworld.inverse() * standard.grid.mapping

    output_grid = SamplingGrid(mapping=svoxel2fworld, shape=standard.grid.shape)

    interp = ImageInterpolator(inimage)
    interp_data = interp(output_grid.range())

    new = Image(interp_data, grid=standard.grid)
    new.tofile(outfile, clobber=True)

    del(interp); del(interp_data); del(new) ; gc.collect()


def FIACrun(subj=3, clobber=True):
    for base in ['delays', 'contrasts']:
        for which in ['block', 'event']:
            for contrast in ['overall', 'speaker', 'sentence', 'interaction']:
                FIACfixed(contrast, base=base, which=which, subj=subj, clobber=True)
                print base, which, contrast, 'done'

if __name__ == '__main__':

    import sys
    subj = int(sys.argv[1])

    FIACrun(subj=subj)
