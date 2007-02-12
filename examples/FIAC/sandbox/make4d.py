import sys, gc
import numpy as N
from neuroimaging.modalities.fmri import fMRIImage
from neuroimaging.core.api import Image
from neuroimaging.core.reference.mapping import Affine

def make4d(subj=int(sys.argv[1]), run=int(sys.argv[2])):
    dir = 'http://kff.stanford.edu/FIAC/fiac%d/fonc%d' % (subj, run)
    framefiles = ['%s/fiac%d_fonc%d_%04d.img' % (dir, subj, run, i) for i in range(5, 196)]

    print framefiles[0]
    frame = Image(framefiles[0], usematfile=False)

    # setup fMRI grid

    t = frame.grid.mapping.transform
    T = N.zeros((5,5))
    T[1:,1:] = t
    T[0:4,4] = 0.
    T[0,0] = 2.5

    fmrigrid = frame.grid.replicate(191)
    fmrigrid.mapping = Affine(
        fmrigrid.mapping.input_coords,
        fmrigrid.mapping.output_coords, T)

    _fmri = fMRIImage('%s/fiac%d_fonc%d.img' % (dir, subj, run), grid=fmrigrid, mode='w', clobber=True, usematfile=False)

    i = 0
    for framefile in framefiles:
        frame = Image(framefile, usematfile=False)
        _fmri.image.memmap[slice(i, i+1, 1)] = frame.readall()
        i += 1
    _fmri.image.memmap.sync()
    del(_fmri); gc.collect()
##     try:
##         os.remove('%s/fiac%d_fonc%d.img.gz' % (dir, subj, run))
##     except:
##         pass
##     os.system('gzip %s/fiac%d_fonc%d.img' % (dir, subj, run))
    
make4d()
