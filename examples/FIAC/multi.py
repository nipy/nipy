import os, shutil

import numpy as N

from neuroimaging.core.image.image import Image
from neuroimaging.algorithms.onesample import ImageOneSample

from fiac import FIACpath


def FIACmulti(contrast='overall', design='block', which='contrasts', clobber=False):

    outdir = 'http://kff.stanford.edu/FIAC/multi/%s/%s/%s' % (design, which, contrast)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    input = []
    
    for subj in range(16):
        subjdir = FIACpath('fixed/%s/%s/%s' % (design, which, contrast), run=-1, subj=subj)

        try:
            sdimg = Image('%s/sd.img' % subjdir)
            effimg = Image('%s/effect.img' % subjdir)
            input.append((effimg, sdimg))
        except:
            pass
    
    fitter = ImageOneSample(input, path=outdir, clobber=clobber, which='sdratio', all=True, use_scale=False)
    fitter = ImageOneSample(input, path=outdir, clobber=clobber, which='mean', all=True, use_scale=False)
    fitter.fit()

if __name__ == '__main__':

    import optparse

    parser = optparse.OptionParser()

    parser.add_option('', '--design', help='block or event?', dest='design', default='block')
    parser.add_option('', '--which', help='contrasts or delays', dest='which',
                      default='contrasts')
    parser.add_option('', '--contrast', help='overall, sentence, speaker or interaction?', dest='contrast', default='overall')
    parser.add_option('', '--clobber', help='clobber files?', dest='clobber', default=False, action='store_true')

    options, args = parser.parse_args()
        
    options = parser.values.__dict__

    FIACmulti(**options)
    print 'done', parser.values.design, parser.values.which, parser.values.contrast
