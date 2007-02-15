"""
Spatial normalization classes
"""
import os.path

import numpy as N

from mlabwrap import mlab

from neuroimaging.core.api import Image
from neuroimaging.sandbox.spm import utils as spmutils

class NormalizeSPM(object):
    ''' SPM normalization process

    For example:
    from neuroimaging.core.api import Image
    from neuroimaging.sandbox.spatial import normalization
    img = Image("some_image.nii")
    N = normalization.NormalizeSPM(reg=0.01)
    params = N.normalization_parameters(img)
    '''

    def __init__(self,
                 template = None,
                 template_weighting = '',
                 smosrc = 8,
                 smoref = 0,
                 regtype = 'mni',
                 cutoff = 30,
                 nits = 16,
                 reg = 0.1):
        spm_path = mlab.spm('dir')
        if template is None:
            template_path = os.path.join(spm_path, 'templates', 'T1.nii')
            template = mlab.spm_vol(template_path)
        self.template = template
        self.template_weighting = template_weighting
        self.smosrc = smosrc
        self.smoref = smoref
        self.regtype = regtype
        self.cutoff = cutoff
        self.nits = nits
        self.reg = reg

    def normalization_parameters(self, source_image, matname='', source_weighting=''):
        ''' Calculate normalization parameters '''
        if matname is None:
            matname = 'test_sn.mat'
        VF = spmutils.image_to_vol(source_image)
        flags = mlab.struct('smosrc', self.smosrc,
                            'smoref', self.smoref,
                            'regtype', self.regtype,
                            'cutoff', self.cutoff,
                            'nits', self.nits,
                            'reg', self.reg)
        params =  mlab.spm_normalise(self.template,
                                     VF,
                                     matname,
                                     self.template_weighting,
                                     source_weighting,
                                     flags)
        return params
                 
