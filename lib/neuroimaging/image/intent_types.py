
from neuroimaging.image.formats import nifti1
from neuroimaging import traits
from neuroimaging.image import Image
from neuroimaging.reference import axis
from neuroimaging.data import DataSource

class IntentError(Exception):
    """
    Errors raised in intents.
    """

parameters = {nifti1.NIFTI_INTENT_LOG10PVAL:(),
              nifti1.NIFTI_INTENT_BETA:('a','b'),
              nifti1.NIFTI_INTENT_LOGISTIC:('location', 'scale'),
              nifti1.NIFTI_INTENT_BINOM:('n','p'),
              nifti1.NIFTI_INTENT_LOGPVAL:(),
              nifti1.NIFTI_INTENT_CHI:('df',),
              nifti1.NIFTI_INTENT_NEURONAME:(),
              nifti1.NIFTI_INTENT_CHISQ:('df',),
              nifti1.NIFTI_INTENT_NONE:(),
              nifti1.NIFTI_INTENT_CHISQ_NONC:('df', 'ncp'),
              nifti1.NIFTI_INTENT_NORMAL:('mean', 'sd'),
              nifti1.NIFTI_INTENT_CORREL:('df',),
              nifti1.NIFTI_INTENT_POINTSET:(),
              nifti1.NIFTI_INTENT_DIMLESS:(),
              nifti1.NIFTI_INTENT_POISSON:('mean'),
              nifti1.NIFTI_INTENT_DISPVECT:(),
              nifti1.NIFTI_INTENT_PVAL:(),
              nifti1.NIFTI_INTENT_ESTIMATE:(),
              nifti1.NIFTI_INTENT_QUATERNION:(),
              nifti1.NIFTI_INTENT_EXTVAL:('location', 'scale'),
              nifti1.NIFTI_INTENT_SYMMATRIX:('n'),
              nifti1.NIFTI_INTENT_FTEST:('df_num', 'df_denom'),
              nifti1.NIFTI_INTENT_TRIANGLE:(),
              nifti1.NIFTI_INTENT_FTEST_NONC:('df_num', 'df_denom', 'ncp'),
              nifti1.NIFTI_INTENT_TTEST:('df',),
              nifti1.NIFTI_INTENT_GAMMA:('shape', 'scale'),
              nifti1.NIFTI_INTENT_TTEST_NONC:('df', 'ncp'),
              nifti1.NIFTI_INTENT_GENMATRIX:('m', 'n'),
              nifti1.NIFTI_INTENT_UNIFORM:('lower', 'upper'),
              nifti1.NIFTI_INTENT_INVGAUSS:('mu', 'scale'), #scale=lambda
              nifti1.NIFTI_INTENT_VECTOR:(),
              nifti1.NIFTI_INTENT_LABEL:(),
              nifti1.NIFTI_INTENT_WEIBULL:('location', 'scale', 'power'),
              nifti1.NIFTI_INTENT_LAPLACE:('location', 'scale'),
              nifti1.NIFTI_INTENT_ZSCORE:(),
              }

            
intent_trait = traits.Trait(nifti1.NIFTI_INTENT, desc='Allowable intent codes -- currently only NIFTI1 codes.')

class Intent(traits.HasTraits):
    parameters = traits.ReadOnly(desc='Parameters associated to intent code.')
    intent_code = intent_trait
    intent_name = traits.Str(desc='Intent name.')
    format = nifti1.NIFTI1

    def __init__(self, intent_code, name, *parameters):
        self.intent_code = intent_code
        self.name = name
        self.parameters = list(parameters)

    def __call__(self, image, **parameter_values):
        self._set_intent(image, **parameter_values)
        return image

    def _set_intent(self, image, **parameter_values):
        self._set_intent_code(image)
        self._set_intent_parameters(image, **parameter_values)

    def _set_intent_code(self, image):
        if image.trait('intent_code') is not None:
            for par in image.intent_parameters:
                image.remove_trait(par)
            image.remove_trait('intent_code')
        if not isinstance(image, nifti1.NIFTI1):
            image.add_trait('intent_code', intent_trait)
            image.intent_code = self.intent_code
        else:
            trait = traits.Delegate('image', 'intent_code', modify=True)
            image.add_trait('intent_code', trait)
            image.intent_code = self.intent_code 

    def _set_intent_parameters(self, image, **parameter_values):
        for i in range(len(self.parameters)):
            par = self.parameters[i]
            if par not in parameter_values.keys():
                raise IntentError, 'parameters %s must be specified' % `self.parameters`
            if not isinstance(image.source, nifti1.NIFTI1):
                trait = traits.Float(parameter_values[par])
                image.add_trait(par, parameter_values[par])
            else:
                trait = traits.Delegate('image', 'intent_p%d' % (i+1,), modify=True)
                image.add_trait(par, trait)
                setattr(image.source, 'intent_p%d' % (i+1,), parameter_values[par])                
            image.intent_parameters = self.parameters

    def create(self, filename, datasource=DataSource(), grid=None,
               clobber=False, **keywords):

        parameter_values = {}
        for par in self.parameters:
            parameter_values[par] = keywords[par]
                
        
        imgsource = self.format(filename=filename,
                                datasource=datasource,
                                grid=grid,
                                mode='w',
                                clobber=clobber)
        image = Image(imgsource)
        return self(image, **parameter_values)

class StatIntent(Intent):

    voxel_mode = traits.false
    
    def _set_intent_parameters(self, image, voxel_mode=False,
                               **parameter_values):

        if not voxel_mode:
            Intent._set_intent_parameters(self, image, **parameter_values)
        else:
            image_trait = traits.Instance(Image)
            _slice = [slice(0, 1)] + [slice(0,1)]*(5-self.grid.ndim-1)
            setattr(image, self.name, Image(image[_slice], grid=self.grid))
            for i in range(len(self.parameters)):
                par = self.parameters[i]
                image.add_trait(par, image_trait)
                _slice = [slice(i+1, i+2)] + [slice(0,1)]*(5-self.grid.ndim-1)
                setattr(image, par, Image(image[_slice], grid=self.grid))

    def create(self, filename, datasource=DataSource(), grid=None,
               clobber=False, voxel_mode=False,
               **keywords):

        parameter_values = {}
                
        npar = len(self.parameters)
        outgrid = grid
        if voxel_mode and self.parameters:
            self.grid = grid
            if grid.ndim == 5:
                if grid.shape[0] != npar + 1:
                    raise IntentError, '5th dimension of grid should be the number of parameters + 1'

            for i in range(5 - grid.ndim - 1):
                _i = grid.ndim + i
                outgrid = outgrid.replicate(1, concataxis=axis.valid[_i])
            outgrid = outgrid.replicate(npar+1)
        elif voxel_mode and not self.parameters:
            raise IntentError, 'intent type "%s" does not support voxel_mode' % self.name
        else:
            for par in self.parameters:
                parameter_values[par] = keywords[par]

        imgsource = self.format(filename=filename,
                                datasource=datasource,
                                grid=outgrid,
                                mode='w',
                                clobber=clobber)

        image = Image(imgsource)
        return self(image, voxel_mode=voxel_mode, **parameter_values)

Vector = Intent(nifti1.NIFTI_INTENT_VECTOR, 'vector')
Label = Intent(nifti1.NIFTI_INTENT_LABEL, 'label')
SymMatrix = Intent(nifti1.NIFTI_INTENT_SYMMATRIX, 'symmatrix', 'n')
DispVect = Intent(nifti1.NIFTI_INTENT_DISPVECT, 'dispvect')
Estimate = Intent(nifti1.NIFTI_INTENT_ESTIMATE, 'estimate')
Quaternion = Intent(nifti1.NIFTI_INTENT_QUATERNION, 'quaternion')
NeuroName = Intent(nifti1.NIFTI_INTENT_NEURONAME, 'neuroname')

Log10Pval = StatIntent(nifti1.NIFTI_INTENT_LOG10PVAL, 'log10pval')
Beta = StatIntent(nifti1.NIFTI_INTENT_BETA, 'beta', 'a', 'b')
Logistic = StatIntent(nifti1.NIFTI_INTENT_LOGISTIC, 'logistic', 'location', 'scale')
Binom = StatIntent(nifti1.NIFTI_INTENT_BINOM, 'binom', 'n', 'p')
LogPval = StatIntent(nifti1.NIFTI_INTENT_LOGPVAL, 'logpval')
Chi = StatIntent(nifti1.NIFTI_INTENT_CHI, 'chi', 'df')
Chisq = StatIntent(nifti1.NIFTI_INTENT_CHISQ, 'chisq', 'df')
NoneIntent = StatIntent(nifti1.NIFTI_INTENT_NONE, 'none')
NoncentralChisq = StatIntent(nifti1.NIFTI_INTENT_CHISQ_NONC, 'chisq_nonc', 'df', 'ncp')
Normal = StatIntent(nifti1.NIFTI_INTENT_NORMAL, 'normal', 'mean', 'sd')
Correl = StatIntent(nifti1.NIFTI_INTENT_CORREL, 'correl', 'df')
PointSet = Intent(nifti1.NIFTI_INTENT_POINTSET, 'pointset')
DimLess = Intent(nifti1.NIFTI_INTENT_DIMLESS, 'dimless')
Poisson = StatIntent(nifti1.NIFTI_INTENT_POISSON, 'poisson', 'mean')
PVal = StatIntent(nifti1.NIFTI_INTENT_PVAL, 'pval')
ExtVal = StatIntent(nifti1.NIFTI_INTENT_EXTVAL, 'extval', 'location', 'scale')
FStat = StatIntent(nifti1.NIFTI_INTENT_FTEST, 'fstat', 'df_num', 'df_denom')
Triangle = StatIntent(nifti1.NIFTI_INTENT_TRIANGLE, 'triangle')
NoncentralFStat = StatIntent(nifti1.NIFTI_INTENT_FTEST_NONC,
                             'fstat_nonc', 'df_num', 'df_denom', 'ncp')
TStat = StatIntent(nifti1.NIFTI_INTENT_TTEST, 'tstat', 'df')
Gamma = StatIntent(nifti1.NIFTI_INTENT_GAMMA, 'gamma' 'shape', 'scale')
NoncentralTStat = StatIntent(nifti1.NIFTI_INTENT_TTEST_NONC, 'tstat_nonc', 'df', 'ncp')
Matrix = StatIntent(nifti1.NIFTI_INTENT_GENMATRIX, 'matrix', 'm', 'n')
Uniform = StatIntent(nifti1.NIFTI_INTENT_UNIFORM, 'uniform', 'lower', 'upper')
InvGauss = StatIntent(nifti1.NIFTI_INTENT_INVGAUSS, 'invgauss', 'mu', 'scale')
Weibull = StatIntent(nifti1.NIFTI_INTENT_WEIBULL, 'weibull', 'location', 'scale', 'power')
Laplace = StatIntent(nifti1.NIFTI_INTENT_LAPLACE, 'laplace', 'location', 'scale')
ZStat = StatIntent(nifti1.NIFTI_INTENT_ZSCORE, 'zstat')

if __name__ == '__main__':

    zimage = Image('http://nifti.nimh.nih.gov/nifti-1/data/zstat1.nii.gz')
    a = ZStat.create('out.nii', grid=zimage.grid, clobber=True)
    print a.shape

    ## a = ZStat.create('out.nii', grid=zimage.grid, clobber=True, voxel_mode=True)
    ## print a.shape

    a = TStat.create('out.nii', df=5, grid=zimage.grid, clobber=True)
    print a.shape, a.df, a.source.intent_p1

    a = TStat.create('out.nii', grid=zimage.grid, clobber=True, voxel_mode=True)
    print a.shape, a.df.shape, a.tstat.shape
