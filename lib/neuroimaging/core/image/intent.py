"""
TODO
"""
__docformat__ = 'restructuredtext'

import new

from neuroimaging.data_io.formats import nifti1
from neuroimaging import traits
from neuroimaging.core.api import Image
from neuroimaging.core.reference import axis
from neuroimaging.core.reference.grid import SamplingGrid
from neuroimaging.data_io import DataSource

class IntentError(Exception):
    """
    Errors raised in intents.
    """

intent_trait = traits.Trait(nifti1.NIFTI_INTENT_NONE, desc='Allowable intent codes -- currently only NIFTI1 codes.')

class Intent(traits.HasTraits):
    """
    TODO
    """

    parameters = traits.ReadOnly(desc='Parameters associated to intent code.')
    intent_code = intent_trait
    intent_name = traits.Str(desc='Intent name.')
    format = nifti1.Nifti1

    def __init__(self, intent_code, name, *parameters):
        """
        :Parameters:
            intent_code : TODO
                TODO
            name : TODO
                TODO
            parameters : TODO
                TODO
        """
        self.intent_code = intent_code
        self.name = name
        self.parameters = list(parameters)

    def __call__(self, image, **parameter_values):
        """
        :Parameters:
            image : TODO
                TODO
            parameter_values : TODO
                TODO

        :Returns: `Image`
        """
        if not isinstance(image, Image):
            image = Image(image, format=self.format)
        self._set_intent(image, **parameter_values)
        return image

    def validate(self, image):
        """
        :Parameters:
            image : TODO
                TODO

        :Returns: ``None``
        """
        pass
    
    def _set_intent(self, image, **parameter_values):
        self.validate(image)
        self._set_intent_code(image)
        self._set_intent_parameters(image, **parameter_values)

    def _set_intent_code(self, image):
        if image.trait('intent_code') is not None:
            for par in image.intent_parameters:
                image.remove_trait(par)
            image.remove_trait('intent_code')
        if not isinstance(image._source, nifti1.Nifti1):
            image.add_trait('intent_code', intent_trait)
            image.intent_code = self.intent_code
        else:
            trait = traits.Delegate('_source', 'intent_code', modify=True)
            image.add_trait('intent_code', trait)
            image.intent_code = self.intent_code 

    def _set_intent_parameters(self, image, **parameter_values):
        for i in range(len(self.parameters)):
            par = self.parameters[i]
            if par not in parameter_values.keys():
                raise IntentError, 'parameters %s must be specified' % `self.parameters`
            if not isinstance(image._source, nifti1.Nifti1):
                trait = traits.Float(parameter_values[par])
                image.add_trait(par, parameter_values[par])
            else:
                trait = traits.Delegate('image', 'intent_p%d' % (i+1,), modify=True)
                image.add_trait(par, trait)
                setattr(image._source, 'intent_p%d' % (i+1,), parameter_values[par])                
            image.intent_parameters = self.parameters

    def create(self, filename, datasource=DataSource(), grid=None,
               clobber=False, **keywords):
        """
        :Parameters:
            filename : TODO
                TODO
            datasource : `DataSource`
                TODO
            grid : TODO
                TODO
            clobber : ``bool``
                TODO
            keywords : ``dict``
                TODO

        :Returns; `Image`
        """

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
    """
    TODO
    """

    voxelwise = traits.false
    
    def _set_intent_parameters(self, image, voxelwise=False,
                               **parameter_values):

        if not voxelwise:
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
               clobber=False, voxelwise=False,
               **keywords):
        """
        :Parameters:
            filename : TODO
                TODO
            datasource : `DataSource`
                TODO
            grid : TODO
                TODO
            clobber : ``bool``
                TODO
            voxelwise : ``bool``
                TODO
            keywords : ``dict``
                TODO

        :Returns: `Image`
        """
        parameter_values = {}
                
        npar = len(self.parameters)
        outgrid = grid
        if voxelwise and self.parameters:
            self.grid = grid
            if grid.ndim == 5:
                if grid.shape[0] != npar + 1:
                    raise IntentError, '5th dimension of grid should be the number of parameters + 1'

            for i in range(5 - grid.ndim - 1):
                _i = grid.ndim + i
                outgrid = outgrid.replicate(1, concataxis=axis.valid[_i])
            outgrid = outgrid.replicate(npar+1)
        elif voxelwise and not self.parameters:
            raise IntentError, 'intent type "%s" does not support voxelwise mode' % self.name
        else:
            for par in self.parameters:
                parameter_values[par] = keywords[par]

        imgsource = self.format(filename=filename,
                                datasource=datasource,
                                grid=outgrid,
                                mode='w',
                                clobber=clobber)

        image = Image(imgsource)
        return self(image, voxelwise=voxelwise, **parameter_values)

def _replicate_create(n, format, filename, datasource=DataSource(), grid=None,
                      clobber=False):
    outgrid = grid

    for i in range(5 - grid.ndim - 1):
        _i = grid.ndim + i
        outgrid = outgrid.replicate(1, concataxis=axis.valid[_i])
        outgrid = outgrid.replicate(n)

    imgsource = format(filename=filename,
                       datasource=datasource,
                       grid=outgrid,
                       mode='w',
                       clobber=clobber)

    image = Image(imgsource)

    return image

Vector = Intent(nifti1.NIFTI_INTENT_VECTOR, 'vector')
def vector_validate(self, image):
    """
    Ensure that image is interpretable as an image
    of vectors.

    :Parameters:
        image : TODO
            TODO

    :Returns: ``None``
    """

    if image.grid.ndim != 5:
        raise IntentError, 'vector needs 5 dimensions'
    self.n = image.grid.shape[0]

def vector_create(self, filename, datasource=DataSource(), grid=None,
                  clobber=False, n=1):
    """
    Create a vector image with n coordinates
    based on a given grid.

    :Parameters:
        filename : TODO
            TODO
        datasource : `DataSource`
            TODO
        grid : TODO
            TODO
        clobber : ``bool``
            TODO
        n : ``int``
            TODO

    :Returns: `Image`
    
    """
    self.grid = grid
    image = _replicate_create(n, self.format, filename, datasource=datasource,
                              grid=grid, clobber=clobber)
    return self(image)


Vector.validate = new.instancemethod(vector_validate, Vector, Intent)
Vector.create = new.instancemethod(vector_create, Vector, Intent)

SymMatrix = Intent(nifti1.NIFTI_INTENT_SYMMATRIX, 'symmatrix', 'n')
def symmatrix_validate(self, image):
    """
    Ensure that image is interpretable as an image
    of symmetric matrices of shape (self.n,)*2

    :Parameters:
        image : TODO
            TODO

    :Returns: ``None``

    :Raises IntentError: TODO
    """
    if image.grid.ndim != 5:
        raise IntentError, 'symmetric matrix needs 5 dimensions'
    if image.grid.shape[0] != self.n * (self.n + 1.) / 2:
        raise IntentError, 'first dimension should be %d if n is %d' % (self.n*(self.n+1)/2, self.n)

def symmatrix_create(self, filename, datasource=DataSource(), grid=None,
                  clobber=False, n=1):
    """
    Create an image of symmetric matrices of shape (self.n,)*2 with
    coordinates based on a given grid.

    :Parameters:
        filename : TODO
            TODO
        datasource : `DataSource`
            TODO
        grid : TODO
            TODO
        clobber : ``bool``
            TODO
        n : ``int``
            TODO

    :Returns: `Image`
    """
    self.grid = grid
    self.n = n
    n = n * (n + 1) / 2
    image = _replicate_create(n, self.format, filename, datasource=datasource,
                              grid=grid, clobber=clobber)
    return self(image, n=n)

SymMatrix.validate = new.instancemethod(symmatrix_validate, SymMatrix, Intent)
SymMatrix.create = new.instancemethod(symmatrix_create, SymMatrix, Intent)

DispVector = Intent(nifti1.NIFTI_INTENT_DISPVECT, 'dispvect')

def dispvector_validate(self, image):
    """
    Ensure that image is interpretable as an
    image of 3d displacement vectors.

    :Parameters:
        image : `Image`
            TODO

    :Returns: ``None``

    :Raises IntentError: TODO
    """
    if image.grid.ndim != 5:
        raise IntentError, 'displacement vector needs 5 dimensions'
    if image.grid.shape[0] != 3:
        raise IntentError, 'first dimension should be 3'

def dispvector_create(self, filename, datasource=DataSource(), grid=None,
                      clobber=False):
    """
    Create a 3d displacement vector image with coordinates
    based on a given grid.

    :Parameters:
        filename : TODO
            TODO
        datasource : `DataSource`
            TODO
        grid : TODO
            TODO
        clobber : ``bool``
            TODO

    :Returns: `Image`
    """
    self.grid = grid
    image = _replicate_create(3, self.format, filename, datasource=datasource,
                              grid=grid, clobber=clobber)
    return self(image)

DispVector.validate = new.instancemethod(dispvector_validate, DispVector, Intent)
DispVector.create = new.instancemethod(dispvector_create, DispVector, Intent)


Matrix = Intent(nifti1.NIFTI_INTENT_GENMATRIX, 'matrix', 'm', 'n')

def matrix_validate(self, image):
    """
    Ensure that image is interpretable as an image
    of matrices of shape (self.m, self.n)

    :Parameters:
        image : `Image`
            TODO

    :Returns: ``None``

    :Raises IntentError: TODO
    """

    if image.grid.ndim != 5:
        raise IntentError, 'matrix needs 5 dimensions'
    if image.grid.shape[0] != self.m * self.n:
        raise IntentError, 'first dimension should be %d' % self.m*self.n

def matrix_create(self, filename, datasource=DataSource(), grid=None,
                  clobber=False, m=1, n=1):
    """
    Create an image of matrices of shape (self.m, self.n) with coordinates
    based on a given grid.

    :Parameters:
        filename : TODO
            TODO
        datasource : `DataSource`
            TODO
        grid : TODO
            TODO
        clobber : ``bool``
            TODO
        m : ``int``
            TODO
        n : ``int``
            TODO

    :Returns: `Image`
    """
    self.m = m; self.n = n
    self.grid = grid
    n = self.n * self.m
    image = _replicate_create(n, self.format, filename, datasource=datasource,
                              grid=grid, clobber=clobber)
    return self(image, n=n, m=m)

Matrix.validate = new.instancemethod(matrix_validate, Matrix, Intent)
Matrix.create = new.instancemethod(matrix_create, Matrix, Intent)

Quaternion = Intent(nifti1.NIFTI_INTENT_QUATERNION, 'quaternion')

def quaternion_validate(self, image):
    """
    Ensure that image is interpretable as an image
    of quaternions.

    :Parameters:
        image : `Image`
            TODO

    :Returns: ``None``

    :Raises IntentError: TODO    
    """
    
    if image.grid.ndim != 5:
        raise IntentError, 'triangle needs 5 dimensions'
    if image.grid.shape[0] != 3:
        raise IntentError, 'first dimension should be 4 for quaternion'

def quaternion_create(self, filename, datasource=DataSource(), grid=None,
                      clobber=False):
    """
    Create an image of quaternions with coordinates
    based on a given grid.

    :Parameters:
        filename : TODO
            TODO
        datasource : `DataSource`
            TODO
        grid : TODO
            TODO
        clobber : ``bool``
            TODO

    :Returns: `Image`
    """
    self.grid = grid
    n = 4
    image = _replicate_create(n, self.format, filename, datasource=datasource,
                              grid=grid, clobber=clobber)
    return self(image)

Quaternion.validate = new.instancemethod(quaternion_validate, Quaternion, Intent)
Quaternion.create = new.instancemethod(quaternion_create, Quaternion, Intent)


PointSet = Intent(nifti1.NIFTI_INTENT_POINTSET, 'pointset')
def pointset_validate(self, image):
    """
    :Paramters:
        image : `Image`
            TODO

    :Returns: ``None``

    :Raises IntentError: TODO
    """
    if image.grid.ndim != 5:
        raise IntentError, 'pointset needs 5 dimensions'
    if image.grid.shape[1:4] != (1,1,1):
        raise IntentError, 'shape[1:4] should be (1,1,1) for pointset'

def pointset_create(self, filename, datasource=DataSource(),
                    clobber=False, n=1, dim=3):
    """
    Create a pointset image with n points of dimension dim.

    :Parameters:
        filename : TODO
            TODO
        datasource : `DataSource`
            TODO
        clobber : ``bool``
            TODO
        n : ``int``
            TODO
        dim : ``int``
            TODO
             
    """
    grid = SamplingGrid.from_start_step(names=['concat'],
                                        shape=[n],
                                        step=[1],
                                        start=[0])
    self.grid = grid
    image = _replicate_create(dim, self.format, filename,
                              datasource=datasource,
                              grid=grid, clobber=clobber)
    return self(image)

PointSet.validate = new.instancemethod(pointset_validate, PointSet, Intent)
PointSet.create = new.instancemethod(pointset_create, PointSet, Intent)

Triangle = Intent(nifti1.NIFTI_INTENT_TRIANGLE, 'triangle')
def triangle_validate(self, image):
    """
    :Parameters:
        image : `Image`
            TODO

    :Returns: ``None``

    :Raises IntentError: TODO
    """
    if image.grid.ndim != 5:
        raise IntentError, 'triangle needs 5 dimensions'
    if image.grid.shape[0] != 3:
        raise IntentError, 'first dimension should be 3 for triangle'
    if image.grid.shape[1:4] != (1,1,1):
        raise IntentError, 'shape[1:4] should be (1,1,1) for triangle'

def triangle_create(self, filename, datasource=DataSource(),
                    clobber=False, n=1):
    """
    Create a triangles image with n triangles.

    :Parameters:
        filename : TODO
            TODO
        datasource : `DataSource`
            TODO
        clobber : ``bool``
            TODO
        n : ``int``
            TODO
    """
    grid = SamplingGrid.from_start_step(names=['concat'],
                                        shape=[n],
                                        step=[1],
                                        start=[0])
    self.grid = grid
    image = _replicate_create(3, self.format, filename, datasource=datasource,
                              grid=grid, clobber=clobber)
    return self(image)

Triangle.validate = new.instancemethod(triangle_validate, Triangle, Intent)
Triangle.create = new.instancemethod(triangle_create, Triangle, Intent)

# Statistical intents

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
Poisson = StatIntent(nifti1.NIFTI_INTENT_POISSON, 'poisson', 'mean')
PVal = StatIntent(nifti1.NIFTI_INTENT_PVAL, 'pval')
ExtVal = StatIntent(nifti1.NIFTI_INTENT_EXTVAL, 'extval', 'location', 'scale')
FStat = StatIntent(nifti1.NIFTI_INTENT_FTEST, 'fstat', 'df_num', 'df_denom')
NoncentralFStat = StatIntent(nifti1.NIFTI_INTENT_FTEST_NONC,
                             'fstat_nonc', 'df_num', 'df_denom', 'ncp')
TStat = StatIntent(nifti1.NIFTI_INTENT_TTEST, 'tstat', 'df')
Gamma = StatIntent(nifti1.NIFTI_INTENT_GAMMA, 'gamma' 'shape', 'scale')
NoncentralTStat = StatIntent(nifti1.NIFTI_INTENT_TTEST_NONC, 'tstat_nonc', 'df', 'ncp')
Uniform = StatIntent(nifti1.NIFTI_INTENT_UNIFORM, 'uniform', 'lower', 'upper')
InvGauss = StatIntent(nifti1.NIFTI_INTENT_INVGAUSS, 'invgauss', 'mu', 'scale')
Weibull = StatIntent(nifti1.NIFTI_INTENT_WEIBULL, 'weibull', 'location', 'scale', 'power')
Laplace = StatIntent(nifti1.NIFTI_INTENT_LAPLACE, 'laplace', 'location', 'scale')
ZStat = StatIntent(nifti1.NIFTI_INTENT_ZSCORE, 'zstat')

# Miscellaneous intents

DimLess = Intent(nifti1.NIFTI_INTENT_DIMLESS, 'dimless')
Label = Intent(nifti1.NIFTI_INTENT_LABEL, 'label')
Estimate = Intent(nifti1.NIFTI_INTENT_ESTIMATE, 'estimate')
NeuroName = Intent(nifti1.NIFTI_INTENT_NEURONAME, 'neuroname')


if __name__ == '__main__':

    zimage = Image('http://nifti.nimh.nih.gov/nifti-1/data/zstat1.nii.gz')
    a = ZStat.create('out.nii', grid=zimage.grid, clobber=True)
    print a.shape

    ## a = ZStat.create('out.nii', grid=zimage.grid, clobber=True, voxelwise=True)
    ## print a.shape

    a = TStat.create('out.nii', df=5, grid=zimage.grid, clobber=True)
    print a.shape, a.df, a._source.intent_p1

    a = TStat.create('out.nii', grid=zimage.grid, clobber=True, voxelwise=True)
    print a.shape, a.df.shape, a.tstat.shape
