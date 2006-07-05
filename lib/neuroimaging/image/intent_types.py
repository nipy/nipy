
from neuroimaging.image.formats import nifti1
from neuroimaging import traits
from neuroimaging.image import Image

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

intentid_trait = traits.Trait(parameters.keys())

def set_intent(image, intent_code, **params):
    """
    Set the intent to an image following NIFTI-1's rules.
    Should be able to add intents to all Image instances,
    only validation follows NIFTI-1 rules.

    If the Image instance is a NIFTI-1 file, then this
    should write out the NIFTI-1 file correctly.

    If the image is 5-dimensional and the intent is
    a statistical code, then the intent attributes
    are not set as they are slices of the image data itself.

    For example, an image with intent 'ttest' with 5 dimensions must
    have the 5th dimension==2 and the second slice is the voxel-dependent
    degrees of freedom. See help(neuroimaging.image.formats.nifti1) for details.

    """

    try:
        old = image.trait('intent_code')
        for par in parameters[image.intent_code]:
            image.remove_trait(par)
    except:
        image.add_trait('intent_code', intentid_trait)
        image.intent_code = intent_code

    if isinstance(image.image, nifti1.NIFTI1):
        image.remove_trait('intent_code')
        trait = traits.Delegate('image', 'intent_code', modify=True)
        image.add_trait('intent_code', trait)
        image.intent_code = intent_code 

    dim = image.grid.shape[::-1]
    dim = (len(dim),) + dim

    if intent_code > nifti1.NIFTI_LAST_STATCODE or \
           (intent_code >= nifti1.NIFTI_FIRST_STATCODE and dim[0] < 5):
        for i in range(len(parameters[intent_code])):
            par = parameters[intent_code][i]
            if par not in params.keys():
                params[par] = 0.

            if not isinstance(image.image, nifti1.NIFTI1):
                trait = traits.Float(params[par])
            else:
                trait = traits.Delegate('image', 'intent_p%d' % (i+1,), modify=True)
                setattr(image.image, 'intent_p%d' % (i+1,), params[par])
            image.add_trait(par, trait)

    if intent_code == nifti1.NIFTI_INTENT_GENMATRIX:
        if dim[0] != 5:
            raise IntentError, 'image must be 5d for "matrix" intent'
        if not params.has_key('m') or not params.has_key('n'):
            raise IntentError, '"m" and "n" must be specified for "matrix" intent'
        m = params['m']; n = params['n']
        if m*n != dim[1]:
            raise IntentError, 'size of matrix does not agree with image shape'
            
    if intent_code == nifti1.NIFTI_INTENT_SYMMATRIX:
        if dim[0] != 5:
            raise IntentError, 'image must be 5d for "symmatrix" intent'
        if not params.has_key('n'):
            raise IntentError, '"n" must be specified for "symmatrix" intent'
        n = params['n']
        if n(n+1)/2 != dim[1]:
            raise IntentError, 'size of symmatrix does not agree with image shape'
            
class IntentModifier(traits.HasTraits):
    intent_code = intentid_trait

    def __init__(self, intent_code):
        self.intent_code = intent_code
        
    def __call__(self, image, **keywords):
        set_intent(image, self.intent_code, **keywords)
        return image

Log10Pval = IntentModifier(nifti1.NIFTI_INTENT_LOG10PVAL)
Beta = IntentModifier(nifti1.NIFTI_INTENT_BETA)
Logistic = IntentModifier(nifti1.NIFTI_INTENT_LOGISTIC)
Binom = IntentModifier(nifti1.NIFTI_INTENT_BINOM)
LogPval = IntentModifier(nifti1.NIFTI_INTENT_LOGPVAL)
Chi = IntentModifier(nifti1.NIFTI_INTENT_CHI)
NeuroName = IntentModifier(nifti1.NIFTI_INTENT_NEURONAME)
Chisq = IntentModifier(nifti1.NIFTI_INTENT_CHISQ)
NoneIntent = IntentModifier(nifti1.NIFTI_INTENT_NONE)
NoncChisq = IntentModifier(nifti1.NIFTI_INTENT_CHISQ_NONC)
Normal = IntentModifier(nifti1.NIFTI_INTENT_NORMAL)
Correl = IntentModifier(nifti1.NIFTI_INTENT_CORREL)
PointSet = IntentModifier(nifti1.NIFTI_INTENT_POINTSET)
DimLess = IntentModifier(nifti1.NIFTI_INTENT_DIMLESS)
Poisson = IntentModifier(nifti1.NIFTI_INTENT_POISSON)
DispVect = IntentModifier(nifti1.NIFTI_INTENT_DISPVECT)
PVal = IntentModifier(nifti1.NIFTI_INTENT_PVAL)
Estimate = IntentModifier(nifti1.NIFTI_INTENT_ESTIMATE)
Quaternion = IntentModifier(nifti1.NIFTI_INTENT_QUATERNION)
ExtVal = IntentModifier(nifti1.NIFTI_INTENT_EXTVAL)
SymMatrix = IntentModifier(nifti1.NIFTI_INTENT_SYMMATRIX)
FTest = IntentModifier(nifti1.NIFTI_INTENT_FTEST)
Triangle = IntentModifier(nifti1.NIFTI_INTENT_TRIANGLE)
NoncFTest = IntentModifier(nifti1.NIFTI_INTENT_FTEST_NONC)
TTest = IntentModifier(nifti1.NIFTI_INTENT_TTEST)
Gamma = IntentModifier(nifti1.NIFTI_INTENT_GAMMA)
NoncTTest = IntentModifier(nifti1.NIFTI_INTENT_TTEST_NONC)
Matrix = IntentModifier(nifti1.NIFTI_INTENT_GENMATRIX)
Uniform = IntentModifier(nifti1.NIFTI_INTENT_UNIFORM)
InvGauss = IntentModifier(nifti1.NIFTI_INTENT_INVGAUSS)
Vector = IntentModifier(nifti1.NIFTI_INTENT_VECTOR)
Label = IntentModifier(nifti1.NIFTI_INTENT_LABEL)
Weibull = IntentModifier(nifti1.NIFTI_INTENT_WEIBULL)
Laplace = IntentModifier(nifti1.NIFTI_INTENT_LAPLACE)
ZScore = IntentModifier(nifti1.NIFTI_INTENT_ZSCORE)

