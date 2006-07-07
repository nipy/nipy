import numpy as N
import numpy.linalg as L

from enthought import traits
from neuroimaging.image import Image
from neuroimaging.statistics.utils import recipr


class RFXMean(traits.HasTraits):
    clobber = traits.false
    max_varatio = traits.Float(10.)
    df_limit = traits.Float(4.)
    df_target = traits.Float(100.)
    niter = traits.Int(10)
    verbose = traits.false
    fixed = traits.false
    mask = traits.Any()
    fwhm_varatio = traits.Any()

    """
    Perform a RFX analysis for the mean -- i.e. the fixed effect design matrix is a column of 1's.

    Input is a sequence: if the entries are 
    """

    def __init__(self, input, df=None, fwhm=None, tol=1.0e+05, outputs=None, **keywords):

        traits.HasTraits.__init__(self, **keywords)

        self.nsubject = len(input_files)

        self.X = N.ones((self.nsubject, 1), N.float64)
        self.Xsq = N.power(N.transpose(self.design_matrix), 2)
        self.pinvX = L.pinv(self.X)

        # Prepare files for reading in

        self.input = []
        if sd_files:
            if len(sd_files) != len(input_files):
                raise ValueError, 'expecting the same number of SD files as input files in MultiStat'
            self.sd = []

        for subject in range(self.nsubject):
            self.input.append(iter(Image(input_files[subject], **keywords)))
            if sd_files:
                self.sd.append(iter(Image(sd_files[subject], **keywords)))
    
        resid_files = {}

    def _df(self):
        """
        Work out degrees of freedom from input.
        """

        npred = statistics.utils.rank(self.X)

        self.df_resid = self.nsubject - npred

        if self.df_resid > 0:
            if sd_files:
                try:
                    if (len(df) != len(input_files)):
                        raise ValueError, 'len(df) != len(input_files) in MultiStat'
                    self.input_df = array(list(df))
                except TypeError:
                    self.input_df = N.ones((len(input_files),), N.float64) * df
            else:
                self.input_df = N.inf * N.ones((len(input_files),))

        self.df_fixed = N.add.reduce(self.input_df)
    
        self.df_target = df_target
        if not fixed:
            if fwhm:
                self.input_fwhm = fwhm
            else:
                self.fwhmraw = iter(iterFWHM(self.template, df_resid=self.df_resid, fwhm=fwhmfile, resels=reselsfile, mask=fwhmmask))
        else:
            self.input_fwhm = 6.0
                

    def estimate_varatio(self, Y, S=None, df=None):

        self.Y = N.zeros((self.nsubject, self.npixel), N.float64)
        self.S = N.ones((self.nsubject, self.npixel), N.float64)

    def fit(self, Y, S=None, df=None):

        if not self.fixed and self.varatio is None:
            self.estimate_varatio(Y, S, df)
            
        effect = N.zeros(Y.shape[1:], N.float64)
        sdeffect = N.zeros(Y.shape[1:], N.float64)

        ncpinvX = N.sqrt(N.add.reduce(N.power(N.squeeze(self.pinvX), 2)))
   
        sigma2 = self.varfix * self.varatio

        Sigma = self.S + multiply.outer(N.ones((self.nsubject,), N.float64), sigma2)

        # To ensure that Sigma is always positive:
        if self.fwhm_varatio > 0:
            Sigma = maximum(Sigma, self.S * 0.25)

        W = recipr(Sigma)
        X2W = N.dot(self.Xsq, W)
        XWXinv = recipr(X2W)
        betahat = XWXinv * N.dot(N.transpose(self.X), W * self.Y)
        sdeffect = N.transpose(N.dot(self.contrast, sqrt(XWXinv)))
    
        betahat = N.dot(self.pinvX, self.Y)
        varatio_smooth = self.varatio_smooth.next()
        varatio_smooth.setshape(product(varatio_smooth.shape))
        sigma2 = N.transpose(varatio_smooth)
        sdeffect = ncpinvX * sqrt(N.transpose(sigma2))
        Sigma = N.dot(N.one((self.nsubject,), N.float64), sigma2)
        W = recipr(Sigma)
        effect = N.transpose(N.dot(self.contrast, betahat))

        tstat = effect * recipr(sdeffect)



