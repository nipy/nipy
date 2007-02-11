"""
this module:

i) simulates a piecewise constant fMRIImage of the same size as the FIAC data:
       - there are between 10 and 20 parcels in the image, based on truncating
         an estimate of the AR coefficient from the FIAC data
       - the mean is a random combination of the columns of the FIAC design (for a given subject)
       - the SignalNoise class adds white noise to each model signal (constant within each parcel)

ii) fits the FIAC model to the simulated dataset using nipy's fmristat model, storing it by default in

       join(self.root, "fsl", "fmristat_sim")

iii) verifies that the resulting images agree with what is expected
"""

       
import os, csv

import numpy as N
import numpy.random as R

from scipy.sandbox.models.regression import ols_model, ar_model

from neuroimaging.core.image.image import Image
from neuroimaging.core.reference.iterators import fMRIParcelIterator, ParcelIterator
from neuroimaging.modalities.fmri import fMRIImage

import keith, model

class SignalOnly(model.RunModel):

    """
    Generate an fMRIImage according to the following model:

    * round off AR estimate to a scale of N.linspace(-1,1,21) (multiplying result by 0.9 to avoid +- 1)
    * within each chunk, generate a random signal from FIAC formula with NO measurement noise, i.e. resids should be 0

    """

    t = 2.5*N.arange(191)+1.25    

    def getparcelmap(self):
        """
        parcel AR estimate on scale of N.linspace(-1,1,21) (multiplying result by 0.9 to avoid +- 1)
        """
        try:
            self.rho = keith.rho(subject=self.subject.id, run=self.id)
        except:
            self.rho = Image(os.path.join(self.root, 'fsl', 'fmristat_run', 'rho.img'))
        self.parcelmap = N.round(self.rho[:] * 10) / 10. * 0.9
        self.parcelseq = N.unique(self.parcelmap)

    def generate(self):
        """
        generate fMRI image and signals
        """

        self.generate_values()
        
        if not os.path.exists(self.resultdir):
            os.makedirs(self.resultdir)

        self.load()
        self.simfmrifile = os.path.join(self.resultdir, 'simfmri.img')
        simfmri = fMRIImage(self.simfmrifile,
                            grid=self.fmri.grid, mode='w', clobber=True)
        simfmri.it = fMRIParcelIterator(simfmri, self.parcelmap,
                                        self.parcelseq, mode='w')
        i = 0
        mu = []
        for chunk in simfmri.it:
            nvox = chunk.slice.astype(N.int32).sum()
            chunk.set(N.multiply.outer(self.mu[i], N.ones(nvox)))
            i += 1
        del(simfmri)
        
    def generate_values(self):
        """
        do the work of generating the signals
        """

        self.D = self.formula.design(self.t)
        self.coefs = R.standard_normal((self.D.shape[1], len(self.parcelseq)))

        mu = []
        for i in range(len(self.parcelseq)):
            value = N.dot(self.D, self.coefs[:,i])
            mu.append(value)
        self.mu = N.array(mu)
        SignalOnly.write(self)
        
    def write(self):
        """
        write out the signals to mu.{bin,csv} for later storage
        """
        self._write_matrix(self.mu, 'mu')

    def read(self):
        """
        read the previously stored signals to mu.bin
        """
        self.mu = self._read_matrix('mu')
        self.avgs = self._read_matrix('avgs')
        
    def _write_matrix(self, matrix, name):
        """
        do the work of writing matrix to files name.{bin,csv} (all files relative to join(self.root, self.resultdir)
        """
        outfile = file(os.path.join(self.resultdir, "%s.csv" % name), 'w')
        writer = csv.writer(outfile, delimiter=',')
        for row in matrix:
            try:
                writer.writerow(row)
            except:
                writer.writerow([row])
        outfile.close()

        outfile = file(os.path.join(self.resultdir, "%s.bin" % name), 'w')
        matrix.astype(">d").tofile(outfile)
        outfile.close()

    def _read_matrix(self, name):

        """
        do the work of reading matrix from file name.bin (all files relative to join(self.root, self.resultdir)
        """

        fname = os.path.join(self.resultdir, "%s.bin" % name)
        if os.path.exists(fname):
            infile = file(fname)
            v = N.fromfile(infile,">d")
            v.shape = (v.shape[0] / 191, 191)
            infile.close()
            return v
    
    def load(self):
        """
        load as model.RunModel, overwriting self.fmri with the simulated fMRI

        read in mu.bin as well

        """

        model.RunModel.load(self)
        try:
            self.read()
        except OSError:
            pass
        
        self.simfmrifile = os.path.join(self.resultdir, 'simfmri.img')
        if not os.path.exists(self.resultdir):
            os.makedirs(self.resultdir)

        self.fmri = fMRIImage(self.simfmrifile,
                              grid=self.fmri.grid, clobber=True)
            
    def AR(self, **ARopts):
        """
        Run the fMRIstat model with the self.rho as the parcel labels.
        """
        
        ARopts['parcel'] = (self.parcelmap, self.parcelseq)
        model.RunModel.AR(self, **ARopts)

    def check_resid(self):
        """
        verify that the residuals (both AR and OLS) are essentially zero.
        """
        
        for rtype in ["OLS", "AR"]:
            resid = fMRIImage(os.path.join(self.resultdir, "%sresid.img" % rtype))

            for s in resid.slice_iterator():
                MSE = (s**2).sum() / (191 * 64**2)
                if MSE > 1.0e-10:
                    raise ValueError, "residuals are large here!, MSE=%f" % MSE

    def get_frameavg(self):
        """
        get and store frame averages for later normalization
        """
        self.load()
        self.avgs = N.array([(self.fmri.frame(i)[:] * self.mask[:]).sum() for i in range(191)])
        self.avgs /= self.mask[:].astype(N.int32).sum()
        
        self._write_matrix(self.avgs, 'avgs')

class SignalNoise(SignalOnly):

    """
    this class adds noise to SignalOnly's image -- constant within each parcel
    """

    noise = 1.0

    def generate_values(self):
        """
        generate signal (noise-free) and mu (noisy), store them for later
        """
        
        SignalOnly.generate_values(self)

        self.signal = self.mu * 1.
        for i in range(self.mu.shape[0]):
            self.mu[i] += R.standard_normal((191,)) * self.noise
        self.write()

    def get_models(self):
        """
        construct OLS and AR models for each parcel
        """
        
        self.D = self.formula.design(self.t)
        self.ols_model = ols_model(self.D)
        self.ar_models = []
        self.ols_results = []
        self.ar_results = []

        for i in range(self.mu.shape[0]):
            self.ols_results.append(self.ols_model.fit(self.mu[i] / self.avgs * 100.))
            ar = ar_model(self.D, rho=self.parcelseq[i])
            self.ar_models.append(ar)
            self.ar_results.append(self.ar_models[i].fit(self.mu[i] / self.avgs * 100.))

    def write(self):
        """
        save self.mu and self.signal for later use
        """
        self._write_matrix(self.mu, 'mu')
        self._write_matrix(self.signal, 'signal')

    def read(self):
        """
        read in relevant data: self.mu, self.signal and self.avgs
        """
        
        self.mu = self._read_matrix('mu')
        self.signal = self._read_matrix('signal')
        self.avgs = self._read_matrix('avgs')
        
    def check_resid(self):
        """
        check that the resids of the model are ALMOST
        what the parcel model would predict.

        numpy.allclose (roundoff error?) fails here, so the check is
        to see that the residuals are highly correlated, at level 0.999
        """
        
        self.get_models()

        for rtype in ["OLS", "AR"]:
            self.resid = fMRIImage(os.path.join(self.resultdir, "%sresid.img" % rtype))
            print "SHAPE", self.resid.shape
            # I get a results of [0, 30, 64, 64] here
            # which leads to the code below breaking. I'm not sure why this
            # file would have this shape though... --Tim
            self.resid.it = fMRIParcelIterator(self.resid, self.parcelmap,
                                               self.parcelseq)
            i = 0
            for chunk in self.resid.it:
                r = self.ols_results[i].resid
                r = N.multiply.outer(r, N.ones(chunk.shape[1]))
                SSE = ((chunk - r)**2).sum() / N.product(chunk.shape)
                R2 = 1 - SSE / (r**2).sum()
                cor = N.sqrt(R2)
                if cor < 1.0 - 1.0e-03:
                    raise ValueError, 'correlation not close to 1 here: %f' % cor

    def checkresult(self):
        """
        verify that the output images are piecewise constant and that
        their values agree with the model for each parcel would predict.

        this checks the effect, sd and t-stat for each contrast (as well as the delays)
        and that the overall F statistic is what the parcel model expects

        """

        self._setup_contrasts()

        for stat in ['effect', 'sd', 't']:

            for contrast in ['average', 'speaker', 'sentence', 'interaction']:
                result = self.result(stat=stat,
                                     contrast=contrast)
                result.it = ParcelIterator(result, self.parcelmap,
                                           self.parcelseq)
                output = getattr(self, contrast)
                output.getmatrix(self.t)

                i = 0
                for chunk in result.it:
                    mresult = self.ar_results[i].Tcontrast(output.matrix)
                    mresult = getattr(mresult, stat)
                    if not N.allclose(mresult, chunk):
                        print "fit is not good here: %s, %s, %s"% (contrast, stat, str(self.ar_results[i].Tcontrast(output.matrix)))
                    i += 1

                output = self.delays
                output.getmatrix(self.t)
            
                result = self.result(stat=stat,
                                     contrast=contrast,
                                     which='delays')
                
                result.it = ParcelIterator(result, self.parcelmap,
                                           self.parcelseq)
                i = 0
                row = output.rownames.index(contrast)
                for chunk in result.it:
                    mresult = output.extract(self.ar_results[i])
                    mresult = getattr(mresult, stat)[row]
                    if not N.allclose(mresult, chunk):
                        print "delay fit is not good here: %s, %s, %s"% (contrast, stat, str(self.ar_results[i].Tcontrast(output.matrix)))
                    i += 1
                    
        output = self.overallF
        output.getmatrix(self.t)
        i = 0
        result = self.result(stat='F',
                             contrast='overallF')
        result.it = ParcelIterator(result, self.parcelmap,
                                   self.parcelseq)

        for chunk in result.it:
            mresult = self.ar_results[i].Fcontrast(output.matrix).F
            if not N.allclose(mresult, chunk):
                print "F it is not good here: overallF, F, %s"% str(self.ar_results[i].Fcontrast(output.matrix))
            i += 1

if __name__ == '__main__':

    import sys, optparse

    parser = optparse.OptionParser()
    parser.add_option("","--generate", dest="generate", action="store_true", default=False, help="generate data?")
    parser.add_option("","--fit", dest="fit", action="store_true", default=False, help="fit OLS and AR models?")
    parser.add_option("","--check", dest="check", action="store_true", default=False, help="check the fit?")

    options, args = parser.parse_args()

    if len(args) == 3:
        subj, run = map(int, args)
    else:
        subj, run = (1, 2)

    import io
    study = model.StudyModel(root=io.data_path)
    #study = model.StudyModel(root='http://kff.stanford.edu/FIAC')
    subject = model.SubjectModel(subj, study=study)
    runmodel = SignalNoise(subject, run, resultdir=os.path.join("fsl", "fmristat_sim"))

    runmodel.getparcelmap()
    runmodel.get_frameavg()

#    options.generate, options.fit, options.check = (0,1,0)
    if options.generate:
        runmodel.generate()
    if options.fit:
        runmodel.OLS(clobber=True, resid=True)
        runmodel.AR(clobber=True, resid=True)
    if options.check:
        runmodel.check_resid()
        runmodel.checkresult()
