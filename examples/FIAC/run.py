from fiac import *

import neuroimaging.modalities.fmri.protocol as protocol
import scipy.sandbox.models.contrast as contrast
from neuroimaging.modalities.fmri.fmristat.delay import DelayHRF
import neuroimaging.modalities.fmri.fmristat as fmristat

def FIACformula(subj=3, run=3, normalize=True, df=5):
    p = FIACprotocol(subj=subj, run=run)

    delay_irf = DelayHRF()
    irf = delay_irf[0]

    f = FIACfmri(subj=subj, run=run)
    m = FIACmask(subj=subj, run=run)

    if p:
            
        p.convolve(delay_irf)

        drift_fn = protocol.SplineConfound(window=[0,f.frametimes.max()+2.5],
                                           df=df)
        drift = protocol.ExperimentalQuantitative('drift', drift_fn)

        formula = p + drift

        if p.design_type == 'block':
            begin = FIACbegin_block(subj=subj, run=run)
        else:
            begin = FIACbegin_event(subj=subj, run=run)
        formula += begin
        begin.convolve(irf)
           
        del(f); del(m); gc.collect()
        return formula
    else:
        return None


def FIACrun(subj=3, run=3, output_fwhm=False, normalize=True):

    ARopts = {'clobber':True,
              'resid':False}

    if output_fwhm:
        ARopts['resid'] = True

    OLSopts = {'clobber':True}
    tshift = 1.25

    delay_irf = DelayHRF()
    irf = delay_irf[0]

    p = FIACprotocol(subj=subj, run=run)

    if p:
        formula = FIACformula(subj=subj, run=run, df=5)

        f = FIACfmri(subj=subj, run=run)
        m = FIACmask(subj=subj, run=run)

        if normalize:

            brainavg = fmristat.WholeBrainNormalize(f, mask=m)
            
            brainavg_fn = protocol.InterpolatedConfound(times=f.frametimes + tshift,
                                                        values=brainavg.avg)

            wholebrain = protocol.ExperimentalQuantitative('whole_brain',
                                                           brainavg_fn)

        if normalize:
            formula += wholebrain
            OLSopts['normalize'] = brainavg

        # output some contrasts, here is one from the term "p" in "formula",
        # i.e. an F for all effects of interest

        p = formula['FIAC_design']
        task = contrast.Contrast(p, formula, name='task')

        # another built by linear combinations of functions of (experiment) time
        SSt_SSp = p['SSt_SSp'].astimefn()
        DSt_SSp = p['DSt_SSp'].astimefn()
        SSt_DSp = p['SSt_DSp'].astimefn()
        DSt_DSp = p['DSt_DSp'].astimefn()

        overall = (SSt_SSp + DSt_SSp + SSt_DSp + DSt_DSp) * 0.25

        # important: overall is NOT convolved with HRF even though p was!!!
        # irf here is the first PC of the shifted Glover HRF

        overall = irf.convolve(overall)

        overall = contrast.Contrast(overall,
                                    formula,
                                    name='overall')

        # the "FIAC" contrasts
        # annoying syntax: floats must be on LHS -- have to fix __add__, __mul__ methods of TimeFunction
        
        sentence = (DSt_SSp + DSt_DSp) * 0.5 - (SSt_SSp + SSt_DSp) * 0.5
        sentence = irf.convolve(sentence)
        sentence = contrast.Contrast(sentence, formula, name='sentence')
        
        speaker =  (SSt_DSp + DSt_DSp) / 2. - (SSt_SSp + DSt_SSp) / 2.
        speaker = irf.convolve(speaker)
        speaker = contrast.Contrast(speaker, formula, name='speaker')
        
        interaction = SSt_SSp - SSt_DSp - DSt_SSp + DSt_DSp
        interaction = irf.convolve(interaction)
        interaction = contrast.Contrast(interaction, formula, name='interaction')
        
        # delay
    
        delays = fmristat.DelayContrast([SSt_DSp, DSt_DSp, SSt_SSp, DSt_SSp],
                                        [[0.5,0.5,-0.5,-0.5],
                                         [-0.5,0.5,-0.5,0.5],
                                         [-1,1,1,-1],
                                         [0.25,0.25,0.25,0.25]],
                                        formula,
                                        name='task',
                                        rownames=['speaker',
                                                  'sentence',
                                                  'interaction',
                                                  'overall'],
                                        IRF=delay_irf)

        # OLS pass
        
        OLS = fmristat.fMRIStatOLS(f, formula=formula, mask=m,
                                   tshift=tshift, 
                                   **OLSopts)
        OLS.reference = overall
        
        toc = time.time()
        OLS.fit()
        tic = time.time()
        
        print 'OLS time', `tic-toc`
        
        # maybe you want to save the estimates of the AR(1) parameter
        
        rho = OLS.rho_estimator.img
        rho.tofile('fmristat_run/rho.img', clobber=True)
        
        # AR pass
        
        contrasts = [task, overall, sentence, speaker, interaction, delays]
        toc = time.time()
        AR = fmristat.fMRIStatAR(OLS, contrasts=contrasts, tshift=tshift, **ARopts)
        AR.fit()
        tic = time.time()
        
        print 'AR time', `tic-toc`

        # if we output the AR whitened residuals, we might as
        # well output the FWHM, too

        if output_fwhm:
            resid = neuroimaging.modalities.fmri.fMRIImage(FIACpath('fsl/fmristat_run/ARresid.img', subj=subj, run=run))
            fwhmest = fastFWHM(resid, fwhm=FIACpath('fsl/fmristat_run/fwhm.img'), clobber=True)
            fwhmest()

        del(OLS); del(AR); gc.collect()
        return formula

if __name__ == '__main__':

    import sys
    if len(sys.argv) == 3:
        subj, run = map(int, sys.argv[1:])
    else:
        subj, run = (3, 3)

    FIACrun(subj=subj, run=run)
