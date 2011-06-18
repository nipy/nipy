"""
This module is for canonical hrf specification.
Here we provide for SPM, Glover hrfs and FIR models.
This module closely follows SPM implementation

Author: Bertrand Thirion, 2011
"""

import numpy as np
from scipy.stats import gamma


def gamma_difference_hrf(tr, oversampling=16, time_length=32., onset=0.,
                         delay=6, undershoot=16., dispersion=1.,
                         u_dispersion=1., ratio=0.167):
    """ Compute an hrf as the difference of two gamma functions

    Parameters
    ----------
    tr: float, scan repeat time, in seconds
    oversampling: int, temporal oversampling factor, optional
    time_length: float, hrf kernel length, in seconds
    onset: float, onset of the hrf

    Returns
    -------
    hrf: array of shape(length / tr * oversampling, float),
         hrf sampling on the oversampled time grid
    """
    dt = tr / oversampling
    time_stamps = np.linspace(0, time_length, float(time_length) / dt)
    time_stamps -= onset / dt
    hrf = gamma.pdf(time_stamps, delay / dispersion, dt / dispersion) - \
        ratio * gamma.pdf(
        time_stamps, undershoot / u_dispersion, dt / u_dispersion)
    hrf /= hrf.sum()
    return hrf


def spm_hrf(tr, oversampling=16, time_length=32., onset=0.):
    """ Implementation of the SPM hrf model

    Parameters
    ----------
    tr: float, scan repeat time, in seconds
    oversampling: int, temporal oversampling factor, optional
    time_length: float, hrf kernel length, in seconds
    onset: float, onset of the response

    Returns
    -------
    hrf: array of shape(length / tr * oversampling, float),
         hrf sampling on the oversampled time grid
    """
    return gamma_difference_hrf(tr, oversampling, time_length, onset)


def glover_hrf(tr, oversampling=16, time_length=32., onset=0.):
    """ Implementation of the Glover hrf model

    Parameters
    ----------
    tr: float, scan repeat time, in seconds
    oversampling: int, temporal oversampling factor, optional
    time_length: float, hrf kernel length, in seconds
    onset: float, onset of the response

    Returns
    -------
    hrf: array of shape(length / tr * oversampling, float),
         hrf sampling on the oversampled time grid
    """
    return gamma_difference_hrf(tr, oversampling, time_length, onset,
                                delay=6, undershoot=12., dispersion=.9,
                                u_dispersion=.9, ratio=.35)


def spm_time_derivative(tr, oversampling=16, time_length=32., onset=0.):
    """Implementation of the SPM time derivative hrf (dhrf) model

    Parameters
    ----------
    tr: float, scan repeat time, in seconds
    oversampling: int, temporal oversampling factor, optional
    time_length: float, hrf kernel length, in seconds
    onset: float, onset of the response

    Returns
    -------
    dhrf: array of shape(length / tr, float),
          dhrf sampling on the provided grid
    """
    do = .1
    dhrf = 1. / do * (spm_hrf(tr, oversampling, time_length, onset + do) -
                      spm_hrf(tr, oversampling, time_length, onset))
    return dhrf


def glover_time_derivative(tr, oversampling=16, time_length=32., onset=0.):
    """Implementation of the flover time derivative hrf (dhrf) model

    Parameters
    ----------
    tr: float, scan repeat time, in seconds
    oversampling: int, temporal oversampling factor, optional
    time_length: float, hrf kernel length, in seconds
    onset: float, onset of the response

    Returns
    -------
    dhrf: array of shape(length / tr, float),
          dhrf sampling on the provided grid
    """
    do = .1
    dhrf = 1. / do * (glover_hrf(tr, oversampling, time_length, onset + do) -
                      glover_hrf(tr, oversampling, time_length, onset))
    return dhrf


def spm_dispersion_derivative(tr, oversampling=16, time_length=32., onset=0.):
    """Implementation of the SPM dispersion derivative hrf model

    Parameters
    ----------
    tr: float, scan repeat time, in seconds
    oversampling: int, temporal oversampling factor, optional
    time_length: float, hrf kernel length, in seconds
    onset: float, onset of the response

    Returns
    -------
    dhrf: array of shape(length / tr * oversampling, float),
          dhrf sampling on the oversampled time grid
    """
    dd = .01
    dhrf = 1. / dd * (gamma_difference_hrf(tr, oversampling, time_length,
                                           onset, dispersion=1. + dd) -
                      spm_hrf(tr, oversampling, time_length, onset))
    return dhrf


def sample_condition(exp_condition, frametimes, oversampling=16):
    """ this function samples the experimental condition at frametimes

    Parameters
    ----------
    exp_condition: a tuple of 3 arrays of shape n, corresponding
                   to  (onsets, duration, value),
                   describing the experimental condition
    frametimes: array of shape(n)
    over_sampling: int, over_sampling factor

    Returns
    -------
    regressor: array of shape(n)
    """
    # generate the oversampled frame times
    n = frametimes.size
    hr_frametimes = np.linspace(0, frametimes.max() * (1 + 1. / (n - 1)),
                                n * oversampling + 1)

    # get the regressor information
    onsets, duration, values = exp_condition
    onsets, duration, values = np.asanyarray(onsets), np.asanyarray(duration),\
        np.asanyarray(values)

    # generate the regressor time course
    tmax = len(hr_frametimes)
    regressor = np.zeros_like(hr_frametimes).astype(np.float)
    t_onset = np.minimum(np.searchsorted(hr_frametimes, onsets), tmax - 1)
    regressor[t_onset] += values
    t_offset = np.minimum(np.searchsorted(hr_frametimes, onsets + duration),
                          tmax - 1)
    # for event related, shift the offset by 1
    for i, to in enumerate(t_offset):
        if to > 0 and to < tmax - 1 and to == t_onset[i]:
            t_offset[i] += 1

    regressor[t_offset] -= values
    regressor = np.cumsum(regressor)

    # normalize the regressor in case of  block design
    #if (duration > 0).any():
    #    regressor /= oversampling
    return regressor, hr_frametimes


def resample_regressor(hr_regressor, hr_frametimes, frametimes, kind='linear'):
    """ this function samples the regressors at frametimes

    Parameters
    ----------
    hr_regressor: array of shape(n),
                  the regressor time course sampled at high temporal resolution
    hr_frametimes: array of shape(n),
                   the corresponding time stamps
    frametimes: array of shape(p),
                the desired time stamps
    kind: string, optional, the kind of desired interpolation

    Returns
    -------
    regressor: array of shape(p), the resampled regressor
    """
    from scipy.interpolate import interp1d
    f = interp1d(hr_frametimes, hr_regressor)
    return f(frametimes).T


def _orthogonalize(X):
    """ This function orthogonalizes the columns of the design matrix wrt
    to the first column

    Parameters
    ----------
    X: array of shape(n, p), the data to be orthogonalized

    Returns
    -------
    X: after orthogonalization

    Note
    ----
    X is changed in place. the columns are not normalized
    """
    if X.size == X.shape[0]:
        return X
    from numpy.linalg import pinv
    for i in range(1, X.shape[1]):
        X[:, i] -= np.dot(X[:, i], np.dot(X[:, :i], pinv(X[:, :i])))
    return X


def _regressor_names(con_name, hrf_model, fir_delays=None):
    """ returns a list of regressor names, computed from con-name and hrf type

    Parameters
    ----------
    con_name: string, identifier of the condition
    hrf_model: string, identifier of the hrf model

    Returns
    -------
    names: a list of strings yielding the regressor names
    """
    if hrf_model == 'Canonical':
        return [con_name]
    elif hrf_model == "Canonical With Derivative":
        return [con_name, con_name + "_derivative"]
    elif hrf_model == 'spm':
        return [con_name]
    elif hrf_model == 'spm_time':
        return [con_name, con_name + "_derivative"]
    elif hrf_model == 'spm_time_dispersion':
        return [con_name, con_name + "_derivative", con_name + "_dispersion"]
    elif hrf_model == 'FIR':
        return [con_name + "_delay_%d" % i for i in fir_delays]


def _hrf_kernel(hrf_model, tr, oversampling=16, fir_delays=None):
    """ Given the specification of the hemodynamic model and time parameters,
    return the list of matching kernels

    Parameters
    ----------
    hrf_model: string, identifier of the hrf model
    tr: the repetition time in seconds
    oversampling: int, temporal oversampling factor to have a smooth hrf
    fir_delays: list of for delays

    Returns
    -------
    hkernel: a list of hrf kernels, depending on the specified hrf model
    """
    if hrf_model == 'spm':
        hkernel = [spm_hrf(tr, oversampling)]
    elif hrf_model == 'spm_time':
        hkernel = [spm_hrf(tr, oversampling),
                   spm_time_derivative(tr, oversampling)]
    elif hrf_model == 'spm_time_dispersion':
        hkernel = [spm_hrf(tr, oversampling),
                   spm_time_derivative(tr, oversampling),
                   spm_dispersion_derivative(tr, oversampling)]
    elif hrf_model == 'Canonical':
        hkernel = [glover_hrf(tr, oversampling)]
    elif hrf_model == 'Canonical With Derivative':
        hkernel = [glover_hrf(tr, oversampling),
                   glover_time_derivative(tr, oversampling)]
    elif hrf_model == 'FIR':
        hkernel = [np.hstack((np.zeros(f * oversampling),
                              np.ones(oversampling)))
                   for f in fir_delays]
    else:
        raise ValueError('Unknown hrf model')
    return hkernel


def compute_regressor(exp_condition, hrf_model, frametimes, con_id='cond',
                      oversampling=16, fir_delays=None):
    """ This is the main function to convolve regressors with hrf model

    Parameters
    ----------
    exp_condition: descriptor of an experimental condition
    hrf_model: string, the hrf model to be used. Can be chosen among:
               'spm', 'spm_time', 'spm_time_dispersion', 'canonical',
               'canonical_derivative', 'fir'
    frametimes: array of shape (n):the sought
    con_id: string, optional identifier of the condition
    oversampling: int, optional, oversampling factor to perform the convolution
    fir_delays: array-like of int, onsets corresponding to the fir basis

    Returns
    -------
    creg: array of shape(n_scans, n_reg): computed regressors sampled
                                          at frametimes
    reg_names: list of strings, corresponding regressor names

    Note
    ----
    The different hemodynamic models can be understood as follows:
    'spm': this is the hrf model used in spm
    'spm_time': this is the spm model plus its time derivative (2 regressors)
    'spm_time_dispersion': idem, plus dispersion derivative (3 regressors)
    'canonical': this one corresponds to the Glover hrf
    'canonical_derivative': the Glover hrf + time derivative (2 regressors)
    'fir': finite impulse response basis, a set of delayed dirac models 
           with arbitrary length. This one currently assumes regularly spaced 
           frametimes (i.e. fixed time of repetition).
    It is expected that spm standard and Glover model would not yield 
    large differences in most cases.
    """
    # this is the average tr in this session, not necessarily the true tr
    tr = float(frametimes.max()) / (np.size(frametimes) - 1)

    # 1. create the high temporal resolution regressor
    hr_regressor, hr_frametimes = sample_condition(
        exp_condition, frametimes, oversampling)

    # 2. create the  hrf model(s)
    hkernel = _hrf_kernel(hrf_model, tr, oversampling, fir_delays)

    # 3. convolve the regressor and hrf, and downsample the regressor
    conv_reg = np.array([np.convolve(hr_regressor, h)[:hr_regressor.size]
                         for h in hkernel])

    # 4. temporally resample the regressors
    creg = resample_regressor(conv_reg, hr_frametimes, frametimes)

    # 5. ortogonalize the regressors
    creg = _orthogonalize(creg)

    # 6 generate regressor names
    reg_names = _regressor_names(con_id, hrf_model, fir_delays=fir_delays)
    return creg, reg_names
