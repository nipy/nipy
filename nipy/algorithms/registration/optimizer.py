from scipy.optimize import (fmin as fmin_simplex,
                            fmin_powell,
                            fmin_cg,
                            fmin_bfgs,
                            fmin_ncg)
from ..optimize import fmin_steepest


def subdict(dic, keys):
    sdic = {}
    for k in keys:
        sdic[k] = dic[k]
    return sdic


def configure_optimizer(optimizer, fprime=None, fhess=None, **kwargs):
    """
    Return the minimization function
    """
    args = []
    kwargs['fprime'] = fprime
    kwargs['fhess'] = fhess
    kwargs['avextol'] = kwargs['xtol']

    if optimizer == 'simplex':
        keys = ('xtol', 'ftol', 'maxiter', 'maxfun')
        fmin = fmin_simplex
    elif optimizer == 'powell':
        keys = ('xtol', 'ftol', 'maxiter', 'maxfun')
        fmin = fmin_powell
    elif optimizer == 'cg':
        keys = ('gtol', 'maxiter', 'fprime')
        fmin = fmin_cg
    elif optimizer == 'bfgs':
        keys = ('gtol', 'maxiter', 'fprime')
        fmin = fmin_bfgs
    elif optimizer == 'ncg':
        args = [fprime]
        keys = ('avextol', 'maxiter', 'fhess')
        fmin = fmin_ncg
    elif optimizer == 'steepest':
        keys = ('xtol', 'ftol', 'maxiter', 'fprime')
        fmin = fmin_steepest
    else:
        raise ValueError('unknown optimizer: %s' % optimizer)

    return fmin, args, subdict(kwargs, keys)


def use_derivatives(optimizer):
    if optimizer in ('simplex', 'powell'):
        return False
    else:
        return True
