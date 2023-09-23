from scipy.optimize import fmin as fmin_simplex
from scipy.optimize import fmin_bfgs, fmin_cg, fmin_ncg, fmin_powell

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
        raise ValueError(f'unknown optimizer: {optimizer}')

    return fmin, args, subdict(kwargs, keys)


def use_derivatives(optimizer):
    if optimizer in ('simplex', 'powell'):
        return False
    else:
        return True
