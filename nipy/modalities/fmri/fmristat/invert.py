from __future__ import absolute_import
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

__docformat__ = 'restructuredtext'

import numpy as np

from nipy.algorithms.statistics.models.nlsmodel import NLSModel

def invertR(delta, IRF, niter=20):
    """
    If IRF has 2 components (w0, w1) return an estimate of the inverse of
    r=w1/w0, as in Liao et al. (2002). Fits a simple arctan model to the
    ratio w1/w0.
    """

    R = IRF[1](delta) / IRF[0](delta)

    def f(x, theta):
        a, b, c = theta
        _x = x[:,0]
        return a * np.arctan(b * _x) + c

    def grad(x, theta):
        a, b, c = theta
        value = np.zeros((3, x.shape[0]))
        _x = x[:,0]
        value[0] = np.arctan(b * _x)
        value[1] = a / (1. + np.power((b * _x), 2.)) * _x
        value[2] = 1.
        return value.T

    c = delta.max() / (np.pi/2)
    n = delta.shape[0]
    delta0 = ((delta[n // 2 + 2] - delta[n // 2 + 1])
              / (R[n // 2 + 2] - R[n // 2 + 1]))
    if delta0 < 0:
        c = (delta.max() / (np.pi/2)) * 1.2
    else:
        c = -(delta.max() / (np.pi/2)) * 1.2

    design = R.reshape(R.shape[0], 1)
    model = NLSModel(Y=delta,
                     design=design,
                     f=f,
                     grad=grad,
                     theta=np.array([4., 0.5, 0]),
                     niter=niter)

    for iteration in model:
        next(model)

    a, b, c = model.theta

    def _deltahat(r):
        return a * np.arctan(b * r) + c

    def _ddeltahat(r):
        return a * b / (1 + (b * r)**2)

    def _deltahatinv(d):
        return np.tan((d - c) / a) / b

    def _ddeltahatinv(d):
        return 1. / (a * b * np.cos((d - c) / a)**2)

    for fn in [_deltahat, _ddeltahat, _deltahatinv, _ddeltahatinv]:
        setattr(fn, 'a', a)
        setattr(fn, 'b', b)
        setattr(fn, 'c', c)

    return model.theta, _deltahat, _ddeltahat, _deltahatinv, _ddeltahatinv
