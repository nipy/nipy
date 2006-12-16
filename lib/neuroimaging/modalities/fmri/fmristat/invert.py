import numpy as N

def invertR(delta, IRF, niter=20, verbose=False):
    """
    If IRF has 2 components (w0, w1) return an estimate of the inverse of
    r=w1/w0, as in Liao et al. (2002). Fits a simple arctan model to the
    ratio w1/w0.?

    """

    R = IRF[1](delta) / IRF[0](delta)

    def f(x, theta):
        a, b, c = theta
        _x = x[:,0]
        return a * N.arctan(b * _x) + c

    def grad(x, theta):
        a, b, c = theta
        value = N.zeros((3, x.shape[0]), N.float64)
        _x = x[:,0]
        value[0] = N.arctan(b * _x)
        value[1] = a / (1. + N.power((b * _x), 2.)) * _x
        value[2] = 1.
        return N.transpose(value)

    c = delta.max() / (N.pi/2)
    n = delta.shape[0]
    delta0 = (delta[n/2+2] - delta[n/2+1])/(R[n/2+2] - R[n/2+1])
    if delta0 < 0:
        c = (delta.max() / (N.pi/2)) * 1.2
    else:
        c = -(delta.max() / (N.pi/2)) * 1.2

    from neuroimaging.algorithms.statistics import nlsmodel
    design = R.reshape(R.shape[0], 1)
    model = nlsmodel.NLSModel(Y=delta,
                              design=design,
                              f=f,
                              grad=grad,
                              theta=N.array([4., 0.5, 0]),
                              niter=niter)

    for iteration in model:
        model.next()

    a, b, c = model.theta

    def _deltahat(r):
        return a * N.arctan(b * r) + c

    def _ddeltahat(r):
        return a * b / (1 + (b * r)**2) 

    def _deltahatinv(d):
        return N.tan((d - c) / a) / b

    def _ddeltahatinv(d):
        return 1. / (a * b * N.cos((d - c) / a)**2)

    for fn in [_deltahat, _ddeltahat, _deltahatinv, _ddeltahatinv]:
        setattr(fn, 'a', a)
        setattr(fn, 'b', b)
        setattr(fn, 'c', c)

    return model.theta, _deltahat, _ddeltahat, _deltahatinv, _ddeltahatinv
