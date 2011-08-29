# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np
from scipy.special import gammaln, hermitenorm
import scipy.stats
from scipy.misc import factorial

from .. import rft

from nipy.testing import assert_almost_equal, dec

#def rho(x, dim, df=np.inf):
#    """
#    EC densities for T and Gaussian (df=inf) random fields. 
#    """
#
#    m = df
#    
#    if dim > 0:
#        x = np.asarray(x, np.float64)
#--jarrod: shouldn't Q be rft.Q??
#        q = Q(dim, dfd=df)(x)
#
#        if np.isfinite(m):
#            q *= np.power(1 + x**2/m, -(m-1)/2.)
#        else:
#            q *= np.exp(-x**2/2)
#
#        return q * np.power(2*np.pi, -(dim+1)/2.)
#    else:
#        if np.isfinite(m):
#            return scipy.stats.t.sf(x, df)
#        else:
#            return scipy.stats.norm.sf(x)

def K(dim=4, dfn=7, dfd=np.inf):
    """
    Determine the polynomial K in:

        Worsley, K.J. (1994). 'Local maxima and the expected Euler
        characteristic of excursion sets of \chi^2, F and t fields.' Advances in
        Applied Probability, 26:13-42.

    If dfd=inf, return the limiting polynomial.
    """
    def lbinom(n, j):
        return gammaln(n+1) - gammaln(j+1) - gammaln(n-j+1)

    m = dfd
    n = dfn
    D = dim
    k = np.arange(D)
    coef = 0
    for j in range(int(np.floor((D-1)/2.)+1)):
        if np.isfinite(m):
            t = (gammaln((m+n-D)/2.+j) - 
                 gammaln(j+1) -
                 gammaln((m+n-D)/2.))
            t += lbinom(m-1, k-j) - k * np.log(m)
        else:
            _t = np.power(2., -j) / (factorial(k-j) * factorial(j))
            t = np.log(_t)
            t[np.isinf(_t)] = -np.inf
        t += lbinom(n-1, D-1-j-k) 
        coef += (-1)**(D-1) * factorial(D-1) * np.exp(t) * np.power(-1.*n, k) 
    return np.poly1d(coef[::-1])


def F(x, dim, dfd=np.inf, dfn=1):
    """
    EC densities for F and Chi^2 (dfd=inf) random fields. 
    """
    m = float(dfd)
    n = float(dfn)
    D = float(dim)
    if dim > 0:
        x = np.asarray(x, np.float64)
        k = K(dim=dim, dfd=dfd, dfn=dfn)(x)
        if np.isfinite(m):
            f = x*n/m
            t = -np.log(1 + f) * (m+n-2.) / 2.
            t += np.log(f) * (n-D) / 2.
            t += gammaln((m+n-D)/2.) - gammaln(m/2.)
        else:
            f = x*n
            t = np.log(f/2.) * (n-D) / 2. - f/2. 
        t -= np.log(2*np.pi) * D / 2. + np.log(2) * (D-2)/2. + gammaln(n/2.)
        k *= np.exp(t)
        return k
    else:
        if np.isfinite(m):
            return scipy.stats.f.sf(x, dfn, dfd)
        else:
            return scipy.stats.chi.sf(x, dfn)


def polyF(dim, dfd=np.inf, dfn=1):
    """
    Return the polynomial part of the EC density when evaluating the polynomial
    on the sqrt(F) scale (or sqrt(chi^2)=chi scale).

    The polynomial is such that, if dfd=inf, the F EC density in is just::

        polyF(dim,dfn=dfn)(sqrt(dfn*x)) * exp(-dfn*x/2) * (2\pi)^{-(dim+1)/2}
    """
    n = float(dfn)
    m = float(dfd)
    D = float(dim)
    p = K(dim=D, dfd=m, dfn=n)
    c = p.c
    # Take care of the powers of n (i.e. we want polynomial K evaluated
    # at */n).
    for i in range(p.order+1):
        c[i] /= np.power(n, p.order-i)
    # Now, turn it into a polynomial of x when evaluated at x**2
    C = np.zeros((2*c.shape[0]-1,))
    for i in range(c.shape[0]):
        C[2*i] = c[i]
    # Multiply by the factor x^(dfn-dim) in front (see Theorem 4.6 of
    # Worsley (1994), cited above.
    if dim > dfn: # divide by x^(dim-dfn)
        C = C[0:(C.shape[0] - (dim-dfn))]
    else: # multiply by x^(dim-dfn)
        C = np.hstack([C, np.zeros((dfn-dim,))])
    # Fix up constant in front
    if np.isfinite(m):
        C *= np.exp(gammaln((m+n-D)/2.) - gammaln(m/2.)) * np.power(m, -(n-D)/2.)
    else:
        C *= np.power(2, -(n-D)/2.)
    C /= np.power(2, (dim-2)/2.) * np.exp(gammaln(n/2.))
    C *= np.sqrt(2*np.pi)
    return np.poly1d(C)


def F_alternative(x, dim, dfd=np.inf, dfn=1):
    """
    Another way to compute F EC density as a product of a polynomial and a power
    of (1+x^2/m).
    """
    n = float(dfn)
    m = float(dfd)
    x = np.asarray(x, np.float64)
    p = polyF(dim=dim, dfd=dfd, dfn=dfn)
    v = p(np.sqrt(n*x))
    if np.isfinite(m):
        v *= np.power(1 + n*x/m, -(m+n-2.) / 2.)
    else:
        v *= np.exp(-n*x/2)
    v *= np.power(2*np.pi, -(dim+1)/2.)
    return v


def test_polynomial1():
    # Polynomial part of Gaussian densities are Hermite polynomials.
    for dim in range(1,10):
        q = rft.Gaussian().quasi(dim)
        h = hermitenorm(dim-1)
        yield assert_almost_equal, q.c, h.c


def test_polynomial2():
    # EC density of chi^2(1) is 2 * EC density of Gaussian so polynomial part is
    # a factor of 2 as well.
    for dim in range(1,10):
        q = rft.ChiSquared(dfn=1).quasi(dim)
        h = hermitenorm(dim-1)
        yield assert_almost_equal, q.c, 2*h.c


@dec.slow
def test_polynomial3():
    # EC density of F with infinite dfd is the same as chi^2 --
    # polynomials should be the same.
    for dim in range(10):
        for dfn in range(5,10):
            q1 = rft.FStat(dfn=dfn, dfd=np.inf).quasi(dim)
            q2 = rft.ChiSquared(dfn=dfn).quasi(dim)
            yield assert_almost_equal, q1.c, q2.c


@dec.slow
def test_chi1():
    # EC density of F with infinite dfd is the same as chi^2 -- EC should be the
    # same.
    x = np.linspace(0.1,10,100)
    for dim in range(10):
        for dfn in range(5,10):
            c = rft.ChiSquared(dfn=dfn)
            f = rft.FStat(dfn=dfn, dfd=np.inf)
            chi1 = c.density(dfn*x, dim)
            chi2 = f.density(x, dim)
            yield assert_almost_equal, chi1, chi2


def test_chi2():
    # Quasi-polynomial part of the chi^2 EC density should
    # be the limiting polyF.
    for dim in range(1,10):
        for dfn in range(5,10):
            c = rft.ChiSquared(dfn=dfn)
            p1 = c.quasi(dim=dim)
            p2 = polyF(dim=dim, dfn=dfn)
            yield assert_almost_equal, p1.c, p2.c


def test_chi3():
    # EC density of chi^2(1) is 2 * EC density of Gaussian squared so EC
    # densities factor of 2 as well.
    x = np.linspace(0.1,10,100)
    for dim in range(10):
        g = rft.Gaussian()
        c = rft.ChiSquared(dfn=1)
        ec1 = g.density(np.sqrt(x), dim)
        ec2 = c.density(x, dim)
        yield assert_almost_equal, 2*ec1, ec2


def test_T1():
    # O-dim EC density should be tail probality.
    x = np.linspace(0.1,10,100)
    for dfd in [40,50]:
        t = rft.TStat(dfd=dfd)
        yield assert_almost_equal, t(x), scipy.stats.t.sf(x, dfd)
    t = rft.TStat(dfd=np.inf)
    yield assert_almost_equal, t(x), scipy.stats.norm.sf(x)


def test_search():
    #  Test that the search region works.
    search = rft.IntrinsicVolumes([3,4,5])
    x = np.linspace(0.1,10,100)
    stat = rft.Gaussian(search=search)
    v1 = stat(x)
    v2 = ((5*x + 4*np.sqrt(2*np.pi)) *
          np.exp(-x**2/2.) / np.power(2*np.pi, 1.5) +
          3 * scipy.stats.norm.sf(x))
    assert_almost_equal(v1, v2)


@dec.slow
def test_search1():
    # Test that the search region works.
    # XXX - we are not testing anything
    search = rft.IntrinsicVolumes([3,4,5])
    x = np.linspace(0.1,10,100)
    stats = [rft.Gaussian()]
    for dfn in range(5,10):
        for dfd in [40,50,np.inf]:
            stats.append(rft.FStat(dfn=dfn, dfd=dfd))
            stats.append(rft.TStat(dfd=dfd))
        stats.append(rft.ChiSquared(dfn=dfn))
    for dim in range(7):
        for stat in stats:
            # XXX - v1 appears to be unused
            v1 = stat(x, search=search)
            v2 = 0
            for i in range(search.mu.shape[0]):
                v2 += stat.density(x, i) * search.mu[i]


@dec.slow
def test_search2():
    # Test that the search region works.
    search = rft.IntrinsicVolumes([3,4,5])
    x = np.linspace(0.1,10,100)
    stats = [rft.Gaussian(search=search)]
    ostats = [rft.Gaussian()]
    for dfn in range(5,10):
        for dfd in [40,50,np.inf]:
            stats.append(rft.FStat(dfn=dfn, dfd=dfd, search=search))
            ostats.append(rft.FStat(dfn=dfn, dfd=dfd))
            stats.append(rft.TStat(dfd=dfd, search=search))
            ostats.append(rft.TStat(dfd=dfd))
        stats.append(rft.ChiSquared(dfn=dfn, search=search))
        ostats.append(rft.ChiSquared(dfn=dfn))
    for i in range(len(stats)):
        stat = stats[i]
        ostat = ostats[i]
        v1 = stat(x)
        v2 = 0
        for j in range(search.mu.shape[0]):
            v2 += ostat.density(x, j) * search.mu[j]
        assert_almost_equal(v1, v2)


def test_search3():
    # In the Gaussian case, test that search and product give same results.
    search = rft.IntrinsicVolumes([3,4,5,7])
    g1 = rft.Gaussian(search=search)
    g2 = rft.Gaussian(product=search)
    x = np.linspace(0.1,10,100)
    y1 = g1(x)
    y2 = g2(x)
    assert_almost_equal(y1, y2)


def test_search4():
    # Test that the search/product work well together
    search = rft.IntrinsicVolumes([3,4,5])
    product = rft.IntrinsicVolumes([1,2])
    x = np.linspace(0.1,10,100)
    g1 = rft.Gaussian()
    g2 = rft.Gaussian(product=product)
    y = g2(x, search=search)
    z = g1(x, search=search*product)
    assert_almost_equal(y, z)


def test_search5():
    # Test that the search/product work well together
    search = rft.IntrinsicVolumes([3,4,5])
    product = rft.IntrinsicVolumes([1,2])
    prodsearch = product * search
    x = np.linspace(0,5,101)
    g1 = rft.Gaussian()
    g2 = rft.Gaussian(product=product)
    z = 0
    for i in range(prodsearch.mu.shape[0]):
        z += g1.density(x, i) * prodsearch.mu[i]
    y = g2(x, search=search)
    assert_almost_equal(y, z)


@dec.slow
def test_T2():
    # T**2 is an F with dfn=1
    x = np.linspace(0,5,101)
    for dfd in [40,50,np.inf]:
        t = rft.TStat(dfd=dfd)
        f = rft.FStat(dfd=dfd, dfn=1)
        for dim in range(7):
            y = 2*t.density(x, dim)
            z = f.density(x**2, dim)
            yield assert_almost_equal, y, z


@dec.slow
def test_hotelling1():
    # Asymptotically, Hotelling is the same as F which is the same as chi^2.
    x = np.linspace(0.1,10,100)
    for dim in range(7):
        for dfn in range(5,10):
            h = rft.Hotelling(k=dfn).density(x*dfn, dim)
            f = rft.FStat(dfn=dfn).density(x, dim)
            yield assert_almost_equal, h, f


@dec.slow
def test_hotelling4():
    # Hotelling T^2 should just be like taking product with sphere.
    x = np.linspace(0.1,10,100)
    for dim in range(7):
        search = rft.IntrinsicVolumes([0]*(dim) + [1])
        for k in range(5, 10):
            p = rft.spherical_search(k)
            for dfd in [np.inf,40,50]:
                f = rft.FStat(dfd=dfd, dfn=1)(x, search=p*search)
                t = 2*rft.TStat(dfd=dfd)(np.sqrt(x), search=p*search)
                h2 = 2*rft.Hotelling(k=k, dfd=dfd).density(x, dim)
                h = 2*rft.Hotelling(k=k, dfd=dfd)(x, search=search)

                yield assert_almost_equal, h, t
                yield assert_almost_equal, h, f
                yield assert_almost_equal, h, h2
    search = rft.IntrinsicVolumes([3,4,5])
    for k in range(5, 10):
        p = rft.spherical_search(k)
        for dfd in [np.inf,40,50]:
            f = rft.FStat(dfd=dfd, dfn=1)(x, search=p*search)
            h = 2*rft.Hotelling(k=k, dfd=dfd)(x, search=search)

            h2 = 0
            for i in range(search.mu.shape[0]):
                h2 += 2*rft.Hotelling(k=k, dfd=dfd).density(x, i) * search.mu[i]
            yield assert_almost_equal, h, f
            yield assert_almost_equal, h, h2


def test_hotelling2():
    # Marginally, Hotelling's T^2(k) with m degrees of freedom
    # in the denominator satisfies
    # (m-k+1)/(mk) T^2 \sim  F_{k,m-k+1}.
    x = np.linspace(0.1,10,100)
    for dfn in range(6, 10):
        h = rft.Hotelling(k=dfn)(x)
        chi = rft.ChiSquared(dfn=dfn)(x)
        assert_almost_equal(h, chi)
        chi2 = scipy.stats.chi2.sf(x, dfn)
        yield assert_almost_equal, h, chi2
        # XXX - p appears to be unused
        p = rft.spherical_search(dfn)
        for dfd in [40,50]:
            fac = (dfd-dfn+1.)/(dfd*dfn)
            h = rft.Hotelling(dfd=dfd,k=dfn)(x)
            f = scipy.stats.f.sf(x*fac, dfn, dfd-dfn+1)
            f2 = rft.FStat(dfd=dfd-dfn+1,dfn=dfn)(x*fac)
            yield assert_almost_equal, f2, f
            yield assert_almost_equal, h, f


@dec.slow
def test_roy1():
    # EC densities of Roy with dfn=1 should be twice EC densities of Hotelling
    # T^2's.
    x = np.linspace(0.1,10,100)
    for dfd in [40,50,np.inf]:
        for k in [1,4,6]:
            for dim in range(7):
                h = 2*rft.Hotelling(dfd=dfd,k=k).density(x, dim)
                r = rft.Roy(dfd=dfd,k=k,dfn=1).density(x, dim)
                yield assert_almost_equal, h, r


@dec.slow
def test_onesidedF():
    # EC densities of one sided F should be a difference of
    # F EC densities
    x = np.linspace(0.1,10,100)
    for dfd in [40,50,np.inf]:
        for dfn in range(2,10):
            for dim in range(7):
                f1 = rft.FStat(dfd=dfd,dfn=dfn).density(x, dim)
                f2 = rft.FStat(dfd=dfd,dfn=dfn-1).density(x, dim)
                onesided = rft.OneSidedF(dfd=dfd,dfn=dfn).density(x, dim)
                yield assert_almost_equal, onesided, 0.5*(f1-f2)


@dec.slow
def test_multivariate_forms():
    # MVform with one sphere is sqrt(chi^2), two spheres is sqrt(Roy) with
    # infinite degrees of freedom.
    x = np.linspace(0.1,10,100)
    for k1 in range(5,10):
        m = rft.MultilinearForm(k1)
        c = rft.ChiSquared(k1)
        for dim in range(7):
            mx = m.density(x, dim)
            cx = c.density(x**2, dim)
            yield assert_almost_equal, mx, cx
        for k2 in range(5,10):
            m = rft.MultilinearForm(k1,k2)
            r = rft.Roy(k=k1, dfn=k2, dfd=np.inf)
            for dim in range(7):
                mx = 2*m.density(x, dim)
                rx = r.density(x**2/k2, dim)
                yield assert_almost_equal, mx, rx


def test_scale():
    # Smoke test?
    a = rft.IntrinsicVolumes([2,3,4])
    b = rft.scale_space(a, [3,4], kappa=0.5)


def test_F1():
    x = np.linspace(0.1,10,100)
    for dim in range(1,10):
        for dfn in range(5,10):
            for dfd in [40,50,np.inf]:
                f1 = F(x, dim, dfn=dfn, dfd=dfd)
                f2 = F_alternative(x, dim, dfn=dfn, dfd=dfd)
                yield assert_almost_equal, f1, f2


@dec.slow
def test_F2():
    x = np.linspace(0.1,10,100)
    for dim in range(3,7):
        for dfn in range(5,10):
            for dfd in [40,50,np.inf]:
                f1 = rft.FStat(dfn=dfn, dfd=dfd).density(x, dim) 
                f2 = F_alternative(x, dim, dfn=dfn, dfd=dfd)
                yield assert_almost_equal, f1, f2


@dec.slow
def test_F3():
    x = np.linspace(0.1,10,100)
    for dim in range(3,7):
        for dfn in range(5,10):
            for dfd in [40,50,np.inf]:
                f1 = rft.FStat(dfn=dfn, dfd=dfd).density(x, dim) 
                f2 = F(x, dim, dfn=dfn, dfd=dfd)
                yield assert_almost_equal, f1, f2
