import numpy as N
import numpy.linalg as L

def orth(X, tol=1.0e-07):
    """
    
    Compute orthonormal basis for the column span of X.
    
    Rank is determined by zeroing all singular values, u, less
    than or equal to tol*u.max().

    INPUTS:
        X  -- n-by-p matrix

    OUTPUTS:
        B  -- n-by-rank(X) matrix with orthonormal columns spanning
              the column rank of X
    """

    B, u, _ = L.svd(X, full_matrices=False)
    nkeep = N.greater(u, tol*u.max()).astype(N.int).sum()
    return B[:,:nkeep]
    
def reml(sigma, components, design=None, n=1, niter=128,
         penalty_cov=N.exp(-32), penalty_mean=0):
    """

    Adapted from spm_reml.m

    ReML estimation of covariance components from sigma using design matrix.

    INPUTS:
        sigma        -- m-by-m covariance matrix
        components   -- q-by-m-by-m array of variance components
                        mean of sigma is modeled as a some over components[i]
        design       -- m-by-p design matrix whose effect is to be removed for
                        ReML. If None, no effect removed (???)
        n            -- degrees of freedom of sigma
        penalty_cov  -- quadratic penalty to be applied in Fisher algorithm.
                        If the value is a float, f, the penalty is
                        f * identity(m). If the value is a 1d array, this is
                        the diagonal of the penalty. 
        penalty_mean -- mean of quadratic penalty to be applied in Fisher
                        algorithm. If the value is a float, f, the location
                        is f * N.ones(m).


    OUTPUTS:
        C            -- estimated mean of sigma
        h            -- array of length q representing coefficients
                        of variance components
        cov_h        -- estimated covariance matrix of h
    """

    # initialise coefficient, gradient, Hessian

    Q = components
    PQ = N.zeros(Q.shape)
    
    q = Q.shape[0]
    m = Q.shape[1]

    # coefficient
    h = N.array([N.diag(Q[i]).mean() for i in range(q)])

    ## SPM initialization
    ## h = N.array([N.any(N.diag(Q[i])) for i in range(q)]).astype(N.float)

    C = N.sum([h[i] * Q[i] for i in range(Q.shape[0])], axis=0)

    # gradient in Fisher algorithm
    
    dFdh = N.zeros(q)

    # Hessian in Fisher algorithm
    dFdhh = N.zeros((q,q))

    # penalty terms

    penalty_cov = N.asarray(penalty_cov)
    if penalty_cov.shape == ():
        penalty_cov = penalty_cov * N.identity(q)
    elif penalty_cov.shape == (q,):
        penalty_cov = N.diag(penalty_cov)
        
    penalty_mean = N.asarray(penalty_mean)
    if penalty_mean.shape == ():
        penalty_mean = N.ones(q) * penalty_mean
        
    # compute orthonormal basis of design space

    if design is not None:
        X = orth(design)
    else:
        X = None

    _iter = 0
    _F = N.inf
    
    while True:

        # Current estimate of mean parameter

        iC = L.inv(C + N.identity(m) / N.exp(32))

        # E-step: conditional covariance 

        if X is not None:
            iCX = N.dot(iC, X)
            Cq = L.inv(X.T, iCX)
            P = iC - N.dot(iCX, N.dot(Cq, iCX))
        else:
            P = iC

        # M-step: ReML estimate of hyperparameters
 
        # Gradient dF/dh (first derivatives)
        # Expected curvature (second derivatives)

        U = N.identity(m) - N.dot(P, sigma) / n

        for i in range(q):
            PQ[i] = N.dot(P, Q[i])
            dFdh[i] = -(PQ[i] * U).sum() * n / 2

            for j in range(i+1):
                dFdhh[i,j] = -(PQ[i]*PQ[j]).sum() * n / 2
                dFdhh[j,i] = dFdhh[i,j]
                
        # Enforce penalties:

        dFdh  = dFdh  - N.dot(penalty_cov, h - penalty_mean)
        dFdhh = dFdhh - penalty_cov

        dh = L.solve(dFdhh, dFdh)
        h -= dh
        C = N.sum([h[i] * Q[i] for i in range(Q.shape[0])], axis=0)
        
        df = (dFdh * dh).sum()
        if N.fabs(df) < 1.0e-01:
            break

        _iter += 1
        if _iter >= niter:
            break

    return C, h, -dFdhh


if __name__ == "__main__":

    import numpy.random as R

    X = R.standard_normal((500,3))
    Q = N.array([N.identity(3), N.array([[0,1,0],[1,0,0],[0,0,1]]),
                 N.array([[1,0,0],[0,1,1],[0,1,1]])], N.float)
    
    print reml(N.dot(X.T,X), Q), 
