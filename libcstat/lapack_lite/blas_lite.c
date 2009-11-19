/*
NOTE: This is generated code. Look in Misc/lapack_lite for information on
      remaking this file.
*/
#include "f2c.h"

#ifdef HAVE_CONFIG
#include "config.h"
#else
extern doublereal dlamch_(char *);
#define EPSILON dlamch_("Epsilon")
#define SAFEMINIMUM dlamch_("Safe minimum")
#define PRECISION dlamch_("Precision")
#define BASE dlamch_("Base")
#endif

extern doublereal dlapy2_(doublereal *x, doublereal *y);



/* Table of constant values */

static doublereal c_b90 = 1.;
static integer c__1 = 1;

doublereal dasum_(integer *n, doublereal *dx, integer *incx)
{
    /* System generated locals */
    integer i__1, i__2;
    doublereal ret_val, d__1, d__2, d__3, d__4, d__5, d__6;

    /* Local variables */
    static integer i__, m;
    static doublereal dtemp;
    static integer nincx, mp1;


/*
    Purpose
    =======

       takes the sum of the absolute values.
       jack dongarra, linpack, 3/11/78.
       modified 3/93 to return if incx .le. 0.
       modified 12/3/93, array(1) declarations changed to array(*)
*/


    /* Parameter adjustments */
    --dx;

    /* Function Body */
    ret_val = 0.;
    dtemp = 0.;
    if (*n <= 0 || *incx <= 0) {
	return ret_val;
    }
    if (*incx == 1) {
	goto L20;
    }

/*        code for increment not equal to 1 */

    nincx = *n * *incx;
    i__1 = nincx;
    i__2 = *incx;
    for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {
	dtemp += (d__1 = dx[i__], abs(d__1));
/* L10: */
    }
    ret_val = dtemp;
    return ret_val;

/*
          code for increment equal to 1


          clean-up loop
*/

L20:
    m = *n % 6;
    if (m == 0) {
	goto L40;
    }
    i__2 = m;
    for (i__ = 1; i__ <= i__2; ++i__) {
	dtemp += (d__1 = dx[i__], abs(d__1));
/* L30: */
    }
    if (*n < 6) {
	goto L60;
    }
L40:
    mp1 = m + 1;
    i__2 = *n;
    for (i__ = mp1; i__ <= i__2; i__ += 6) {
	dtemp = dtemp + (d__1 = dx[i__], abs(d__1)) + (d__2 = dx[i__ + 1],
		abs(d__2)) + (d__3 = dx[i__ + 2], abs(d__3)) + (d__4 = dx[i__
		+ 3], abs(d__4)) + (d__5 = dx[i__ + 4], abs(d__5)) + (d__6 =
		dx[i__ + 5], abs(d__6));
/* L50: */
    }
L60:
    ret_val = dtemp;
    return ret_val;
} /* dasum_ */

/* Subroutine */ int daxpy_(integer *n, doublereal *da, doublereal *dx,
	integer *incx, doublereal *dy, integer *incy)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    static integer i__, m, ix, iy, mp1;


/*
    Purpose
    =======

       constant times a vector plus a vector.
       uses unrolled loops for increments equal to one.
       jack dongarra, linpack, 3/11/78.
       modified 12/3/93, array(1) declarations changed to array(*)
*/


    /* Parameter adjustments */
    --dy;
    --dx;

    /* Function Body */
    if (*n <= 0) {
	return 0;
    }
    if (*da == 0.) {
	return 0;
    }
    if (*incx == 1 && *incy == 1) {
	goto L20;
    }

/*
          code for unequal increments or equal increments
            not equal to 1
*/

    ix = 1;
    iy = 1;
    if (*incx < 0) {
	ix = (-(*n) + 1) * *incx + 1;
    }
    if (*incy < 0) {
	iy = (-(*n) + 1) * *incy + 1;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	dy[iy] += *da * dx[ix];
	ix += *incx;
	iy += *incy;
/* L10: */
    }
    return 0;

/*
          code for both increments equal to 1


          clean-up loop
*/

L20:
    m = *n % 4;
    if (m == 0) {
	goto L40;
    }
    i__1 = m;
    for (i__ = 1; i__ <= i__1; ++i__) {
	dy[i__] += *da * dx[i__];
/* L30: */
    }
    if (*n < 4) {
	return 0;
    }
L40:
    mp1 = m + 1;
    i__1 = *n;
    for (i__ = mp1; i__ <= i__1; i__ += 4) {
	dy[i__] += *da * dx[i__];
	dy[i__ + 1] += *da * dx[i__ + 1];
	dy[i__ + 2] += *da * dx[i__ + 2];
	dy[i__ + 3] += *da * dx[i__ + 3];
/* L50: */
    }
    return 0;
} /* daxpy_ */

doublereal dcabs1_(doublecomplex *z__)
{
    /* System generated locals */
    doublereal ret_val, d__1, d__2;

    /* Builtin functions */
    double d_imag(doublecomplex *);

/*
    Purpose
    =======

    DCABS1 computes absolute value of a double complex number
*/


    ret_val = (d__1 = z__->r, abs(d__1)) + (d__2 = d_imag(z__), abs(d__2));
    return ret_val;
} /* dcabs1_ */

/* Subroutine */ int dcopy_(integer *n, doublereal *dx, integer *incx,
	doublereal *dy, integer *incy)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    static integer i__, m, ix, iy, mp1;


/*
    Purpose
    =======

       copies a vector, x, to a vector, y.
       uses unrolled loops for increments equal to one.
       jack dongarra, linpack, 3/11/78.
       modified 12/3/93, array(1) declarations changed to array(*)
*/


    /* Parameter adjustments */
    --dy;
    --dx;

    /* Function Body */
    if (*n <= 0) {
	return 0;
    }
    if (*incx == 1 && *incy == 1) {
	goto L20;
    }

/*
          code for unequal increments or equal increments
            not equal to 1
*/

    ix = 1;
    iy = 1;
    if (*incx < 0) {
	ix = (-(*n) + 1) * *incx + 1;
    }
    if (*incy < 0) {
	iy = (-(*n) + 1) * *incy + 1;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	dy[iy] = dx[ix];
	ix += *incx;
	iy += *incy;
/* L10: */
    }
    return 0;

/*
          code for both increments equal to 1


          clean-up loop
*/

L20:
    m = *n % 7;
    if (m == 0) {
	goto L40;
    }
    i__1 = m;
    for (i__ = 1; i__ <= i__1; ++i__) {
	dy[i__] = dx[i__];
/* L30: */
    }
    if (*n < 7) {
	return 0;
    }
L40:
    mp1 = m + 1;
    i__1 = *n;
    for (i__ = mp1; i__ <= i__1; i__ += 7) {
	dy[i__] = dx[i__];
	dy[i__ + 1] = dx[i__ + 1];
	dy[i__ + 2] = dx[i__ + 2];
	dy[i__ + 3] = dx[i__ + 3];
	dy[i__ + 4] = dx[i__ + 4];
	dy[i__ + 5] = dx[i__ + 5];
	dy[i__ + 6] = dx[i__ + 6];
/* L50: */
    }
    return 0;
} /* dcopy_ */

doublereal ddot_(integer *n, doublereal *dx, integer *incx, doublereal *dy,
	integer *incy)
{
    /* System generated locals */
    integer i__1;
    doublereal ret_val;

    /* Local variables */
    static integer i__, m;
    static doublereal dtemp;
    static integer ix, iy, mp1;


/*
    Purpose
    =======

       forms the dot product of two vectors.
       uses unrolled loops for increments equal to one.
       jack dongarra, linpack, 3/11/78.
       modified 12/3/93, array(1) declarations changed to array(*)
*/


    /* Parameter adjustments */
    --dy;
    --dx;

    /* Function Body */
    ret_val = 0.;
    dtemp = 0.;
    if (*n <= 0) {
	return ret_val;
    }
    if (*incx == 1 && *incy == 1) {
	goto L20;
    }

/*
          code for unequal increments or equal increments
            not equal to 1
*/

    ix = 1;
    iy = 1;
    if (*incx < 0) {
	ix = (-(*n) + 1) * *incx + 1;
    }
    if (*incy < 0) {
	iy = (-(*n) + 1) * *incy + 1;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	dtemp += dx[ix] * dy[iy];
	ix += *incx;
	iy += *incy;
/* L10: */
    }
    ret_val = dtemp;
    return ret_val;

/*
          code for both increments equal to 1


          clean-up loop
*/

L20:
    m = *n % 5;
    if (m == 0) {
	goto L40;
    }
    i__1 = m;
    for (i__ = 1; i__ <= i__1; ++i__) {
	dtemp += dx[i__] * dy[i__];
/* L30: */
    }
    if (*n < 5) {
	goto L60;
    }
L40:
    mp1 = m + 1;
    i__1 = *n;
    for (i__ = mp1; i__ <= i__1; i__ += 5) {
	dtemp = dtemp + dx[i__] * dy[i__] + dx[i__ + 1] * dy[i__ + 1] + dx[
		i__ + 2] * dy[i__ + 2] + dx[i__ + 3] * dy[i__ + 3] + dx[i__ +
		4] * dy[i__ + 4];
/* L50: */
    }
L60:
    ret_val = dtemp;
    return ret_val;
} /* ddot_ */

/* Subroutine */ int dgemm_(char *transa, char *transb, integer *m, integer *
	n, integer *k, doublereal *alpha, doublereal *a, integer *lda,
	doublereal *b, integer *ldb, doublereal *beta, doublereal *c__,
	integer *ldc)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, c_dim1, c_offset, i__1, i__2,
	    i__3;

    /* Local variables */
    static integer info;
    static logical nota, notb;
    static doublereal temp;
    static integer i__, j, l, ncola;
    extern logical lsame_(char *, char *);
    static integer nrowa, nrowb;
    extern /* Subroutine */ int xerbla_(char *, integer *);


/*
    Purpose
    =======

    DGEMM  performs one of the matrix-matrix operations

       C := alpha*op( A )*op( B ) + beta*C,

    where  op( X ) is one of

       op( X ) = X   or   op( X ) = X',

    alpha and beta are scalars, and A, B and C are matrices, with op( A )
    an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.

    Arguments
    ==========

    TRANSA - CHARACTER*1.
             On entry, TRANSA specifies the form of op( A ) to be used in
             the matrix multiplication as follows:

                TRANSA = 'N' or 'n',  op( A ) = A.

                TRANSA = 'T' or 't',  op( A ) = A'.

                TRANSA = 'C' or 'c',  op( A ) = A'.

             Unchanged on exit.

    TRANSB - CHARACTER*1.
             On entry, TRANSB specifies the form of op( B ) to be used in
             the matrix multiplication as follows:

                TRANSB = 'N' or 'n',  op( B ) = B.

                TRANSB = 'T' or 't',  op( B ) = B'.

                TRANSB = 'C' or 'c',  op( B ) = B'.

             Unchanged on exit.

    M      - INTEGER.
             On entry,  M  specifies  the number  of rows  of the  matrix
             op( A )  and of the  matrix  C.  M  must  be at least  zero.
             Unchanged on exit.

    N      - INTEGER.
             On entry,  N  specifies the number  of columns of the matrix
             op( B ) and the number of columns of the matrix C. N must be
             at least zero.
             Unchanged on exit.

    K      - INTEGER.
             On entry,  K  specifies  the number of columns of the matrix
             op( A ) and the number of rows of the matrix op( B ). K must
             be at least  zero.
             Unchanged on exit.

    ALPHA  - DOUBLE PRECISION.
             On entry, ALPHA specifies the scalar alpha.
             Unchanged on exit.

    A      - DOUBLE PRECISION array of DIMENSION ( LDA, ka ), where ka is
             k  when  TRANSA = 'N' or 'n',  and is  m  otherwise.
             Before entry with  TRANSA = 'N' or 'n',  the leading  m by k
             part of the array  A  must contain the matrix  A,  otherwise
             the leading  k by m  part of the array  A  must contain  the
             matrix A.
             Unchanged on exit.

    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in the calling (sub) program. When  TRANSA = 'N' or 'n' then
             LDA must be at least  max( 1, m ), otherwise  LDA must be at
             least  max( 1, k ).
             Unchanged on exit.

    B      - DOUBLE PRECISION array of DIMENSION ( LDB, kb ), where kb is
             n  when  TRANSB = 'N' or 'n',  and is  k  otherwise.
             Before entry with  TRANSB = 'N' or 'n',  the leading  k by n
             part of the array  B  must contain the matrix  B,  otherwise
             the leading  n by k  part of the array  B  must contain  the
             matrix B.
             Unchanged on exit.

    LDB    - INTEGER.
             On entry, LDB specifies the first dimension of B as declared
             in the calling (sub) program. When  TRANSB = 'N' or 'n' then
             LDB must be at least  max( 1, k ), otherwise  LDB must be at
             least  max( 1, n ).
             Unchanged on exit.

    BETA   - DOUBLE PRECISION.
             On entry,  BETA  specifies the scalar  beta.  When  BETA  is
             supplied as zero then C need not be set on input.
             Unchanged on exit.

    C      - DOUBLE PRECISION array of DIMENSION ( LDC, n ).
             Before entry, the leading  m by n  part of the array  C must
             contain the matrix  C,  except when  beta  is zero, in which
             case C need not be set on entry.
             On exit, the array  C  is overwritten by the  m by n  matrix
             ( alpha*op( A )*op( B ) + beta*C ).

    LDC    - INTEGER.
             On entry, LDC specifies the first dimension of C as declared
             in  the  calling  (sub)  program.   LDC  must  be  at  least
             max( 1, m ).
             Unchanged on exit.


    Level 3 Blas routine.

    -- Written on 8-February-1989.
       Jack Dongarra, Argonne National Laboratory.
       Iain Duff, AERE Harwell.
       Jeremy Du Croz, Numerical Algorithms Group Ltd.
       Sven Hammarling, Numerical Algorithms Group Ltd.


       Set  NOTA  and  NOTB  as  true if  A  and  B  respectively are not
       transposed and set  NROWA, NCOLA and  NROWB  as the number of rows
       and  columns of  A  and the  number of  rows  of  B  respectively.
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1 * 1;
    b -= b_offset;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1 * 1;
    c__ -= c_offset;

    /* Function Body */
    nota = lsame_(transa, "N");
    notb = lsame_(transb, "N");
    if (nota) {
	nrowa = *m;
	ncola = *k;
    } else {
	nrowa = *k;
	ncola = *m;
    }
    if (notb) {
	nrowb = *k;
    } else {
	nrowb = *n;
    }

/*     Test the input parameters. */

    info = 0;
    if (! nota && ! lsame_(transa, "C") && ! lsame_(
	    transa, "T")) {
	info = 1;
    } else if (! notb && ! lsame_(transb, "C") && !
	    lsame_(transb, "T")) {
	info = 2;
    } else if (*m < 0) {
	info = 3;
    } else if (*n < 0) {
	info = 4;
    } else if (*k < 0) {
	info = 5;
    } else if (*lda < max(1,nrowa)) {
	info = 8;
    } else if (*ldb < max(1,nrowb)) {
	info = 10;
    } else if (*ldc < max(1,*m)) {
	info = 13;
    }
    if (info != 0) {
	xerbla_("DGEMM ", &info);
	return 0;
    }

/*     Quick return if possible. */

    if (*m == 0 || *n == 0 || (*alpha == 0. || *k == 0) && *beta == 1.) {
	return 0;
    }

/*     And if  alpha.eq.zero. */

    if (*alpha == 0.) {
	if (*beta == 0.) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = *m;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    c__[i__ + j * c_dim1] = 0.;
/* L10: */
		}
/* L20: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = *m;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
/* L30: */
		}
/* L40: */
	    }
	}
	return 0;
    }

/*     Start the operations. */

    if (notb) {
	if (nota) {

/*           Form  C := alpha*A*B + beta*C. */

	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		if (*beta == 0.) {
		    i__2 = *m;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = 0.;
/* L50: */
		    }
		} else if (*beta != 1.) {
		    i__2 = *m;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
/* L60: */
		    }
		}
		i__2 = *k;
		for (l = 1; l <= i__2; ++l) {
		    if (b[l + j * b_dim1] != 0.) {
			temp = *alpha * b[l + j * b_dim1];
			i__3 = *m;
			for (i__ = 1; i__ <= i__3; ++i__) {
			    c__[i__ + j * c_dim1] += temp * a[i__ + l *
				    a_dim1];
/* L70: */
			}
		    }
/* L80: */
		}
/* L90: */
	    }
	} else {

/*           Form  C := alpha*A'*B + beta*C */

	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = *m;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    temp = 0.;
		    i__3 = *k;
		    for (l = 1; l <= i__3; ++l) {
			temp += a[l + i__ * a_dim1] * b[l + j * b_dim1];
/* L100: */
		    }
		    if (*beta == 0.) {
			c__[i__ + j * c_dim1] = *alpha * temp;
		    } else {
			c__[i__ + j * c_dim1] = *alpha * temp + *beta * c__[
				i__ + j * c_dim1];
		    }
/* L110: */
		}
/* L120: */
	    }
	}
    } else {
	if (nota) {

/*           Form  C := alpha*A*B' + beta*C */

	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		if (*beta == 0.) {
		    i__2 = *m;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = 0.;
/* L130: */
		    }
		} else if (*beta != 1.) {
		    i__2 = *m;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
/* L140: */
		    }
		}
		i__2 = *k;
		for (l = 1; l <= i__2; ++l) {
		    if (b[j + l * b_dim1] != 0.) {
			temp = *alpha * b[j + l * b_dim1];
			i__3 = *m;
			for (i__ = 1; i__ <= i__3; ++i__) {
			    c__[i__ + j * c_dim1] += temp * a[i__ + l *
				    a_dim1];
/* L150: */
			}
		    }
/* L160: */
		}
/* L170: */
	    }
	} else {

/*           Form  C := alpha*A'*B' + beta*C */

	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = *m;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    temp = 0.;
		    i__3 = *k;
		    for (l = 1; l <= i__3; ++l) {
			temp += a[l + i__ * a_dim1] * b[j + l * b_dim1];
/* L180: */
		    }
		    if (*beta == 0.) {
			c__[i__ + j * c_dim1] = *alpha * temp;
		    } else {
			c__[i__ + j * c_dim1] = *alpha * temp + *beta * c__[
				i__ + j * c_dim1];
		    }
/* L190: */
		}
/* L200: */
	    }
	}
    }

    return 0;

/*     End of DGEMM . */

} /* dgemm_ */

/* Subroutine */ int dgemv_(char *trans, integer *m, integer *n, doublereal *
	alpha, doublereal *a, integer *lda, doublereal *x, integer *incx,
	doublereal *beta, doublereal *y, integer *incy)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;

    /* Local variables */
    static integer info;
    static doublereal temp;
    static integer lenx, leny, i__, j;
    extern logical lsame_(char *, char *);
    static integer ix, iy, jx, jy, kx, ky;
    extern /* Subroutine */ int xerbla_(char *, integer *);


/*
    Purpose
    =======

    DGEMV  performs one of the matrix-vector operations

       y := alpha*A*x + beta*y,   or   y := alpha*A'*x + beta*y,

    where alpha and beta are scalars, x and y are vectors and A is an
    m by n matrix.

    Arguments
    ==========

    TRANS  - CHARACTER*1.
             On entry, TRANS specifies the operation to be performed as
             follows:

                TRANS = 'N' or 'n'   y := alpha*A*x + beta*y.

                TRANS = 'T' or 't'   y := alpha*A'*x + beta*y.

                TRANS = 'C' or 'c'   y := alpha*A'*x + beta*y.

             Unchanged on exit.

    M      - INTEGER.
             On entry, M specifies the number of rows of the matrix A.
             M must be at least zero.
             Unchanged on exit.

    N      - INTEGER.
             On entry, N specifies the number of columns of the matrix A.
             N must be at least zero.
             Unchanged on exit.

    ALPHA  - DOUBLE PRECISION.
             On entry, ALPHA specifies the scalar alpha.
             Unchanged on exit.

    A      - DOUBLE PRECISION array of DIMENSION ( LDA, n ).
             Before entry, the leading m by n part of the array A must
             contain the matrix of coefficients.
             Unchanged on exit.

    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in the calling (sub) program. LDA must be at least
             max( 1, m ).
             Unchanged on exit.

    X      - DOUBLE PRECISION array of DIMENSION at least
             ( 1 + ( n - 1 )*abs( INCX ) ) when TRANS = 'N' or 'n'
             and at least
             ( 1 + ( m - 1 )*abs( INCX ) ) otherwise.
             Before entry, the incremented array X must contain the
             vector x.
             Unchanged on exit.

    INCX   - INTEGER.
             On entry, INCX specifies the increment for the elements of
             X. INCX must not be zero.
             Unchanged on exit.

    BETA   - DOUBLE PRECISION.
             On entry, BETA specifies the scalar beta. When BETA is
             supplied as zero then Y need not be set on input.
             Unchanged on exit.

    Y      - DOUBLE PRECISION array of DIMENSION at least
             ( 1 + ( m - 1 )*abs( INCY ) ) when TRANS = 'N' or 'n'
             and at least
             ( 1 + ( n - 1 )*abs( INCY ) ) otherwise.
             Before entry with BETA non-zero, the incremented array Y
             must contain the vector y. On exit, Y is overwritten by the
             updated vector y.

    INCY   - INTEGER.
             On entry, INCY specifies the increment for the elements of
             Y. INCY must not be zero.
             Unchanged on exit.


    Level 2 Blas routine.

    -- Written on 22-October-1986.
       Jack Dongarra, Argonne National Lab.
       Jeremy Du Croz, Nag Central Office.
       Sven Hammarling, Nag Central Office.
       Richard Hanson, Sandia National Labs.


       Test the input parameters.
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;
    --x;
    --y;

    /* Function Body */
    info = 0;
    if (! lsame_(trans, "N") && ! lsame_(trans, "T") && ! lsame_(trans, "C")
	    ) {
	info = 1;
    } else if (*m < 0) {
	info = 2;
    } else if (*n < 0) {
	info = 3;
    } else if (*lda < max(1,*m)) {
	info = 6;
    } else if (*incx == 0) {
	info = 8;
    } else if (*incy == 0) {
	info = 11;
    }
    if (info != 0) {
	xerbla_("DGEMV ", &info);
	return 0;
    }

/*     Quick return if possible. */

    if (*m == 0 || *n == 0 || *alpha == 0. && *beta == 1.) {
	return 0;
    }

/*
       Set  LENX  and  LENY, the lengths of the vectors x and y, and set
       up the start points in  X  and  Y.
*/

    if (lsame_(trans, "N")) {
	lenx = *n;
	leny = *m;
    } else {
	lenx = *m;
	leny = *n;
    }
    if (*incx > 0) {
	kx = 1;
    } else {
	kx = 1 - (lenx - 1) * *incx;
    }
    if (*incy > 0) {
	ky = 1;
    } else {
	ky = 1 - (leny - 1) * *incy;
    }

/*
       Start the operations. In this version the elements of A are
       accessed sequentially with one pass through A.

       First form  y := beta*y.
*/

    if (*beta != 1.) {
	if (*incy == 1) {
	    if (*beta == 0.) {
		i__1 = leny;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    y[i__] = 0.;
/* L10: */
		}
	    } else {
		i__1 = leny;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    y[i__] = *beta * y[i__];
/* L20: */
		}
	    }
	} else {
	    iy = ky;
	    if (*beta == 0.) {
		i__1 = leny;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    y[iy] = 0.;
		    iy += *incy;
/* L30: */
		}
	    } else {
		i__1 = leny;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    y[iy] = *beta * y[iy];
		    iy += *incy;
/* L40: */
		}
	    }
	}
    }
    if (*alpha == 0.) {
	return 0;
    }
    if (lsame_(trans, "N")) {

/*        Form  y := alpha*A*x + y. */

	jx = kx;
	if (*incy == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		if (x[jx] != 0.) {
		    temp = *alpha * x[jx];
		    i__2 = *m;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			y[i__] += temp * a[i__ + j * a_dim1];
/* L50: */
		    }
		}
		jx += *incx;
/* L60: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		if (x[jx] != 0.) {
		    temp = *alpha * x[jx];
		    iy = ky;
		    i__2 = *m;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			y[iy] += temp * a[i__ + j * a_dim1];
			iy += *incy;
/* L70: */
		    }
		}
		jx += *incx;
/* L80: */
	    }
	}
    } else {

/*        Form  y := alpha*A'*x + y. */

	jy = ky;
	if (*incx == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		temp = 0.;
		i__2 = *m;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    temp += a[i__ + j * a_dim1] * x[i__];
/* L90: */
		}
		y[jy] += *alpha * temp;
		jy += *incy;
/* L100: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		temp = 0.;
		ix = kx;
		i__2 = *m;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    temp += a[i__ + j * a_dim1] * x[ix];
		    ix += *incx;
/* L110: */
		}
		y[jy] += *alpha * temp;
		jy += *incy;
/* L120: */
	    }
	}
    }

    return 0;

/*     End of DGEMV . */

} /* dgemv_ */

/* Subroutine */ int dger_(integer *m, integer *n, doublereal *alpha,
	doublereal *x, integer *incx, doublereal *y, integer *incy,
	doublereal *a, integer *lda)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;

    /* Local variables */
    static integer info;
    static doublereal temp;
    static integer i__, j, ix, jy, kx;
    extern /* Subroutine */ int xerbla_(char *, integer *);


/*
    Purpose
    =======

    DGER   performs the rank 1 operation

       A := alpha*x*y' + A,

    where alpha is a scalar, x is an m element vector, y is an n element
    vector and A is an m by n matrix.

    Arguments
    ==========

    M      - INTEGER.
             On entry, M specifies the number of rows of the matrix A.
             M must be at least zero.
             Unchanged on exit.

    N      - INTEGER.
             On entry, N specifies the number of columns of the matrix A.
             N must be at least zero.
             Unchanged on exit.

    ALPHA  - DOUBLE PRECISION.
             On entry, ALPHA specifies the scalar alpha.
             Unchanged on exit.

    X      - DOUBLE PRECISION array of dimension at least
             ( 1 + ( m - 1 )*abs( INCX ) ).
             Before entry, the incremented array X must contain the m
             element vector x.
             Unchanged on exit.

    INCX   - INTEGER.
             On entry, INCX specifies the increment for the elements of
             X. INCX must not be zero.
             Unchanged on exit.

    Y      - DOUBLE PRECISION array of dimension at least
             ( 1 + ( n - 1 )*abs( INCY ) ).
             Before entry, the incremented array Y must contain the n
             element vector y.
             Unchanged on exit.

    INCY   - INTEGER.
             On entry, INCY specifies the increment for the elements of
             Y. INCY must not be zero.
             Unchanged on exit.

    A      - DOUBLE PRECISION array of DIMENSION ( LDA, n ).
             Before entry, the leading m by n part of the array A must
             contain the matrix of coefficients. On exit, A is
             overwritten by the updated matrix.

    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in the calling (sub) program. LDA must be at least
             max( 1, m ).
             Unchanged on exit.


    Level 2 Blas routine.

    -- Written on 22-October-1986.
       Jack Dongarra, Argonne National Lab.
       Jeremy Du Croz, Nag Central Office.
       Sven Hammarling, Nag Central Office.
       Richard Hanson, Sandia National Labs.


       Test the input parameters.
*/

    /* Parameter adjustments */
    --x;
    --y;
    a_dim1 = *lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;

    /* Function Body */
    info = 0;
    if (*m < 0) {
	info = 1;
    } else if (*n < 0) {
	info = 2;
    } else if (*incx == 0) {
	info = 5;
    } else if (*incy == 0) {
	info = 7;
    } else if (*lda < max(1,*m)) {
	info = 9;
    }
    if (info != 0) {
	xerbla_("DGER  ", &info);
	return 0;
    }

/*     Quick return if possible. */

    if (*m == 0 || *n == 0 || *alpha == 0.) {
	return 0;
    }

/*
       Start the operations. In this version the elements of A are
       accessed sequentially with one pass through A.
*/

    if (*incy > 0) {
	jy = 1;
    } else {
	jy = 1 - (*n - 1) * *incy;
    }
    if (*incx == 1) {
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    if (y[jy] != 0.) {
		temp = *alpha * y[jy];
		i__2 = *m;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    a[i__ + j * a_dim1] += x[i__] * temp;
/* L10: */
		}
	    }
	    jy += *incy;
/* L20: */
	}
    } else {
	if (*incx > 0) {
	    kx = 1;
	} else {
	    kx = 1 - (*m - 1) * *incx;
	}
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    if (y[jy] != 0.) {
		temp = *alpha * y[jy];
		ix = kx;
		i__2 = *m;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    a[i__ + j * a_dim1] += x[ix] * temp;
		    ix += *incx;
/* L30: */
		}
	    }
	    jy += *incy;
/* L40: */
	}
    }

    return 0;

/*     End of DGER  . */

} /* dger_ */

doublereal dnrm2_(integer *n, doublereal *x, integer *incx)
{
    /* System generated locals */
    integer i__1, i__2;
    doublereal ret_val, d__1;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    static doublereal norm, scale, absxi;
    static integer ix;
    static doublereal ssq;


/*
    Purpose
    =======

    DNRM2 returns the euclidean norm of a vector via the function
    name, so that

       DNRM2 := sqrt( x'*x )


    -- This version written on 25-October-1982.
       Modified on 14-October-1993 to inline the call to DLASSQ.
       Sven Hammarling, Nag Ltd.
*/


    /* Parameter adjustments */
    --x;

    /* Function Body */
    if (*n < 1 || *incx < 1) {
	norm = 0.;
    } else if (*n == 1) {
	norm = abs(x[1]);
    } else {
	scale = 0.;
	ssq = 1.;
/*
          The following loop is equivalent to this call to the LAPACK
          auxiliary routine:
          CALL DLASSQ( N, X, INCX, SCALE, SSQ )
*/

	i__1 = (*n - 1) * *incx + 1;
	i__2 = *incx;
	for (ix = 1; i__2 < 0 ? ix >= i__1 : ix <= i__1; ix += i__2) {
	    if (x[ix] != 0.) {
		absxi = (d__1 = x[ix], abs(d__1));
		if (scale < absxi) {
/* Computing 2nd power */
		    d__1 = scale / absxi;
		    ssq = ssq * (d__1 * d__1) + 1.;
		    scale = absxi;
		} else {
/* Computing 2nd power */
		    d__1 = absxi / scale;
		    ssq += d__1 * d__1;
		}
	    }
/* L10: */
	}
	norm = scale * sqrt(ssq);
    }

    ret_val = norm;
    return ret_val;

/*     End of DNRM2. */

} /* dnrm2_ */

/* Subroutine */ int drot_(integer *n, doublereal *dx, integer *incx,
	doublereal *dy, integer *incy, doublereal *c__, doublereal *s)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    static integer i__;
    static doublereal dtemp;
    static integer ix, iy;


/*
    Purpose
    =======

       applies a plane rotation.
       jack dongarra, linpack, 3/11/78.
       modified 12/3/93, array(1) declarations changed to array(*)
*/


    /* Parameter adjustments */
    --dy;
    --dx;

    /* Function Body */
    if (*n <= 0) {
	return 0;
    }
    if (*incx == 1 && *incy == 1) {
	goto L20;
    }

/*
         code for unequal increments or equal increments not equal
           to 1
*/

    ix = 1;
    iy = 1;
    if (*incx < 0) {
	ix = (-(*n) + 1) * *incx + 1;
    }
    if (*incy < 0) {
	iy = (-(*n) + 1) * *incy + 1;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	dtemp = *c__ * dx[ix] + *s * dy[iy];
	dy[iy] = *c__ * dy[iy] - *s * dx[ix];
	dx[ix] = dtemp;
	ix += *incx;
	iy += *incy;
/* L10: */
    }
    return 0;

/*       code for both increments equal to 1 */

L20:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	dtemp = *c__ * dx[i__] + *s * dy[i__];
	dy[i__] = *c__ * dy[i__] - *s * dx[i__];
	dx[i__] = dtemp;
/* L30: */
    }
    return 0;
} /* drot_ */

/* Subroutine */ int drotg_(doublereal *da, doublereal *db, doublereal *c__,
	doublereal *s)
{
    /* System generated locals */
    doublereal d__1, d__2;

    /* Builtin functions */
    double sqrt(doublereal), d_sign(doublereal *, doublereal *);

    /* Local variables */
    static doublereal r__, scale, z__, roe;


/*
    Purpose
    =======

       construct givens plane rotation.
       jack dongarra, linpack, 3/11/78.
*/


    roe = *db;
    if (abs(*da) > abs(*db)) {
	roe = *da;
    }
    scale = abs(*da) + abs(*db);
    if (scale != 0.) {
	goto L10;
    }
    *c__ = 1.;
    *s = 0.;
    r__ = 0.;
    z__ = 0.;
    goto L20;
L10:
/* Computing 2nd power */
    d__1 = *da / scale;
/* Computing 2nd power */
    d__2 = *db / scale;
    r__ = scale * sqrt(d__1 * d__1 + d__2 * d__2);
    r__ = d_sign(&c_b90, &roe) * r__;
    *c__ = *da / r__;
    *s = *db / r__;
    z__ = 1.;
    if (abs(*da) > abs(*db)) {
	z__ = *s;
    }
    if (abs(*db) >= abs(*da) && *c__ != 0.) {
	z__ = 1. / *c__;
    }
L20:
    *da = r__;
    *db = z__;
    return 0;
} /* drotg_ */

/* Subroutine */ int drotm_(integer *n, doublereal *dx, integer *incx,
	doublereal *dy, integer *incy, doublereal *dparam)
{
    /* Initialized data */

    static doublereal zero = 0.;
    static doublereal two = 2.;

    /* System generated locals */
    integer i__1, i__2;

    /* Local variables */
    static integer i__;
    static doublereal dflag, w, z__;
    static integer kx, ky, nsteps;
    static doublereal dh11, dh12, dh21, dh22;


/*
    Purpose
    =======

       APPLY THE MODIFIED GIVENS TRANSFORMATION, H, TO THE 2 BY N MATRIX

       (DX**T) , WHERE **T INDICATES TRANSPOSE. THE ELEMENTS OF DX ARE IN
       (DY**T)

       DX(LX+I*INCX), I = 0 TO N-1, WHERE LX = 1 IF INCX .GE. 0, ELSE
       LX = (-INCX)*N, AND SIMILARLY FOR SY USING LY AND INCY.
       WITH DPARAM(1)=DFLAG, H HAS ONE OF THE FOLLOWING FORMS..

       DFLAG=-1.D0     DFLAG=0.D0        DFLAG=1.D0     DFLAG=-2.D0

         (DH11  DH12)    (1.D0  DH12)    (DH11  1.D0)    (1.D0  0.D0)
       H=(          )    (          )    (          )    (          )
         (DH21  DH22),   (DH21  1.D0),   (-1.D0 DH22),   (0.D0  1.D0).
       SEE DROTMG FOR A DESCRIPTION OF DATA STORAGE IN DPARAM.

    Arguments
    =========

    N      (input) INTEGER
           number of elements in input vector(s)

    DX     (input/output) DOUBLE PRECISION array, dimension N
           double precision vector with 5 elements

    INCX   (input) INTEGER
           storage spacing between elements of DX

    DY     (input/output) DOUBLE PRECISION array, dimension N
           double precision vector with N elements

    INCY   (input) INTEGER
           storage spacing between elements of DY

    DPARAM (input/output)  DOUBLE PRECISION array, dimension 5
       DPARAM(1)=DFLAG
       DPARAM(2)=DH11
       DPARAM(3)=DH21
       DPARAM(4)=DH12
       DPARAM(5)=DH22

    =====================================================================
*/

    /* Parameter adjustments */
    --dparam;
    --dy;
    --dx;

    /* Function Body */

    dflag = dparam[1];
    if (*n <= 0 || dflag + two == zero) {
	goto L140;
    }
    if (! (*incx == *incy && *incx > 0)) {
	goto L70;
    }

    nsteps = *n * *incx;
    if (dflag < 0.) {
	goto L50;
    } else if (dflag == 0) {
	goto L10;
    } else {
	goto L30;
    }
L10:
    dh12 = dparam[4];
    dh21 = dparam[3];
    i__1 = nsteps;
    i__2 = *incx;
    for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {
	w = dx[i__];
	z__ = dy[i__];
	dx[i__] = w + z__ * dh12;
	dy[i__] = w * dh21 + z__;
/* L20: */
    }
    goto L140;
L30:
    dh11 = dparam[2];
    dh22 = dparam[5];
    i__2 = nsteps;
    i__1 = *incx;
    for (i__ = 1; i__1 < 0 ? i__ >= i__2 : i__ <= i__2; i__ += i__1) {
	w = dx[i__];
	z__ = dy[i__];
	dx[i__] = w * dh11 + z__;
	dy[i__] = -w + dh22 * z__;
/* L40: */
    }
    goto L140;
L50:
    dh11 = dparam[2];
    dh12 = dparam[4];
    dh21 = dparam[3];
    dh22 = dparam[5];
    i__1 = nsteps;
    i__2 = *incx;
    for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {
	w = dx[i__];
	z__ = dy[i__];
	dx[i__] = w * dh11 + z__ * dh12;
	dy[i__] = w * dh21 + z__ * dh22;
/* L60: */
    }
    goto L140;
L70:
    kx = 1;
    ky = 1;
    if (*incx < 0) {
	kx = (1 - *n) * *incx + 1;
    }
    if (*incy < 0) {
	ky = (1 - *n) * *incy + 1;
    }

    if (dflag < 0.) {
	goto L120;
    } else if (dflag == 0) {
	goto L80;
    } else {
	goto L100;
    }
L80:
    dh12 = dparam[4];
    dh21 = dparam[3];
    i__2 = *n;
    for (i__ = 1; i__ <= i__2; ++i__) {
	w = dx[kx];
	z__ = dy[ky];
	dx[kx] = w + z__ * dh12;
	dy[ky] = w * dh21 + z__;
	kx += *incx;
	ky += *incy;
/* L90: */
    }
    goto L140;
L100:
    dh11 = dparam[2];
    dh22 = dparam[5];
    i__2 = *n;
    for (i__ = 1; i__ <= i__2; ++i__) {
	w = dx[kx];
	z__ = dy[ky];
	dx[kx] = w * dh11 + z__;
	dy[ky] = -w + dh22 * z__;
	kx += *incx;
	ky += *incy;
/* L110: */
    }
    goto L140;
L120:
    dh11 = dparam[2];
    dh12 = dparam[4];
    dh21 = dparam[3];
    dh22 = dparam[5];
    i__2 = *n;
    for (i__ = 1; i__ <= i__2; ++i__) {
	w = dx[kx];
	z__ = dy[ky];
	dx[kx] = w * dh11 + z__ * dh12;
	dy[ky] = w * dh21 + z__ * dh22;
	kx += *incx;
	ky += *incy;
/* L130: */
    }
L140:
    return 0;
} /* drotm_ */

/* Subroutine */ int drotmg_(doublereal *dd1, doublereal *dd2, doublereal *
	dx1, doublereal *dy1, doublereal *dparam)
{
    /* Initialized data */

    static doublereal zero = 0.;
    static doublereal one = 1.;
    static doublereal two = 2.;
    static doublereal gam = 4096.;
    static doublereal gamsq = 16777216.;
    static doublereal rgamsq = 5.9604645e-8;

    /* Format strings */
    static char fmt_120[] = "";
    static char fmt_150[] = "";
    static char fmt_180[] = "";
    static char fmt_210[] = "";

    /* System generated locals */
    doublereal d__1;

    /* Local variables */
    static doublereal dflag, dtemp, du, dp1, dp2, dq1, dq2, dh11, dh12, dh21,
	    dh22;
    static integer igo;

    /* Assigned format variables */
    static char *igo_fmt;


/*
    Purpose
    =======

       CONSTRUCT THE MODIFIED GIVENS TRANSFORMATION MATRIX H WHICH ZEROS
       THE SECOND COMPONENT OF THE 2-VECTOR  (DSQRT(DD1)*DX1,DSQRT(DD2)*
       DY2)**T.
       WITH DPARAM(1)=DFLAG, H HAS ONE OF THE FOLLOWING FORMS..

       DFLAG=-1.D0     DFLAG=0.D0        DFLAG=1.D0     DFLAG=-2.D0

         (DH11  DH12)    (1.D0  DH12)    (DH11  1.D0)    (1.D0  0.D0)
       H=(          )    (          )    (          )    (          )
         (DH21  DH22),   (DH21  1.D0),   (-1.D0 DH22),   (0.D0  1.D0).
       LOCATIONS 2-4 OF DPARAM CONTAIN DH11, DH21, DH12, AND DH22
       RESPECTIVELY. (VALUES OF 1.D0, -1.D0, OR 0.D0 IMPLIED BY THE
       VALUE OF DPARAM(1) ARE NOT STORED IN DPARAM.)

       THE VALUES OF GAMSQ AND RGAMSQ SET IN THE DATA STATEMENT MAY BE
       INEXACT.  THIS IS OK AS THEY ARE ONLY USED FOR TESTING THE SIZE
       OF DD1 AND DD2.  ALL ACTUAL SCALING OF DATA IS DONE USING GAM.


    Arguments
    =========

    DD1    (input/output) DOUBLE PRECISION

    DD2    (input/output) DOUBLE PRECISION

    DX1    (input/output) DOUBLE PRECISION

    DY1    (input) DOUBLE PRECISION

    DPARAM (input/output)  DOUBLE PRECISION array, dimension 5
       DPARAM(1)=DFLAG
       DPARAM(2)=DH11
       DPARAM(3)=DH21
       DPARAM(4)=DH12
       DPARAM(5)=DH22

    =====================================================================
*/


    /* Parameter adjustments */
    --dparam;

    /* Function Body */
    if (! (*dd1 < zero)) {
	goto L10;
    }
/*       GO ZERO-H-D-AND-DX1.. */
    goto L60;
L10:
/*     CASE-DD1-NONNEGATIVE */
    dp2 = *dd2 * *dy1;
    if (! (dp2 == zero)) {
	goto L20;
    }
    dflag = -two;
    goto L260;
/*     REGULAR-CASE.. */
L20:
    dp1 = *dd1 * *dx1;
    dq2 = dp2 * *dy1;
    dq1 = dp1 * *dx1;

    if (! (abs(dq1) > abs(dq2))) {
	goto L40;
    }
    dh21 = -(*dy1) / *dx1;
    dh12 = dp2 / dp1;

    du = one - dh12 * dh21;

    if (! (du <= zero)) {
	goto L30;
    }
/*         GO ZERO-H-D-AND-DX1.. */
    goto L60;
L30:
    dflag = zero;
    *dd1 /= du;
    *dd2 /= du;
    *dx1 *= du;
/*         GO SCALE-CHECK.. */
    goto L100;
L40:
    if (! (dq2 < zero)) {
	goto L50;
    }
/*         GO ZERO-H-D-AND-DX1.. */
    goto L60;
L50:
    dflag = one;
    dh11 = dp1 / dp2;
    dh22 = *dx1 / *dy1;
    du = one + dh11 * dh22;
    dtemp = *dd2 / du;
    *dd2 = *dd1 / du;
    *dd1 = dtemp;
    *dx1 = *dy1 * du;
/*         GO SCALE-CHECK */
    goto L100;
/*     PROCEDURE..ZERO-H-D-AND-DX1.. */
L60:
    dflag = -one;
    dh11 = zero;
    dh12 = zero;
    dh21 = zero;
    dh22 = zero;

    *dd1 = zero;
    *dd2 = zero;
    *dx1 = zero;
/*         RETURN.. */
    goto L220;
/*     PROCEDURE..FIX-H.. */
L70:
    if (! (dflag >= zero)) {
	goto L90;
    }

    if (! (dflag == zero)) {
	goto L80;
    }
    dh11 = one;
    dh22 = one;
    dflag = -one;
    goto L90;
L80:
    dh21 = -one;
    dh12 = one;
    dflag = -one;
L90:
    switch (igo) {
	case 0: goto L120;
	case 1: goto L150;
	case 2: goto L180;
	case 3: goto L210;
    }
/*     PROCEDURE..SCALE-CHECK */
L100:
L110:
    if (! (*dd1 <= rgamsq)) {
	goto L130;
    }
    if (*dd1 == zero) {
	goto L160;
    }
    igo = 0;
    igo_fmt = fmt_120;
/*              FIX-H.. */
    goto L70;
L120:
/* Computing 2nd power */
    d__1 = gam;
    *dd1 *= d__1 * d__1;
    *dx1 /= gam;
    dh11 /= gam;
    dh12 /= gam;
    goto L110;
L130:
L140:
    if (! (*dd1 >= gamsq)) {
	goto L160;
    }
    igo = 1;
    igo_fmt = fmt_150;
/*              FIX-H.. */
    goto L70;
L150:
/* Computing 2nd power */
    d__1 = gam;
    *dd1 /= d__1 * d__1;
    *dx1 *= gam;
    dh11 *= gam;
    dh12 *= gam;
    goto L140;
L160:
L170:
    if (! (abs(*dd2) <= rgamsq)) {
	goto L190;
    }
    if (*dd2 == zero) {
	goto L220;
    }
    igo = 2;
    igo_fmt = fmt_180;
/*              FIX-H.. */
    goto L70;
L180:
/* Computing 2nd power */
    d__1 = gam;
    *dd2 *= d__1 * d__1;
    dh21 /= gam;
    dh22 /= gam;
    goto L170;
L190:
L200:
    if (! (abs(*dd2) >= gamsq)) {
	goto L220;
    }
    igo = 3;
    igo_fmt = fmt_210;
/*              FIX-H.. */
    goto L70;
L210:
/* Computing 2nd power */
    d__1 = gam;
    *dd2 /= d__1 * d__1;
    dh21 *= gam;
    dh22 *= gam;
    goto L200;
L220:
    if (dflag < 0.) {
	goto L250;
    } else if (dflag == 0) {
	goto L230;
    } else {
	goto L240;
    }
L230:
    dparam[3] = dh21;
    dparam[4] = dh12;
    goto L260;
L240:
    dparam[2] = dh11;
    dparam[5] = dh22;
    goto L260;
L250:
    dparam[2] = dh11;
    dparam[3] = dh21;
    dparam[4] = dh12;
    dparam[5] = dh22;
L260:
    dparam[1] = dflag;
    return 0;
} /* drotmg_ */

/* Subroutine */ int dscal_(integer *n, doublereal *da, doublereal *dx,
	integer *incx)
{
    /* System generated locals */
    integer i__1, i__2;

    /* Local variables */
    static integer i__, m, nincx, mp1;


/*
    Purpose
    =======
   *
       scales a vector by a constant.
       uses unrolled loops for increment equal to one.
       jack dongarra, linpack, 3/11/78.
       modified 3/93 to return if incx .le. 0.
       modified 12/3/93, array(1) declarations changed to array(*)
*/


    /* Parameter adjustments */
    --dx;

    /* Function Body */
    if (*n <= 0 || *incx <= 0) {
	return 0;
    }
    if (*incx == 1) {
	goto L20;
    }

/*        code for increment not equal to 1 */

    nincx = *n * *incx;
    i__1 = nincx;
    i__2 = *incx;
    for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {
	dx[i__] = *da * dx[i__];
/* L10: */
    }
    return 0;

/*
          code for increment equal to 1


          clean-up loop
*/

L20:
    m = *n % 5;
    if (m == 0) {
	goto L40;
    }
    i__2 = m;
    for (i__ = 1; i__ <= i__2; ++i__) {
	dx[i__] = *da * dx[i__];
/* L30: */
    }
    if (*n < 5) {
	return 0;
    }
L40:
    mp1 = m + 1;
    i__2 = *n;
    for (i__ = mp1; i__ <= i__2; i__ += 5) {
	dx[i__] = *da * dx[i__];
	dx[i__ + 1] = *da * dx[i__ + 1];
	dx[i__ + 2] = *da * dx[i__ + 2];
	dx[i__ + 3] = *da * dx[i__ + 3];
	dx[i__ + 4] = *da * dx[i__ + 4];
/* L50: */
    }
    return 0;
} /* dscal_ */

/* Subroutine */ int dswap_(integer *n, doublereal *dx, integer *incx,
	doublereal *dy, integer *incy)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    static integer i__, m;
    static doublereal dtemp;
    static integer ix, iy, mp1;


/*
    Purpose
    =======

       interchanges two vectors.
       uses unrolled loops for increments equal one.
       jack dongarra, linpack, 3/11/78.
       modified 12/3/93, array(1) declarations changed to array(*)
*/


    /* Parameter adjustments */
    --dy;
    --dx;

    /* Function Body */
    if (*n <= 0) {
	return 0;
    }
    if (*incx == 1 && *incy == 1) {
	goto L20;
    }

/*
         code for unequal increments or equal increments not equal
           to 1
*/

    ix = 1;
    iy = 1;
    if (*incx < 0) {
	ix = (-(*n) + 1) * *incx + 1;
    }
    if (*incy < 0) {
	iy = (-(*n) + 1) * *incy + 1;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	dtemp = dx[ix];
	dx[ix] = dy[iy];
	dy[iy] = dtemp;
	ix += *incx;
	iy += *incy;
/* L10: */
    }
    return 0;

/*
         code for both increments equal to 1


         clean-up loop
*/

L20:
    m = *n % 3;
    if (m == 0) {
	goto L40;
    }
    i__1 = m;
    for (i__ = 1; i__ <= i__1; ++i__) {
	dtemp = dx[i__];
	dx[i__] = dy[i__];
	dy[i__] = dtemp;
/* L30: */
    }
    if (*n < 3) {
	return 0;
    }
L40:
    mp1 = m + 1;
    i__1 = *n;
    for (i__ = mp1; i__ <= i__1; i__ += 3) {
	dtemp = dx[i__];
	dx[i__] = dy[i__];
	dy[i__] = dtemp;
	dtemp = dx[i__ + 1];
	dx[i__ + 1] = dy[i__ + 1];
	dy[i__ + 1] = dtemp;
	dtemp = dx[i__ + 2];
	dx[i__ + 2] = dy[i__ + 2];
	dy[i__ + 2] = dtemp;
/* L50: */
    }
    return 0;
} /* dswap_ */

/* Subroutine */ int dsymm_(char *side, char *uplo, integer *m, integer *n,
	doublereal *alpha, doublereal *a, integer *lda, doublereal *b,
	integer *ldb, doublereal *beta, doublereal *c__, integer *ldc)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, c_dim1, c_offset, i__1, i__2,
	    i__3;

    /* Local variables */
    static integer info;
    static doublereal temp1, temp2;
    static integer i__, j, k;
    extern logical lsame_(char *, char *);
    static integer nrowa;
    static logical upper;
    extern /* Subroutine */ int xerbla_(char *, integer *);


/*
    Purpose
    =======

    DSYMM  performs one of the matrix-matrix operations

       C := alpha*A*B + beta*C,

    or

       C := alpha*B*A + beta*C,

    where alpha and beta are scalars,  A is a symmetric matrix and  B and
    C are  m by n matrices.

    Arguments
    ==========

    SIDE   - CHARACTER*1.
             On entry,  SIDE  specifies whether  the  symmetric matrix  A
             appears on the  left or right  in the  operation as follows:

                SIDE = 'L' or 'l'   C := alpha*A*B + beta*C,

                SIDE = 'R' or 'r'   C := alpha*B*A + beta*C,

             Unchanged on exit.

    UPLO   - CHARACTER*1.
             On  entry,   UPLO  specifies  whether  the  upper  or  lower
             triangular  part  of  the  symmetric  matrix   A  is  to  be
             referenced as follows:

                UPLO = 'U' or 'u'   Only the upper triangular part of the
                                    symmetric matrix is to be referenced.

                UPLO = 'L' or 'l'   Only the lower triangular part of the
                                    symmetric matrix is to be referenced.

             Unchanged on exit.

    M      - INTEGER.
             On entry,  M  specifies the number of rows of the matrix  C.
             M  must be at least zero.
             Unchanged on exit.

    N      - INTEGER.
             On entry, N specifies the number of columns of the matrix C.
             N  must be at least zero.
             Unchanged on exit.

    ALPHA  - DOUBLE PRECISION.
             On entry, ALPHA specifies the scalar alpha.
             Unchanged on exit.

    A      - DOUBLE PRECISION array of DIMENSION ( LDA, ka ), where ka is
             m  when  SIDE = 'L' or 'l'  and is  n otherwise.
             Before entry  with  SIDE = 'L' or 'l',  the  m by m  part of
             the array  A  must contain the  symmetric matrix,  such that
             when  UPLO = 'U' or 'u', the leading m by m upper triangular
             part of the array  A  must contain the upper triangular part
             of the  symmetric matrix and the  strictly  lower triangular
             part of  A  is not referenced,  and when  UPLO = 'L' or 'l',
             the leading  m by m  lower triangular part  of the  array  A
             must  contain  the  lower triangular part  of the  symmetric
             matrix and the  strictly upper triangular part of  A  is not
             referenced.
             Before entry  with  SIDE = 'R' or 'r',  the  n by n  part of
             the array  A  must contain the  symmetric matrix,  such that
             when  UPLO = 'U' or 'u', the leading n by n upper triangular
             part of the array  A  must contain the upper triangular part
             of the  symmetric matrix and the  strictly  lower triangular
             part of  A  is not referenced,  and when  UPLO = 'L' or 'l',
             the leading  n by n  lower triangular part  of the  array  A
             must  contain  the  lower triangular part  of the  symmetric
             matrix and the  strictly upper triangular part of  A  is not
             referenced.
             Unchanged on exit.

    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in the calling (sub) program.  When  SIDE = 'L' or 'l'  then
             LDA must be at least  max( 1, m ), otherwise  LDA must be at
             least  max( 1, n ).
             Unchanged on exit.

    B      - DOUBLE PRECISION array of DIMENSION ( LDB, n ).
             Before entry, the leading  m by n part of the array  B  must
             contain the matrix B.
             Unchanged on exit.

    LDB    - INTEGER.
             On entry, LDB specifies the first dimension of B as declared
             in  the  calling  (sub)  program.   LDB  must  be  at  least
             max( 1, m ).
             Unchanged on exit.

    BETA   - DOUBLE PRECISION.
             On entry,  BETA  specifies the scalar  beta.  When  BETA  is
             supplied as zero then C need not be set on input.
             Unchanged on exit.

    C      - DOUBLE PRECISION array of DIMENSION ( LDC, n ).
             Before entry, the leading  m by n  part of the array  C must
             contain the matrix  C,  except when  beta  is zero, in which
             case C need not be set on entry.
             On exit, the array  C  is overwritten by the  m by n updated
             matrix.

    LDC    - INTEGER.
             On entry, LDC specifies the first dimension of C as declared
             in  the  calling  (sub)  program.   LDC  must  be  at  least
             max( 1, m ).
             Unchanged on exit.


    Level 3 Blas routine.

    -- Written on 8-February-1989.
       Jack Dongarra, Argonne National Laboratory.
       Iain Duff, AERE Harwell.
       Jeremy Du Croz, Numerical Algorithms Group Ltd.
       Sven Hammarling, Numerical Algorithms Group Ltd.


       Set NROWA as the number of rows of A.
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1 * 1;
    b -= b_offset;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1 * 1;
    c__ -= c_offset;

    /* Function Body */
    if (lsame_(side, "L")) {
	nrowa = *m;
    } else {
	nrowa = *n;
    }
    upper = lsame_(uplo, "U");

/*     Test the input parameters. */

    info = 0;
    if (! lsame_(side, "L") && ! lsame_(side, "R")) {
	info = 1;
    } else if (! upper && ! lsame_(uplo, "L")) {
	info = 2;
    } else if (*m < 0) {
	info = 3;
    } else if (*n < 0) {
	info = 4;
    } else if (*lda < max(1,nrowa)) {
	info = 7;
    } else if (*ldb < max(1,*m)) {
	info = 9;
    } else if (*ldc < max(1,*m)) {
	info = 12;
    }
    if (info != 0) {
	xerbla_("DSYMM ", &info);
	return 0;
    }

/*     Quick return if possible. */

    if (*m == 0 || *n == 0 || *alpha == 0. && *beta == 1.) {
	return 0;
    }

/*     And when  alpha.eq.zero. */

    if (*alpha == 0.) {
	if (*beta == 0.) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = *m;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    c__[i__ + j * c_dim1] = 0.;
/* L10: */
		}
/* L20: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = *m;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
/* L30: */
		}
/* L40: */
	    }
	}
	return 0;
    }

/*     Start the operations. */

    if (lsame_(side, "L")) {

/*        Form  C := alpha*A*B + beta*C. */

	if (upper) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = *m;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    temp1 = *alpha * b[i__ + j * b_dim1];
		    temp2 = 0.;
		    i__3 = i__ - 1;
		    for (k = 1; k <= i__3; ++k) {
			c__[k + j * c_dim1] += temp1 * a[k + i__ * a_dim1];
			temp2 += b[k + j * b_dim1] * a[k + i__ * a_dim1];
/* L50: */
		    }
		    if (*beta == 0.) {
			c__[i__ + j * c_dim1] = temp1 * a[i__ + i__ * a_dim1]
				+ *alpha * temp2;
		    } else {
			c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1]
				+ temp1 * a[i__ + i__ * a_dim1] + *alpha *
				temp2;
		    }
/* L60: */
		}
/* L70: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		for (i__ = *m; i__ >= 1; --i__) {
		    temp1 = *alpha * b[i__ + j * b_dim1];
		    temp2 = 0.;
		    i__2 = *m;
		    for (k = i__ + 1; k <= i__2; ++k) {
			c__[k + j * c_dim1] += temp1 * a[k + i__ * a_dim1];
			temp2 += b[k + j * b_dim1] * a[k + i__ * a_dim1];
/* L80: */
		    }
		    if (*beta == 0.) {
			c__[i__ + j * c_dim1] = temp1 * a[i__ + i__ * a_dim1]
				+ *alpha * temp2;
		    } else {
			c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1]
				+ temp1 * a[i__ + i__ * a_dim1] + *alpha *
				temp2;
		    }
/* L90: */
		}
/* L100: */
	    }
	}
    } else {

/*        Form  C := alpha*B*A + beta*C. */

	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    temp1 = *alpha * a[j + j * a_dim1];
	    if (*beta == 0.) {
		i__2 = *m;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    c__[i__ + j * c_dim1] = temp1 * b[i__ + j * b_dim1];
/* L110: */
		}
	    } else {
		i__2 = *m;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1] +
			    temp1 * b[i__ + j * b_dim1];
/* L120: */
		}
	    }
	    i__2 = j - 1;
	    for (k = 1; k <= i__2; ++k) {
		if (upper) {
		    temp1 = *alpha * a[k + j * a_dim1];
		} else {
		    temp1 = *alpha * a[j + k * a_dim1];
		}
		i__3 = *m;
		for (i__ = 1; i__ <= i__3; ++i__) {
		    c__[i__ + j * c_dim1] += temp1 * b[i__ + k * b_dim1];
/* L130: */
		}
/* L140: */
	    }
	    i__2 = *n;
	    for (k = j + 1; k <= i__2; ++k) {
		if (upper) {
		    temp1 = *alpha * a[j + k * a_dim1];
		} else {
		    temp1 = *alpha * a[k + j * a_dim1];
		}
		i__3 = *m;
		for (i__ = 1; i__ <= i__3; ++i__) {
		    c__[i__ + j * c_dim1] += temp1 * b[i__ + k * b_dim1];
/* L150: */
		}
/* L160: */
	    }
/* L170: */
	}
    }

    return 0;

/*     End of DSYMM . */

} /* dsymm_ */

/* Subroutine */ int dsymv_(char *uplo, integer *n, doublereal *alpha,
	doublereal *a, integer *lda, doublereal *x, integer *incx, doublereal
	*beta, doublereal *y, integer *incy)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;

    /* Local variables */
    static integer info;
    static doublereal temp1, temp2;
    static integer i__, j;
    extern logical lsame_(char *, char *);
    static integer ix, iy, jx, jy, kx, ky;
    extern /* Subroutine */ int xerbla_(char *, integer *);


/*
    Purpose
    =======

    DSYMV  performs the matrix-vector  operation

       y := alpha*A*x + beta*y,

    where alpha and beta are scalars, x and y are n element vectors and
    A is an n by n symmetric matrix.

    Arguments
    ==========

    UPLO   - CHARACTER*1.
             On entry, UPLO specifies whether the upper or lower
             triangular part of the array A is to be referenced as
             follows:

                UPLO = 'U' or 'u'   Only the upper triangular part of A
                                    is to be referenced.

                UPLO = 'L' or 'l'   Only the lower triangular part of A
                                    is to be referenced.

             Unchanged on exit.

    N      - INTEGER.
             On entry, N specifies the order of the matrix A.
             N must be at least zero.
             Unchanged on exit.

    ALPHA  - DOUBLE PRECISION.
             On entry, ALPHA specifies the scalar alpha.
             Unchanged on exit.

    A      - DOUBLE PRECISION array of DIMENSION ( LDA, n ).
             Before entry with  UPLO = 'U' or 'u', the leading n by n
             upper triangular part of the array A must contain the upper
             triangular part of the symmetric matrix and the strictly
             lower triangular part of A is not referenced.
             Before entry with UPLO = 'L' or 'l', the leading n by n
             lower triangular part of the array A must contain the lower
             triangular part of the symmetric matrix and the strictly
             upper triangular part of A is not referenced.
             Unchanged on exit.

    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in the calling (sub) program. LDA must be at least
             max( 1, n ).
             Unchanged on exit.

    X      - DOUBLE PRECISION array of dimension at least
             ( 1 + ( n - 1 )*abs( INCX ) ).
             Before entry, the incremented array X must contain the n
             element vector x.
             Unchanged on exit.

    INCX   - INTEGER.
             On entry, INCX specifies the increment for the elements of
             X. INCX must not be zero.
             Unchanged on exit.

    BETA   - DOUBLE PRECISION.
             On entry, BETA specifies the scalar beta. When BETA is
             supplied as zero then Y need not be set on input.
             Unchanged on exit.

    Y      - DOUBLE PRECISION array of dimension at least
             ( 1 + ( n - 1 )*abs( INCY ) ).
             Before entry, the incremented array Y must contain the n
             element vector y. On exit, Y is overwritten by the updated
             vector y.

    INCY   - INTEGER.
             On entry, INCY specifies the increment for the elements of
             Y. INCY must not be zero.
             Unchanged on exit.


    Level 2 Blas routine.

    -- Written on 22-October-1986.
       Jack Dongarra, Argonne National Lab.
       Jeremy Du Croz, Nag Central Office.
       Sven Hammarling, Nag Central Office.
       Richard Hanson, Sandia National Labs.


       Test the input parameters.
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;
    --x;
    --y;

    /* Function Body */
    info = 0;
    if (! lsame_(uplo, "U") && ! lsame_(uplo, "L")) {
	info = 1;
    } else if (*n < 0) {
	info = 2;
    } else if (*lda < max(1,*n)) {
	info = 5;
    } else if (*incx == 0) {
	info = 7;
    } else if (*incy == 0) {
	info = 10;
    }
    if (info != 0) {
	xerbla_("DSYMV ", &info);
	return 0;
    }

/*     Quick return if possible. */

    if (*n == 0 || *alpha == 0. && *beta == 1.) {
	return 0;
    }

/*     Set up the start points in  X  and  Y. */

    if (*incx > 0) {
	kx = 1;
    } else {
	kx = 1 - (*n - 1) * *incx;
    }
    if (*incy > 0) {
	ky = 1;
    } else {
	ky = 1 - (*n - 1) * *incy;
    }

/*
       Start the operations. In this version the elements of A are
       accessed sequentially with one pass through the triangular part
       of A.

       First form  y := beta*y.
*/

    if (*beta != 1.) {
	if (*incy == 1) {
	    if (*beta == 0.) {
		i__1 = *n;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    y[i__] = 0.;
/* L10: */
		}
	    } else {
		i__1 = *n;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    y[i__] = *beta * y[i__];
/* L20: */
		}
	    }
	} else {
	    iy = ky;
	    if (*beta == 0.) {
		i__1 = *n;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    y[iy] = 0.;
		    iy += *incy;
/* L30: */
		}
	    } else {
		i__1 = *n;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    y[iy] = *beta * y[iy];
		    iy += *incy;
/* L40: */
		}
	    }
	}
    }
    if (*alpha == 0.) {
	return 0;
    }
    if (lsame_(uplo, "U")) {

/*        Form  y  when A is stored in upper triangle. */

	if (*incx == 1 && *incy == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		temp1 = *alpha * x[j];
		temp2 = 0.;
		i__2 = j - 1;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    y[i__] += temp1 * a[i__ + j * a_dim1];
		    temp2 += a[i__ + j * a_dim1] * x[i__];
/* L50: */
		}
		y[j] = y[j] + temp1 * a[j + j * a_dim1] + *alpha * temp2;
/* L60: */
	    }
	} else {
	    jx = kx;
	    jy = ky;
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		temp1 = *alpha * x[jx];
		temp2 = 0.;
		ix = kx;
		iy = ky;
		i__2 = j - 1;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    y[iy] += temp1 * a[i__ + j * a_dim1];
		    temp2 += a[i__ + j * a_dim1] * x[ix];
		    ix += *incx;
		    iy += *incy;
/* L70: */
		}
		y[jy] = y[jy] + temp1 * a[j + j * a_dim1] + *alpha * temp2;
		jx += *incx;
		jy += *incy;
/* L80: */
	    }
	}
    } else {

/*        Form  y  when A is stored in lower triangle. */

	if (*incx == 1 && *incy == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		temp1 = *alpha * x[j];
		temp2 = 0.;
		y[j] += temp1 * a[j + j * a_dim1];
		i__2 = *n;
		for (i__ = j + 1; i__ <= i__2; ++i__) {
		    y[i__] += temp1 * a[i__ + j * a_dim1];
		    temp2 += a[i__ + j * a_dim1] * x[i__];
/* L90: */
		}
		y[j] += *alpha * temp2;
/* L100: */
	    }
	} else {
	    jx = kx;
	    jy = ky;
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		temp1 = *alpha * x[jx];
		temp2 = 0.;
		y[jy] += temp1 * a[j + j * a_dim1];
		ix = jx;
		iy = jy;
		i__2 = *n;
		for (i__ = j + 1; i__ <= i__2; ++i__) {
		    ix += *incx;
		    iy += *incy;
		    y[iy] += temp1 * a[i__ + j * a_dim1];
		    temp2 += a[i__ + j * a_dim1] * x[ix];
/* L110: */
		}
		y[jy] += *alpha * temp2;
		jx += *incx;
		jy += *incy;
/* L120: */
	    }
	}
    }

    return 0;

/*     End of DSYMV . */

} /* dsymv_ */

/* Subroutine */ int dsyr_(char *uplo, integer *n, doublereal *alpha,
	doublereal *x, integer *incx, doublereal *a, integer *lda)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;

    /* Local variables */
    static integer info;
    static doublereal temp;
    static integer i__, j;
    extern logical lsame_(char *, char *);
    static integer ix, jx, kx;
    extern /* Subroutine */ int xerbla_(char *, integer *);


/*
    Purpose
    =======

    DSYR   performs the symmetric rank 1 operation

       A := alpha*x*x' + A,

    where alpha is a real scalar, x is an n element vector and A is an
    n by n symmetric matrix.

    Arguments
    ==========

    UPLO   - CHARACTER*1.
             On entry, UPLO specifies whether the upper or lower
             triangular part of the array A is to be referenced as
             follows:

                UPLO = 'U' or 'u'   Only the upper triangular part of A
                                    is to be referenced.

                UPLO = 'L' or 'l'   Only the lower triangular part of A
                                    is to be referenced.

             Unchanged on exit.

    N      - INTEGER.
             On entry, N specifies the order of the matrix A.
             N must be at least zero.
             Unchanged on exit.

    ALPHA  - DOUBLE PRECISION.
             On entry, ALPHA specifies the scalar alpha.
             Unchanged on exit.

    X      - DOUBLE PRECISION array of dimension at least
             ( 1 + ( n - 1 )*abs( INCX ) ).
             Before entry, the incremented array X must contain the n
             element vector x.
             Unchanged on exit.

    INCX   - INTEGER.
             On entry, INCX specifies the increment for the elements of
             X. INCX must not be zero.
             Unchanged on exit.

    A      - DOUBLE PRECISION array of DIMENSION ( LDA, n ).
             Before entry with  UPLO = 'U' or 'u', the leading n by n
             upper triangular part of the array A must contain the upper
             triangular part of the symmetric matrix and the strictly
             lower triangular part of A is not referenced. On exit, the
             upper triangular part of the array A is overwritten by the
             upper triangular part of the updated matrix.
             Before entry with UPLO = 'L' or 'l', the leading n by n
             lower triangular part of the array A must contain the lower
             triangular part of the symmetric matrix and the strictly
             upper triangular part of A is not referenced. On exit, the
             lower triangular part of the array A is overwritten by the
             lower triangular part of the updated matrix.

    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in the calling (sub) program. LDA must be at least
             max( 1, n ).
             Unchanged on exit.


    Level 2 Blas routine.

    -- Written on 22-October-1986.
       Jack Dongarra, Argonne National Lab.
       Jeremy Du Croz, Nag Central Office.
       Sven Hammarling, Nag Central Office.
       Richard Hanson, Sandia National Labs.


       Test the input parameters.
*/

    /* Parameter adjustments */
    --x;
    a_dim1 = *lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;

    /* Function Body */
    info = 0;
    if (! lsame_(uplo, "U") && ! lsame_(uplo, "L")) {
	info = 1;
    } else if (*n < 0) {
	info = 2;
    } else if (*incx == 0) {
	info = 5;
    } else if (*lda < max(1,*n)) {
	info = 7;
    }
    if (info != 0) {
	xerbla_("DSYR  ", &info);
	return 0;
    }

/*     Quick return if possible. */

    if (*n == 0 || *alpha == 0.) {
	return 0;
    }

/*     Set the start point in X if the increment is not unity. */

    if (*incx <= 0) {
	kx = 1 - (*n - 1) * *incx;
    } else if (*incx != 1) {
	kx = 1;
    }

/*
       Start the operations. In this version the elements of A are
       accessed sequentially with one pass through the triangular part
       of A.
*/

    if (lsame_(uplo, "U")) {

/*        Form  A  when A is stored in upper triangle. */

	if (*incx == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		if (x[j] != 0.) {
		    temp = *alpha * x[j];
		    i__2 = j;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			a[i__ + j * a_dim1] += x[i__] * temp;
/* L10: */
		    }
		}
/* L20: */
	    }
	} else {
	    jx = kx;
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		if (x[jx] != 0.) {
		    temp = *alpha * x[jx];
		    ix = kx;
		    i__2 = j;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			a[i__ + j * a_dim1] += x[ix] * temp;
			ix += *incx;
/* L30: */
		    }
		}
		jx += *incx;
/* L40: */
	    }
	}
    } else {

/*        Form  A  when A is stored in lower triangle. */

	if (*incx == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		if (x[j] != 0.) {
		    temp = *alpha * x[j];
		    i__2 = *n;
		    for (i__ = j; i__ <= i__2; ++i__) {
			a[i__ + j * a_dim1] += x[i__] * temp;
/* L50: */
		    }
		}
/* L60: */
	    }
	} else {
	    jx = kx;
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		if (x[jx] != 0.) {
		    temp = *alpha * x[jx];
		    ix = jx;
		    i__2 = *n;
		    for (i__ = j; i__ <= i__2; ++i__) {
			a[i__ + j * a_dim1] += x[ix] * temp;
			ix += *incx;
/* L70: */
		    }
		}
		jx += *incx;
/* L80: */
	    }
	}
    }

    return 0;

/*     End of DSYR  . */

} /* dsyr_ */

/* Subroutine */ int dsyr2_(char *uplo, integer *n, doublereal *alpha,
	doublereal *x, integer *incx, doublereal *y, integer *incy,
	doublereal *a, integer *lda)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;

    /* Local variables */
    static integer info;
    static doublereal temp1, temp2;
    static integer i__, j;
    extern logical lsame_(char *, char *);
    static integer ix, iy, jx, jy, kx, ky;
    extern /* Subroutine */ int xerbla_(char *, integer *);


/*
    Purpose
    =======

    DSYR2  performs the symmetric rank 2 operation

       A := alpha*x*y' + alpha*y*x' + A,

    where alpha is a scalar, x and y are n element vectors and A is an n
    by n symmetric matrix.

    Arguments
    ==========

    UPLO   - CHARACTER*1.
             On entry, UPLO specifies whether the upper or lower
             triangular part of the array A is to be referenced as
             follows:

                UPLO = 'U' or 'u'   Only the upper triangular part of A
                                    is to be referenced.

                UPLO = 'L' or 'l'   Only the lower triangular part of A
                                    is to be referenced.

             Unchanged on exit.

    N      - INTEGER.
             On entry, N specifies the order of the matrix A.
             N must be at least zero.
             Unchanged on exit.

    ALPHA  - DOUBLE PRECISION.
             On entry, ALPHA specifies the scalar alpha.
             Unchanged on exit.

    X      - DOUBLE PRECISION array of dimension at least
             ( 1 + ( n - 1 )*abs( INCX ) ).
             Before entry, the incremented array X must contain the n
             element vector x.
             Unchanged on exit.

    INCX   - INTEGER.
             On entry, INCX specifies the increment for the elements of
             X. INCX must not be zero.
             Unchanged on exit.

    Y      - DOUBLE PRECISION array of dimension at least
             ( 1 + ( n - 1 )*abs( INCY ) ).
             Before entry, the incremented array Y must contain the n
             element vector y.
             Unchanged on exit.

    INCY   - INTEGER.
             On entry, INCY specifies the increment for the elements of
             Y. INCY must not be zero.
             Unchanged on exit.

    A      - DOUBLE PRECISION array of DIMENSION ( LDA, n ).
             Before entry with  UPLO = 'U' or 'u', the leading n by n
             upper triangular part of the array A must contain the upper
             triangular part of the symmetric matrix and the strictly
             lower triangular part of A is not referenced. On exit, the
             upper triangular part of the array A is overwritten by the
             upper triangular part of the updated matrix.
             Before entry with UPLO = 'L' or 'l', the leading n by n
             lower triangular part of the array A must contain the lower
             triangular part of the symmetric matrix and the strictly
             upper triangular part of A is not referenced. On exit, the
             lower triangular part of the array A is overwritten by the
             lower triangular part of the updated matrix.

    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in the calling (sub) program. LDA must be at least
             max( 1, n ).
             Unchanged on exit.


    Level 2 Blas routine.

    -- Written on 22-October-1986.
       Jack Dongarra, Argonne National Lab.
       Jeremy Du Croz, Nag Central Office.
       Sven Hammarling, Nag Central Office.
       Richard Hanson, Sandia National Labs.


       Test the input parameters.
*/

    /* Parameter adjustments */
    --x;
    --y;
    a_dim1 = *lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;

    /* Function Body */
    info = 0;
    if (! lsame_(uplo, "U") && ! lsame_(uplo, "L")) {
	info = 1;
    } else if (*n < 0) {
	info = 2;
    } else if (*incx == 0) {
	info = 5;
    } else if (*incy == 0) {
	info = 7;
    } else if (*lda < max(1,*n)) {
	info = 9;
    }
    if (info != 0) {
	xerbla_("DSYR2 ", &info);
	return 0;
    }

/*     Quick return if possible. */

    if (*n == 0 || *alpha == 0.) {
	return 0;
    }

/*
       Set up the start points in X and Y if the increments are not both
       unity.
*/

    if (*incx != 1 || *incy != 1) {
	if (*incx > 0) {
	    kx = 1;
	} else {
	    kx = 1 - (*n - 1) * *incx;
	}
	if (*incy > 0) {
	    ky = 1;
	} else {
	    ky = 1 - (*n - 1) * *incy;
	}
	jx = kx;
	jy = ky;
    }

/*
       Start the operations. In this version the elements of A are
       accessed sequentially with one pass through the triangular part
       of A.
*/

    if (lsame_(uplo, "U")) {

/*        Form  A  when A is stored in the upper triangle. */

	if (*incx == 1 && *incy == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		if (x[j] != 0. || y[j] != 0.) {
		    temp1 = *alpha * y[j];
		    temp2 = *alpha * x[j];
		    i__2 = j;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			a[i__ + j * a_dim1] = a[i__ + j * a_dim1] + x[i__] *
				temp1 + y[i__] * temp2;
/* L10: */
		    }
		}
/* L20: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		if (x[jx] != 0. || y[jy] != 0.) {
		    temp1 = *alpha * y[jy];
		    temp2 = *alpha * x[jx];
		    ix = kx;
		    iy = ky;
		    i__2 = j;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			a[i__ + j * a_dim1] = a[i__ + j * a_dim1] + x[ix] *
				temp1 + y[iy] * temp2;
			ix += *incx;
			iy += *incy;
/* L30: */
		    }
		}
		jx += *incx;
		jy += *incy;
/* L40: */
	    }
	}
    } else {

/*        Form  A  when A is stored in the lower triangle. */

	if (*incx == 1 && *incy == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		if (x[j] != 0. || y[j] != 0.) {
		    temp1 = *alpha * y[j];
		    temp2 = *alpha * x[j];
		    i__2 = *n;
		    for (i__ = j; i__ <= i__2; ++i__) {
			a[i__ + j * a_dim1] = a[i__ + j * a_dim1] + x[i__] *
				temp1 + y[i__] * temp2;
/* L50: */
		    }
		}
/* L60: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		if (x[jx] != 0. || y[jy] != 0.) {
		    temp1 = *alpha * y[jy];
		    temp2 = *alpha * x[jx];
		    ix = jx;
		    iy = jy;
		    i__2 = *n;
		    for (i__ = j; i__ <= i__2; ++i__) {
			a[i__ + j * a_dim1] = a[i__ + j * a_dim1] + x[ix] *
				temp1 + y[iy] * temp2;
			ix += *incx;
			iy += *incy;
/* L70: */
		    }
		}
		jx += *incx;
		jy += *incy;
/* L80: */
	    }
	}
    }

    return 0;

/*     End of DSYR2 . */

} /* dsyr2_ */

/* Subroutine */ int dsyr2k_(char *uplo, char *trans, integer *n, integer *k,
	doublereal *alpha, doublereal *a, integer *lda, doublereal *b,
	integer *ldb, doublereal *beta, doublereal *c__, integer *ldc)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, c_dim1, c_offset, i__1, i__2,
	    i__3;

    /* Local variables */
    static integer info;
    static doublereal temp1, temp2;
    static integer i__, j, l;
    extern logical lsame_(char *, char *);
    static integer nrowa;
    static logical upper;
    extern /* Subroutine */ int xerbla_(char *, integer *);


/*
    Purpose
    =======

    DSYR2K  performs one of the symmetric rank 2k operations

       C := alpha*A*B' + alpha*B*A' + beta*C,

    or

       C := alpha*A'*B + alpha*B'*A + beta*C,

    where  alpha and beta  are scalars, C is an  n by n  symmetric matrix
    and  A and B  are  n by k  matrices  in the  first  case  and  k by n
    matrices in the second case.

    Arguments
    ==========

    UPLO   - CHARACTER*1.
             On  entry,   UPLO  specifies  whether  the  upper  or  lower
             triangular  part  of the  array  C  is to be  referenced  as
             follows:

                UPLO = 'U' or 'u'   Only the  upper triangular part of  C
                                    is to be referenced.

                UPLO = 'L' or 'l'   Only the  lower triangular part of  C
                                    is to be referenced.

             Unchanged on exit.

    TRANS  - CHARACTER*1.
             On entry,  TRANS  specifies the operation to be performed as
             follows:

                TRANS = 'N' or 'n'   C := alpha*A*B' + alpha*B*A' +
                                          beta*C.

                TRANS = 'T' or 't'   C := alpha*A'*B + alpha*B'*A +
                                          beta*C.

                TRANS = 'C' or 'c'   C := alpha*A'*B + alpha*B'*A +
                                          beta*C.

             Unchanged on exit.

    N      - INTEGER.
             On entry,  N specifies the order of the matrix C.  N must be
             at least zero.
             Unchanged on exit.

    K      - INTEGER.
             On entry with  TRANS = 'N' or 'n',  K  specifies  the number
             of  columns  of the  matrices  A and B,  and on  entry  with
             TRANS = 'T' or 't' or 'C' or 'c',  K  specifies  the  number
             of rows of the matrices  A and B.  K must be at least  zero.
             Unchanged on exit.

    ALPHA  - DOUBLE PRECISION.
             On entry, ALPHA specifies the scalar alpha.
             Unchanged on exit.

    A      - DOUBLE PRECISION array of DIMENSION ( LDA, ka ), where ka is
             k  when  TRANS = 'N' or 'n',  and is  n  otherwise.
             Before entry with  TRANS = 'N' or 'n',  the  leading  n by k
             part of the array  A  must contain the matrix  A,  otherwise
             the leading  k by n  part of the array  A  must contain  the
             matrix A.
             Unchanged on exit.

    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in  the  calling  (sub)  program.   When  TRANS = 'N' or 'n'
             then  LDA must be at least  max( 1, n ), otherwise  LDA must
             be at least  max( 1, k ).
             Unchanged on exit.

    B      - DOUBLE PRECISION array of DIMENSION ( LDB, kb ), where kb is
             k  when  TRANS = 'N' or 'n',  and is  n  otherwise.
             Before entry with  TRANS = 'N' or 'n',  the  leading  n by k
             part of the array  B  must contain the matrix  B,  otherwise
             the leading  k by n  part of the array  B  must contain  the
             matrix B.
             Unchanged on exit.

    LDB    - INTEGER.
             On entry, LDB specifies the first dimension of B as declared
             in  the  calling  (sub)  program.   When  TRANS = 'N' or 'n'
             then  LDB must be at least  max( 1, n ), otherwise  LDB must
             be at least  max( 1, k ).
             Unchanged on exit.

    BETA   - DOUBLE PRECISION.
             On entry, BETA specifies the scalar beta.
             Unchanged on exit.

    C      - DOUBLE PRECISION array of DIMENSION ( LDC, n ).
             Before entry  with  UPLO = 'U' or 'u',  the leading  n by n
             upper triangular part of the array C must contain the upper
             triangular part  of the  symmetric matrix  and the strictly
             lower triangular part of C is not referenced.  On exit, the
             upper triangular part of the array  C is overwritten by the
             upper triangular part of the updated matrix.
             Before entry  with  UPLO = 'L' or 'l',  the leading  n by n
             lower triangular part of the array C must contain the lower
             triangular part  of the  symmetric matrix  and the strictly
             upper triangular part of C is not referenced.  On exit, the
             lower triangular part of the array  C is overwritten by the
             lower triangular part of the updated matrix.

    LDC    - INTEGER.
             On entry, LDC specifies the first dimension of C as declared
             in  the  calling  (sub)  program.   LDC  must  be  at  least
             max( 1, n ).
             Unchanged on exit.


    Level 3 Blas routine.


    -- Written on 8-February-1989.
       Jack Dongarra, Argonne National Laboratory.
       Iain Duff, AERE Harwell.
       Jeremy Du Croz, Numerical Algorithms Group Ltd.
       Sven Hammarling, Numerical Algorithms Group Ltd.


       Test the input parameters.
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1 * 1;
    b -= b_offset;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1 * 1;
    c__ -= c_offset;

    /* Function Body */
    if (lsame_(trans, "N")) {
	nrowa = *n;
    } else {
	nrowa = *k;
    }
    upper = lsame_(uplo, "U");

    info = 0;
    if (! upper && ! lsame_(uplo, "L")) {
	info = 1;
    } else if (! lsame_(trans, "N") && ! lsame_(trans,
	    "T") && ! lsame_(trans, "C")) {
	info = 2;
    } else if (*n < 0) {
	info = 3;
    } else if (*k < 0) {
	info = 4;
    } else if (*lda < max(1,nrowa)) {
	info = 7;
    } else if (*ldb < max(1,nrowa)) {
	info = 9;
    } else if (*ldc < max(1,*n)) {
	info = 12;
    }
    if (info != 0) {
	xerbla_("DSYR2K", &info);
	return 0;
    }

/*     Quick return if possible. */

    if (*n == 0 || (*alpha == 0. || *k == 0) && *beta == 1.) {
	return 0;
    }

/*     And when  alpha.eq.zero. */

    if (*alpha == 0.) {
	if (upper) {
	    if (*beta == 0.) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = j;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = 0.;
/* L10: */
		    }
/* L20: */
		}
	    } else {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = j;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
/* L30: */
		    }
/* L40: */
		}
	    }
	} else {
	    if (*beta == 0.) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *n;
		    for (i__ = j; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = 0.;
/* L50: */
		    }
/* L60: */
		}
	    } else {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *n;
		    for (i__ = j; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
/* L70: */
		    }
/* L80: */
		}
	    }
	}
	return 0;
    }

/*     Start the operations. */

    if (lsame_(trans, "N")) {

/*        Form  C := alpha*A*B' + alpha*B*A' + C. */

	if (upper) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		if (*beta == 0.) {
		    i__2 = j;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = 0.;
/* L90: */
		    }
		} else if (*beta != 1.) {
		    i__2 = j;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
/* L100: */
		    }
		}
		i__2 = *k;
		for (l = 1; l <= i__2; ++l) {
		    if (a[j + l * a_dim1] != 0. || b[j + l * b_dim1] != 0.) {
			temp1 = *alpha * b[j + l * b_dim1];
			temp2 = *alpha * a[j + l * a_dim1];
			i__3 = j;
			for (i__ = 1; i__ <= i__3; ++i__) {
			    c__[i__ + j * c_dim1] = c__[i__ + j * c_dim1] + a[
				    i__ + l * a_dim1] * temp1 + b[i__ + l *
				    b_dim1] * temp2;
/* L110: */
			}
		    }
/* L120: */
		}
/* L130: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		if (*beta == 0.) {
		    i__2 = *n;
		    for (i__ = j; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = 0.;
/* L140: */
		    }
		} else if (*beta != 1.) {
		    i__2 = *n;
		    for (i__ = j; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
/* L150: */
		    }
		}
		i__2 = *k;
		for (l = 1; l <= i__2; ++l) {
		    if (a[j + l * a_dim1] != 0. || b[j + l * b_dim1] != 0.) {
			temp1 = *alpha * b[j + l * b_dim1];
			temp2 = *alpha * a[j + l * a_dim1];
			i__3 = *n;
			for (i__ = j; i__ <= i__3; ++i__) {
			    c__[i__ + j * c_dim1] = c__[i__ + j * c_dim1] + a[
				    i__ + l * a_dim1] * temp1 + b[i__ + l *
				    b_dim1] * temp2;
/* L160: */
			}
		    }
/* L170: */
		}
/* L180: */
	    }
	}
    } else {

/*        Form  C := alpha*A'*B + alpha*B'*A + C. */

	if (upper) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = j;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    temp1 = 0.;
		    temp2 = 0.;
		    i__3 = *k;
		    for (l = 1; l <= i__3; ++l) {
			temp1 += a[l + i__ * a_dim1] * b[l + j * b_dim1];
			temp2 += b[l + i__ * b_dim1] * a[l + j * a_dim1];
/* L190: */
		    }
		    if (*beta == 0.) {
			c__[i__ + j * c_dim1] = *alpha * temp1 + *alpha *
				temp2;
		    } else {
			c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1]
				+ *alpha * temp1 + *alpha * temp2;
		    }
/* L200: */
		}
/* L210: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = *n;
		for (i__ = j; i__ <= i__2; ++i__) {
		    temp1 = 0.;
		    temp2 = 0.;
		    i__3 = *k;
		    for (l = 1; l <= i__3; ++l) {
			temp1 += a[l + i__ * a_dim1] * b[l + j * b_dim1];
			temp2 += b[l + i__ * b_dim1] * a[l + j * a_dim1];
/* L220: */
		    }
		    if (*beta == 0.) {
			c__[i__ + j * c_dim1] = *alpha * temp1 + *alpha *
				temp2;
		    } else {
			c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1]
				+ *alpha * temp1 + *alpha * temp2;
		    }
/* L230: */
		}
/* L240: */
	    }
	}
    }

    return 0;

/*     End of DSYR2K. */

} /* dsyr2k_ */

/* Subroutine */ int dsyrk_(char *uplo, char *trans, integer *n, integer *k,
	doublereal *alpha, doublereal *a, integer *lda, doublereal *beta,
	doublereal *c__, integer *ldc)
{
    /* System generated locals */
    integer a_dim1, a_offset, c_dim1, c_offset, i__1, i__2, i__3;

    /* Local variables */
    static integer info;
    static doublereal temp;
    static integer i__, j, l;
    extern logical lsame_(char *, char *);
    static integer nrowa;
    static logical upper;
    extern /* Subroutine */ int xerbla_(char *, integer *);


/*
    Purpose
    =======

    DSYRK  performs one of the symmetric rank k operations

       C := alpha*A*A' + beta*C,

    or

       C := alpha*A'*A + beta*C,

    where  alpha and beta  are scalars, C is an  n by n  symmetric matrix
    and  A  is an  n by k  matrix in the first case and a  k by n  matrix
    in the second case.

    Arguments
    ==========

    UPLO   - CHARACTER*1.
             On  entry,   UPLO  specifies  whether  the  upper  or  lower
             triangular  part  of the  array  C  is to be  referenced  as
             follows:

                UPLO = 'U' or 'u'   Only the  upper triangular part of  C
                                    is to be referenced.

                UPLO = 'L' or 'l'   Only the  lower triangular part of  C
                                    is to be referenced.

             Unchanged on exit.

    TRANS  - CHARACTER*1.
             On entry,  TRANS  specifies the operation to be performed as
             follows:

                TRANS = 'N' or 'n'   C := alpha*A*A' + beta*C.

                TRANS = 'T' or 't'   C := alpha*A'*A + beta*C.

                TRANS = 'C' or 'c'   C := alpha*A'*A + beta*C.

             Unchanged on exit.

    N      - INTEGER.
             On entry,  N specifies the order of the matrix C.  N must be
             at least zero.
             Unchanged on exit.

    K      - INTEGER.
             On entry with  TRANS = 'N' or 'n',  K  specifies  the number
             of  columns   of  the   matrix   A,   and  on   entry   with
             TRANS = 'T' or 't' or 'C' or 'c',  K  specifies  the  number
             of rows of the matrix  A.  K must be at least zero.
             Unchanged on exit.

    ALPHA  - DOUBLE PRECISION.
             On entry, ALPHA specifies the scalar alpha.
             Unchanged on exit.

    A      - DOUBLE PRECISION array of DIMENSION ( LDA, ka ), where ka is
             k  when  TRANS = 'N' or 'n',  and is  n  otherwise.
             Before entry with  TRANS = 'N' or 'n',  the  leading  n by k
             part of the array  A  must contain the matrix  A,  otherwise
             the leading  k by n  part of the array  A  must contain  the
             matrix A.
             Unchanged on exit.

    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in  the  calling  (sub)  program.   When  TRANS = 'N' or 'n'
             then  LDA must be at least  max( 1, n ), otherwise  LDA must
             be at least  max( 1, k ).
             Unchanged on exit.

    BETA   - DOUBLE PRECISION.
             On entry, BETA specifies the scalar beta.
             Unchanged on exit.

    C      - DOUBLE PRECISION array of DIMENSION ( LDC, n ).
             Before entry  with  UPLO = 'U' or 'u',  the leading  n by n
             upper triangular part of the array C must contain the upper
             triangular part  of the  symmetric matrix  and the strictly
             lower triangular part of C is not referenced.  On exit, the
             upper triangular part of the array  C is overwritten by the
             upper triangular part of the updated matrix.
             Before entry  with  UPLO = 'L' or 'l',  the leading  n by n
             lower triangular part of the array C must contain the lower
             triangular part  of the  symmetric matrix  and the strictly
             upper triangular part of C is not referenced.  On exit, the
             lower triangular part of the array  C is overwritten by the
             lower triangular part of the updated matrix.

    LDC    - INTEGER.
             On entry, LDC specifies the first dimension of C as declared
             in  the  calling  (sub)  program.   LDC  must  be  at  least
             max( 1, n ).
             Unchanged on exit.


    Level 3 Blas routine.

    -- Written on 8-February-1989.
       Jack Dongarra, Argonne National Laboratory.
       Iain Duff, AERE Harwell.
       Jeremy Du Croz, Numerical Algorithms Group Ltd.
       Sven Hammarling, Numerical Algorithms Group Ltd.


       Test the input parameters.
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1 * 1;
    c__ -= c_offset;

    /* Function Body */
    if (lsame_(trans, "N")) {
	nrowa = *n;
    } else {
	nrowa = *k;
    }
    upper = lsame_(uplo, "U");

    info = 0;
    if (! upper && ! lsame_(uplo, "L")) {
	info = 1;
    } else if (! lsame_(trans, "N") && ! lsame_(trans,
	    "T") && ! lsame_(trans, "C")) {
	info = 2;
    } else if (*n < 0) {
	info = 3;
    } else if (*k < 0) {
	info = 4;
    } else if (*lda < max(1,nrowa)) {
	info = 7;
    } else if (*ldc < max(1,*n)) {
	info = 10;
    }
    if (info != 0) {
	xerbla_("DSYRK ", &info);
	return 0;
    }

/*     Quick return if possible. */

    if (*n == 0 || (*alpha == 0. || *k == 0) && *beta == 1.) {
	return 0;
    }

/*     And when  alpha.eq.zero. */

    if (*alpha == 0.) {
	if (upper) {
	    if (*beta == 0.) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = j;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = 0.;
/* L10: */
		    }
/* L20: */
		}
	    } else {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = j;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
/* L30: */
		    }
/* L40: */
		}
	    }
	} else {
	    if (*beta == 0.) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *n;
		    for (i__ = j; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = 0.;
/* L50: */
		    }
/* L60: */
		}
	    } else {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *n;
		    for (i__ = j; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
/* L70: */
		    }
/* L80: */
		}
	    }
	}
	return 0;
    }

/*     Start the operations. */

    if (lsame_(trans, "N")) {

/*        Form  C := alpha*A*A' + beta*C. */

	if (upper) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		if (*beta == 0.) {
		    i__2 = j;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = 0.;
/* L90: */
		    }
		} else if (*beta != 1.) {
		    i__2 = j;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
/* L100: */
		    }
		}
		i__2 = *k;
		for (l = 1; l <= i__2; ++l) {
		    if (a[j + l * a_dim1] != 0.) {
			temp = *alpha * a[j + l * a_dim1];
			i__3 = j;
			for (i__ = 1; i__ <= i__3; ++i__) {
			    c__[i__ + j * c_dim1] += temp * a[i__ + l *
				    a_dim1];
/* L110: */
			}
		    }
/* L120: */
		}
/* L130: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		if (*beta == 0.) {
		    i__2 = *n;
		    for (i__ = j; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = 0.;
/* L140: */
		    }
		} else if (*beta != 1.) {
		    i__2 = *n;
		    for (i__ = j; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
/* L150: */
		    }
		}
		i__2 = *k;
		for (l = 1; l <= i__2; ++l) {
		    if (a[j + l * a_dim1] != 0.) {
			temp = *alpha * a[j + l * a_dim1];
			i__3 = *n;
			for (i__ = j; i__ <= i__3; ++i__) {
			    c__[i__ + j * c_dim1] += temp * a[i__ + l *
				    a_dim1];
/* L160: */
			}
		    }
/* L170: */
		}
/* L180: */
	    }
	}
    } else {

/*        Form  C := alpha*A'*A + beta*C. */

	if (upper) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = j;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    temp = 0.;
		    i__3 = *k;
		    for (l = 1; l <= i__3; ++l) {
			temp += a[l + i__ * a_dim1] * a[l + j * a_dim1];
/* L190: */
		    }
		    if (*beta == 0.) {
			c__[i__ + j * c_dim1] = *alpha * temp;
		    } else {
			c__[i__ + j * c_dim1] = *alpha * temp + *beta * c__[
				i__ + j * c_dim1];
		    }
/* L200: */
		}
/* L210: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = *n;
		for (i__ = j; i__ <= i__2; ++i__) {
		    temp = 0.;
		    i__3 = *k;
		    for (l = 1; l <= i__3; ++l) {
			temp += a[l + i__ * a_dim1] * a[l + j * a_dim1];
/* L220: */
		    }
		    if (*beta == 0.) {
			c__[i__ + j * c_dim1] = *alpha * temp;
		    } else {
			c__[i__ + j * c_dim1] = *alpha * temp + *beta * c__[
				i__ + j * c_dim1];
		    }
/* L230: */
		}
/* L240: */
	    }
	}
    }

    return 0;

/*     End of DSYRK . */

} /* dsyrk_ */

/* Subroutine */ int dtrmm_(char *side, char *uplo, char *transa, char *diag,
	integer *m, integer *n, doublereal *alpha, doublereal *a, integer *
	lda, doublereal *b, integer *ldb)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, i__1, i__2, i__3;

    /* Local variables */
    static integer info;
    static doublereal temp;
    static integer i__, j, k;
    static logical lside;
    extern logical lsame_(char *, char *);
    static integer nrowa;
    static logical upper;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    static logical nounit;


/*
    Purpose
    =======

    DTRMM  performs one of the matrix-matrix operations

       B := alpha*op( A )*B,   or   B := alpha*B*op( A ),

    where  alpha  is a scalar,  B  is an m by n matrix,  A  is a unit, or
    non-unit,  upper or lower triangular matrix  and  op( A )  is one  of

       op( A ) = A   or   op( A ) = A'.

    Arguments
    ==========

    SIDE   - CHARACTER*1.
             On entry,  SIDE specifies whether  op( A ) multiplies B from
             the left or right as follows:

                SIDE = 'L' or 'l'   B := alpha*op( A )*B.

                SIDE = 'R' or 'r'   B := alpha*B*op( A ).

             Unchanged on exit.

    UPLO   - CHARACTER*1.
             On entry, UPLO specifies whether the matrix A is an upper or
             lower triangular matrix as follows:

                UPLO = 'U' or 'u'   A is an upper triangular matrix.

                UPLO = 'L' or 'l'   A is a lower triangular matrix.

             Unchanged on exit.

    TRANSA - CHARACTER*1.
             On entry, TRANSA specifies the form of op( A ) to be used in
             the matrix multiplication as follows:

                TRANSA = 'N' or 'n'   op( A ) = A.

                TRANSA = 'T' or 't'   op( A ) = A'.

                TRANSA = 'C' or 'c'   op( A ) = A'.

             Unchanged on exit.

    DIAG   - CHARACTER*1.
             On entry, DIAG specifies whether or not A is unit triangular
             as follows:

                DIAG = 'U' or 'u'   A is assumed to be unit triangular.

                DIAG = 'N' or 'n'   A is not assumed to be unit
                                    triangular.

             Unchanged on exit.

    M      - INTEGER.
             On entry, M specifies the number of rows of B. M must be at
             least zero.
             Unchanged on exit.

    N      - INTEGER.
             On entry, N specifies the number of columns of B.  N must be
             at least zero.
             Unchanged on exit.

    ALPHA  - DOUBLE PRECISION.
             On entry,  ALPHA specifies the scalar  alpha. When  alpha is
             zero then  A is not referenced and  B need not be set before
             entry.
             Unchanged on exit.

    A      - DOUBLE PRECISION array of DIMENSION ( LDA, k ), where k is m
             when  SIDE = 'L' or 'l'  and is  n  when  SIDE = 'R' or 'r'.
             Before entry  with  UPLO = 'U' or 'u',  the  leading  k by k
             upper triangular part of the array  A must contain the upper
             triangular matrix  and the strictly lower triangular part of
             A is not referenced.
             Before entry  with  UPLO = 'L' or 'l',  the  leading  k by k
             lower triangular part of the array  A must contain the lower
             triangular matrix  and the strictly upper triangular part of
             A is not referenced.
             Note that when  DIAG = 'U' or 'u',  the diagonal elements of
             A  are not referenced either,  but are assumed to be  unity.
             Unchanged on exit.

    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in the calling (sub) program.  When  SIDE = 'L' or 'l'  then
             LDA  must be at least  max( 1, m ),  when  SIDE = 'R' or 'r'
             then LDA must be at least max( 1, n ).
             Unchanged on exit.

    B      - DOUBLE PRECISION array of DIMENSION ( LDB, n ).
             Before entry,  the leading  m by n part of the array  B must
             contain the matrix  B,  and  on exit  is overwritten  by the
             transformed matrix.

    LDB    - INTEGER.
             On entry, LDB specifies the first dimension of B as declared
             in  the  calling  (sub)  program.   LDB  must  be  at  least
             max( 1, m ).
             Unchanged on exit.


    Level 3 Blas routine.

    -- Written on 8-February-1989.
       Jack Dongarra, Argonne National Laboratory.
       Iain Duff, AERE Harwell.
       Jeremy Du Croz, Numerical Algorithms Group Ltd.
       Sven Hammarling, Numerical Algorithms Group Ltd.


       Test the input parameters.
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1 * 1;
    b -= b_offset;

    /* Function Body */
    lside = lsame_(side, "L");
    if (lside) {
	nrowa = *m;
    } else {
	nrowa = *n;
    }
    nounit = lsame_(diag, "N");
    upper = lsame_(uplo, "U");

    info = 0;
    if (! lside && ! lsame_(side, "R")) {
	info = 1;
    } else if (! upper && ! lsame_(uplo, "L")) {
	info = 2;
    } else if (! lsame_(transa, "N") && ! lsame_(transa,
	     "T") && ! lsame_(transa, "C")) {
	info = 3;
    } else if (! lsame_(diag, "U") && ! lsame_(diag,
	    "N")) {
	info = 4;
    } else if (*m < 0) {
	info = 5;
    } else if (*n < 0) {
	info = 6;
    } else if (*lda < max(1,nrowa)) {
	info = 9;
    } else if (*ldb < max(1,*m)) {
	info = 11;
    }
    if (info != 0) {
	xerbla_("DTRMM ", &info);
	return 0;
    }

/*     Quick return if possible. */

    if (*n == 0) {
	return 0;
    }

/*     And when  alpha.eq.zero. */

    if (*alpha == 0.) {
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *m;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		b[i__ + j * b_dim1] = 0.;
/* L10: */
	    }
/* L20: */
	}
	return 0;
    }

/*     Start the operations. */

    if (lside) {
	if (lsame_(transa, "N")) {

/*           Form  B := alpha*A*B. */

	    if (upper) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *m;
		    for (k = 1; k <= i__2; ++k) {
			if (b[k + j * b_dim1] != 0.) {
			    temp = *alpha * b[k + j * b_dim1];
			    i__3 = k - 1;
			    for (i__ = 1; i__ <= i__3; ++i__) {
				b[i__ + j * b_dim1] += temp * a[i__ + k *
					a_dim1];
/* L30: */
			    }
			    if (nounit) {
				temp *= a[k + k * a_dim1];
			    }
			    b[k + j * b_dim1] = temp;
			}
/* L40: */
		    }
/* L50: */
		}
	    } else {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    for (k = *m; k >= 1; --k) {
			if (b[k + j * b_dim1] != 0.) {
			    temp = *alpha * b[k + j * b_dim1];
			    b[k + j * b_dim1] = temp;
			    if (nounit) {
				b[k + j * b_dim1] *= a[k + k * a_dim1];
			    }
			    i__2 = *m;
			    for (i__ = k + 1; i__ <= i__2; ++i__) {
				b[i__ + j * b_dim1] += temp * a[i__ + k *
					a_dim1];
/* L60: */
			    }
			}
/* L70: */
		    }
/* L80: */
		}
	    }
	} else {

/*           Form  B := alpha*A'*B. */

	    if (upper) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    for (i__ = *m; i__ >= 1; --i__) {
			temp = b[i__ + j * b_dim1];
			if (nounit) {
			    temp *= a[i__ + i__ * a_dim1];
			}
			i__2 = i__ - 1;
			for (k = 1; k <= i__2; ++k) {
			    temp += a[k + i__ * a_dim1] * b[k + j * b_dim1];
/* L90: */
			}
			b[i__ + j * b_dim1] = *alpha * temp;
/* L100: */
		    }
/* L110: */
		}
	    } else {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *m;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			temp = b[i__ + j * b_dim1];
			if (nounit) {
			    temp *= a[i__ + i__ * a_dim1];
			}
			i__3 = *m;
			for (k = i__ + 1; k <= i__3; ++k) {
			    temp += a[k + i__ * a_dim1] * b[k + j * b_dim1];
/* L120: */
			}
			b[i__ + j * b_dim1] = *alpha * temp;
/* L130: */
		    }
/* L140: */
		}
	    }
	}
    } else {
	if (lsame_(transa, "N")) {

/*           Form  B := alpha*B*A. */

	    if (upper) {
		for (j = *n; j >= 1; --j) {
		    temp = *alpha;
		    if (nounit) {
			temp *= a[j + j * a_dim1];
		    }
		    i__1 = *m;
		    for (i__ = 1; i__ <= i__1; ++i__) {
			b[i__ + j * b_dim1] = temp * b[i__ + j * b_dim1];
/* L150: */
		    }
		    i__1 = j - 1;
		    for (k = 1; k <= i__1; ++k) {
			if (a[k + j * a_dim1] != 0.) {
			    temp = *alpha * a[k + j * a_dim1];
			    i__2 = *m;
			    for (i__ = 1; i__ <= i__2; ++i__) {
				b[i__ + j * b_dim1] += temp * b[i__ + k *
					b_dim1];
/* L160: */
			    }
			}
/* L170: */
		    }
/* L180: */
		}
	    } else {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    temp = *alpha;
		    if (nounit) {
			temp *= a[j + j * a_dim1];
		    }
		    i__2 = *m;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			b[i__ + j * b_dim1] = temp * b[i__ + j * b_dim1];
/* L190: */
		    }
		    i__2 = *n;
		    for (k = j + 1; k <= i__2; ++k) {
			if (a[k + j * a_dim1] != 0.) {
			    temp = *alpha * a[k + j * a_dim1];
			    i__3 = *m;
			    for (i__ = 1; i__ <= i__3; ++i__) {
				b[i__ + j * b_dim1] += temp * b[i__ + k *
					b_dim1];
/* L200: */
			    }
			}
/* L210: */
		    }
/* L220: */
		}
	    }
	} else {

/*           Form  B := alpha*B*A'. */

	    if (upper) {
		i__1 = *n;
		for (k = 1; k <= i__1; ++k) {
		    i__2 = k - 1;
		    for (j = 1; j <= i__2; ++j) {
			if (a[j + k * a_dim1] != 0.) {
			    temp = *alpha * a[j + k * a_dim1];
			    i__3 = *m;
			    for (i__ = 1; i__ <= i__3; ++i__) {
				b[i__ + j * b_dim1] += temp * b[i__ + k *
					b_dim1];
/* L230: */
			    }
			}
/* L240: */
		    }
		    temp = *alpha;
		    if (nounit) {
			temp *= a[k + k * a_dim1];
		    }
		    if (temp != 1.) {
			i__2 = *m;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    b[i__ + k * b_dim1] = temp * b[i__ + k * b_dim1];
/* L250: */
			}
		    }
/* L260: */
		}
	    } else {
		for (k = *n; k >= 1; --k) {
		    i__1 = *n;
		    for (j = k + 1; j <= i__1; ++j) {
			if (a[j + k * a_dim1] != 0.) {
			    temp = *alpha * a[j + k * a_dim1];
			    i__2 = *m;
			    for (i__ = 1; i__ <= i__2; ++i__) {
				b[i__ + j * b_dim1] += temp * b[i__ + k *
					b_dim1];
/* L270: */
			    }
			}
/* L280: */
		    }
		    temp = *alpha;
		    if (nounit) {
			temp *= a[k + k * a_dim1];
		    }
		    if (temp != 1.) {
			i__1 = *m;
			for (i__ = 1; i__ <= i__1; ++i__) {
			    b[i__ + k * b_dim1] = temp * b[i__ + k * b_dim1];
/* L290: */
			}
		    }
/* L300: */
		}
	    }
	}
    }

    return 0;

/*     End of DTRMM . */

} /* dtrmm_ */

/* Subroutine */ int dtrmv_(char *uplo, char *trans, char *diag, integer *n,
	doublereal *a, integer *lda, doublereal *x, integer *incx)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;

    /* Local variables */
    static integer info;
    static doublereal temp;
    static integer i__, j;
    extern logical lsame_(char *, char *);
    static integer ix, jx, kx;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    static logical nounit;


/*
    Purpose
    =======

    DTRMV  performs one of the matrix-vector operations

       x := A*x,   or   x := A'*x,

    where x is an n element vector and  A is an n by n unit, or non-unit,
    upper or lower triangular matrix.

    Arguments
    ==========

    UPLO   - CHARACTER*1.
             On entry, UPLO specifies whether the matrix is an upper or
             lower triangular matrix as follows:

                UPLO = 'U' or 'u'   A is an upper triangular matrix.

                UPLO = 'L' or 'l'   A is a lower triangular matrix.

             Unchanged on exit.

    TRANS  - CHARACTER*1.
             On entry, TRANS specifies the operation to be performed as
             follows:

                TRANS = 'N' or 'n'   x := A*x.

                TRANS = 'T' or 't'   x := A'*x.

                TRANS = 'C' or 'c'   x := A'*x.

             Unchanged on exit.

    DIAG   - CHARACTER*1.
             On entry, DIAG specifies whether or not A is unit
             triangular as follows:

                DIAG = 'U' or 'u'   A is assumed to be unit triangular.

                DIAG = 'N' or 'n'   A is not assumed to be unit
                                    triangular.

             Unchanged on exit.

    N      - INTEGER.
             On entry, N specifies the order of the matrix A.
             N must be at least zero.
             Unchanged on exit.

    A      - DOUBLE PRECISION array of DIMENSION ( LDA, n ).
             Before entry with  UPLO = 'U' or 'u', the leading n by n
             upper triangular part of the array A must contain the upper
             triangular matrix and the strictly lower triangular part of
             A is not referenced.
             Before entry with UPLO = 'L' or 'l', the leading n by n
             lower triangular part of the array A must contain the lower
             triangular matrix and the strictly upper triangular part of
             A is not referenced.
             Note that when  DIAG = 'U' or 'u', the diagonal elements of
             A are not referenced either, but are assumed to be unity.
             Unchanged on exit.

    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in the calling (sub) program. LDA must be at least
             max( 1, n ).
             Unchanged on exit.

    X      - DOUBLE PRECISION array of dimension at least
             ( 1 + ( n - 1 )*abs( INCX ) ).
             Before entry, the incremented array X must contain the n
             element vector x. On exit, X is overwritten with the
             tranformed vector x.

    INCX   - INTEGER.
             On entry, INCX specifies the increment for the elements of
             X. INCX must not be zero.
             Unchanged on exit.


    Level 2 Blas routine.

    -- Written on 22-October-1986.
       Jack Dongarra, Argonne National Lab.
       Jeremy Du Croz, Nag Central Office.
       Sven Hammarling, Nag Central Office.
       Richard Hanson, Sandia National Labs.


       Test the input parameters.
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;
    --x;

    /* Function Body */
    info = 0;
    if (! lsame_(uplo, "U") && ! lsame_(uplo, "L")) {
	info = 1;
    } else if (! lsame_(trans, "N") && ! lsame_(trans,
	    "T") && ! lsame_(trans, "C")) {
	info = 2;
    } else if (! lsame_(diag, "U") && ! lsame_(diag,
	    "N")) {
	info = 3;
    } else if (*n < 0) {
	info = 4;
    } else if (*lda < max(1,*n)) {
	info = 6;
    } else if (*incx == 0) {
	info = 8;
    }
    if (info != 0) {
	xerbla_("DTRMV ", &info);
	return 0;
    }

/*     Quick return if possible. */

    if (*n == 0) {
	return 0;
    }

    nounit = lsame_(diag, "N");

/*
       Set up the start point in X if the increment is not unity. This
       will be  ( N - 1 )*INCX  too small for descending loops.
*/

    if (*incx <= 0) {
	kx = 1 - (*n - 1) * *incx;
    } else if (*incx != 1) {
	kx = 1;
    }

/*
       Start the operations. In this version the elements of A are
       accessed sequentially with one pass through A.
*/

    if (lsame_(trans, "N")) {

/*        Form  x := A*x. */

	if (lsame_(uplo, "U")) {
	    if (*incx == 1) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    if (x[j] != 0.) {
			temp = x[j];
			i__2 = j - 1;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    x[i__] += temp * a[i__ + j * a_dim1];
/* L10: */
			}
			if (nounit) {
			    x[j] *= a[j + j * a_dim1];
			}
		    }
/* L20: */
		}
	    } else {
		jx = kx;
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    if (x[jx] != 0.) {
			temp = x[jx];
			ix = kx;
			i__2 = j - 1;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    x[ix] += temp * a[i__ + j * a_dim1];
			    ix += *incx;
/* L30: */
			}
			if (nounit) {
			    x[jx] *= a[j + j * a_dim1];
			}
		    }
		    jx += *incx;
/* L40: */
		}
	    }
	} else {
	    if (*incx == 1) {
		for (j = *n; j >= 1; --j) {
		    if (x[j] != 0.) {
			temp = x[j];
			i__1 = j + 1;
			for (i__ = *n; i__ >= i__1; --i__) {
			    x[i__] += temp * a[i__ + j * a_dim1];
/* L50: */
			}
			if (nounit) {
			    x[j] *= a[j + j * a_dim1];
			}
		    }
/* L60: */
		}
	    } else {
		kx += (*n - 1) * *incx;
		jx = kx;
		for (j = *n; j >= 1; --j) {
		    if (x[jx] != 0.) {
			temp = x[jx];
			ix = kx;
			i__1 = j + 1;
			for (i__ = *n; i__ >= i__1; --i__) {
			    x[ix] += temp * a[i__ + j * a_dim1];
			    ix -= *incx;
/* L70: */
			}
			if (nounit) {
			    x[jx] *= a[j + j * a_dim1];
			}
		    }
		    jx -= *incx;
/* L80: */
		}
	    }
	}
    } else {

/*        Form  x := A'*x. */

	if (lsame_(uplo, "U")) {
	    if (*incx == 1) {
		for (j = *n; j >= 1; --j) {
		    temp = x[j];
		    if (nounit) {
			temp *= a[j + j * a_dim1];
		    }
		    for (i__ = j - 1; i__ >= 1; --i__) {
			temp += a[i__ + j * a_dim1] * x[i__];
/* L90: */
		    }
		    x[j] = temp;
/* L100: */
		}
	    } else {
		jx = kx + (*n - 1) * *incx;
		for (j = *n; j >= 1; --j) {
		    temp = x[jx];
		    ix = jx;
		    if (nounit) {
			temp *= a[j + j * a_dim1];
		    }
		    for (i__ = j - 1; i__ >= 1; --i__) {
			ix -= *incx;
			temp += a[i__ + j * a_dim1] * x[ix];
/* L110: */
		    }
		    x[jx] = temp;
		    jx -= *incx;
/* L120: */
		}
	    }
	} else {
	    if (*incx == 1) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    temp = x[j];
		    if (nounit) {
			temp *= a[j + j * a_dim1];
		    }
		    i__2 = *n;
		    for (i__ = j + 1; i__ <= i__2; ++i__) {
			temp += a[i__ + j * a_dim1] * x[i__];
/* L130: */
		    }
		    x[j] = temp;
/* L140: */
		}
	    } else {
		jx = kx;
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    temp = x[jx];
		    ix = jx;
		    if (nounit) {
			temp *= a[j + j * a_dim1];
		    }
		    i__2 = *n;
		    for (i__ = j + 1; i__ <= i__2; ++i__) {
			ix += *incx;
			temp += a[i__ + j * a_dim1] * x[ix];
/* L150: */
		    }
		    x[jx] = temp;
		    jx += *incx;
/* L160: */
		}
	    }
	}
    }

    return 0;

/*     End of DTRMV . */

} /* dtrmv_ */

/* Subroutine */ int dtrsm_(char *side, char *uplo, char *transa, char *diag,
	integer *m, integer *n, doublereal *alpha, doublereal *a, integer *
	lda, doublereal *b, integer *ldb)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, i__1, i__2, i__3;

    /* Local variables */
    static integer info;
    static doublereal temp;
    static integer i__, j, k;
    static logical lside;
    extern logical lsame_(char *, char *);
    static integer nrowa;
    static logical upper;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    static logical nounit;


/*
    Purpose
    =======

    DTRSM  solves one of the matrix equations

       op( A )*X = alpha*B,   or   X*op( A ) = alpha*B,

    where alpha is a scalar, X and B are m by n matrices, A is a unit, or
    non-unit,  upper or lower triangular matrix  and  op( A )  is one  of

       op( A ) = A   or   op( A ) = A'.

    The matrix X is overwritten on B.

    Arguments
    ==========

    SIDE   - CHARACTER*1.
             On entry, SIDE specifies whether op( A ) appears on the left
             or right of X as follows:

                SIDE = 'L' or 'l'   op( A )*X = alpha*B.

                SIDE = 'R' or 'r'   X*op( A ) = alpha*B.

             Unchanged on exit.

    UPLO   - CHARACTER*1.
             On entry, UPLO specifies whether the matrix A is an upper or
             lower triangular matrix as follows:

                UPLO = 'U' or 'u'   A is an upper triangular matrix.

                UPLO = 'L' or 'l'   A is a lower triangular matrix.

             Unchanged on exit.

    TRANSA - CHARACTER*1.
             On entry, TRANSA specifies the form of op( A ) to be used in
             the matrix multiplication as follows:

                TRANSA = 'N' or 'n'   op( A ) = A.

                TRANSA = 'T' or 't'   op( A ) = A'.

                TRANSA = 'C' or 'c'   op( A ) = A'.

             Unchanged on exit.

    DIAG   - CHARACTER*1.
             On entry, DIAG specifies whether or not A is unit triangular
             as follows:

                DIAG = 'U' or 'u'   A is assumed to be unit triangular.

                DIAG = 'N' or 'n'   A is not assumed to be unit
                                    triangular.

             Unchanged on exit.

    M      - INTEGER.
             On entry, M specifies the number of rows of B. M must be at
             least zero.
             Unchanged on exit.

    N      - INTEGER.
             On entry, N specifies the number of columns of B.  N must be
             at least zero.
             Unchanged on exit.

    ALPHA  - DOUBLE PRECISION.
             On entry,  ALPHA specifies the scalar  alpha. When  alpha is
             zero then  A is not referenced and  B need not be set before
             entry.
             Unchanged on exit.

    A      - DOUBLE PRECISION array of DIMENSION ( LDA, k ), where k is m
             when  SIDE = 'L' or 'l'  and is  n  when  SIDE = 'R' or 'r'.
             Before entry  with  UPLO = 'U' or 'u',  the  leading  k by k
             upper triangular part of the array  A must contain the upper
             triangular matrix  and the strictly lower triangular part of
             A is not referenced.
             Before entry  with  UPLO = 'L' or 'l',  the  leading  k by k
             lower triangular part of the array  A must contain the lower
             triangular matrix  and the strictly upper triangular part of
             A is not referenced.
             Note that when  DIAG = 'U' or 'u',  the diagonal elements of
             A  are not referenced either,  but are assumed to be  unity.
             Unchanged on exit.

    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in the calling (sub) program.  When  SIDE = 'L' or 'l'  then
             LDA  must be at least  max( 1, m ),  when  SIDE = 'R' or 'r'
             then LDA must be at least max( 1, n ).
             Unchanged on exit.

    B      - DOUBLE PRECISION array of DIMENSION ( LDB, n ).
             Before entry,  the leading  m by n part of the array  B must
             contain  the  right-hand  side  matrix  B,  and  on exit  is
             overwritten by the solution matrix  X.

    LDB    - INTEGER.
             On entry, LDB specifies the first dimension of B as declared
             in  the  calling  (sub)  program.   LDB  must  be  at  least
             max( 1, m ).
             Unchanged on exit.


    Level 3 Blas routine.


    -- Written on 8-February-1989.
       Jack Dongarra, Argonne National Laboratory.
       Iain Duff, AERE Harwell.
       Jeremy Du Croz, Numerical Algorithms Group Ltd.
       Sven Hammarling, Numerical Algorithms Group Ltd.


       Test the input parameters.
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1 * 1;
    b -= b_offset;

    /* Function Body */
    lside = lsame_(side, "L");
    if (lside) {
	nrowa = *m;
    } else {
	nrowa = *n;
    }
    nounit = lsame_(diag, "N");
    upper = lsame_(uplo, "U");

    info = 0;
    if (! lside && ! lsame_(side, "R")) {
	info = 1;
    } else if (! upper && ! lsame_(uplo, "L")) {
	info = 2;
    } else if (! lsame_(transa, "N") && ! lsame_(transa,
	     "T") && ! lsame_(transa, "C")) {
	info = 3;
    } else if (! lsame_(diag, "U") && ! lsame_(diag,
	    "N")) {
	info = 4;
    } else if (*m < 0) {
	info = 5;
    } else if (*n < 0) {
	info = 6;
    } else if (*lda < max(1,nrowa)) {
	info = 9;
    } else if (*ldb < max(1,*m)) {
	info = 11;
    }
    if (info != 0) {
	xerbla_("DTRSM ", &info);
	return 0;
    }

/*     Quick return if possible. */

    if (*n == 0) {
	return 0;
    }

/*     And when  alpha.eq.zero. */

    if (*alpha == 0.) {
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *m;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		b[i__ + j * b_dim1] = 0.;
/* L10: */
	    }
/* L20: */
	}
	return 0;
    }

/*     Start the operations. */

    if (lside) {
	if (lsame_(transa, "N")) {

/*           Form  B := alpha*inv( A )*B. */

	    if (upper) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    if (*alpha != 1.) {
			i__2 = *m;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    b[i__ + j * b_dim1] = *alpha * b[i__ + j * b_dim1]
				    ;
/* L30: */
			}
		    }
		    for (k = *m; k >= 1; --k) {
			if (b[k + j * b_dim1] != 0.) {
			    if (nounit) {
				b[k + j * b_dim1] /= a[k + k * a_dim1];
			    }
			    i__2 = k - 1;
			    for (i__ = 1; i__ <= i__2; ++i__) {
				b[i__ + j * b_dim1] -= b[k + j * b_dim1] * a[
					i__ + k * a_dim1];
/* L40: */
			    }
			}
/* L50: */
		    }
/* L60: */
		}
	    } else {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    if (*alpha != 1.) {
			i__2 = *m;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    b[i__ + j * b_dim1] = *alpha * b[i__ + j * b_dim1]
				    ;
/* L70: */
			}
		    }
		    i__2 = *m;
		    for (k = 1; k <= i__2; ++k) {
			if (b[k + j * b_dim1] != 0.) {
			    if (nounit) {
				b[k + j * b_dim1] /= a[k + k * a_dim1];
			    }
			    i__3 = *m;
			    for (i__ = k + 1; i__ <= i__3; ++i__) {
				b[i__ + j * b_dim1] -= b[k + j * b_dim1] * a[
					i__ + k * a_dim1];
/* L80: */
			    }
			}
/* L90: */
		    }
/* L100: */
		}
	    }
	} else {

/*           Form  B := alpha*inv( A' )*B. */

	    if (upper) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *m;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			temp = *alpha * b[i__ + j * b_dim1];
			i__3 = i__ - 1;
			for (k = 1; k <= i__3; ++k) {
			    temp -= a[k + i__ * a_dim1] * b[k + j * b_dim1];
/* L110: */
			}
			if (nounit) {
			    temp /= a[i__ + i__ * a_dim1];
			}
			b[i__ + j * b_dim1] = temp;
/* L120: */
		    }
/* L130: */
		}
	    } else {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    for (i__ = *m; i__ >= 1; --i__) {
			temp = *alpha * b[i__ + j * b_dim1];
			i__2 = *m;
			for (k = i__ + 1; k <= i__2; ++k) {
			    temp -= a[k + i__ * a_dim1] * b[k + j * b_dim1];
/* L140: */
			}
			if (nounit) {
			    temp /= a[i__ + i__ * a_dim1];
			}
			b[i__ + j * b_dim1] = temp;
/* L150: */
		    }
/* L160: */
		}
	    }
	}
    } else {
	if (lsame_(transa, "N")) {

/*           Form  B := alpha*B*inv( A ). */

	    if (upper) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    if (*alpha != 1.) {
			i__2 = *m;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    b[i__ + j * b_dim1] = *alpha * b[i__ + j * b_dim1]
				    ;
/* L170: */
			}
		    }
		    i__2 = j - 1;
		    for (k = 1; k <= i__2; ++k) {
			if (a[k + j * a_dim1] != 0.) {
			    i__3 = *m;
			    for (i__ = 1; i__ <= i__3; ++i__) {
				b[i__ + j * b_dim1] -= a[k + j * a_dim1] * b[
					i__ + k * b_dim1];
/* L180: */
			    }
			}
/* L190: */
		    }
		    if (nounit) {
			temp = 1. / a[j + j * a_dim1];
			i__2 = *m;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    b[i__ + j * b_dim1] = temp * b[i__ + j * b_dim1];
/* L200: */
			}
		    }
/* L210: */
		}
	    } else {
		for (j = *n; j >= 1; --j) {
		    if (*alpha != 1.) {
			i__1 = *m;
			for (i__ = 1; i__ <= i__1; ++i__) {
			    b[i__ + j * b_dim1] = *alpha * b[i__ + j * b_dim1]
				    ;
/* L220: */
			}
		    }
		    i__1 = *n;
		    for (k = j + 1; k <= i__1; ++k) {
			if (a[k + j * a_dim1] != 0.) {
			    i__2 = *m;
			    for (i__ = 1; i__ <= i__2; ++i__) {
				b[i__ + j * b_dim1] -= a[k + j * a_dim1] * b[
					i__ + k * b_dim1];
/* L230: */
			    }
			}
/* L240: */
		    }
		    if (nounit) {
			temp = 1. / a[j + j * a_dim1];
			i__1 = *m;
			for (i__ = 1; i__ <= i__1; ++i__) {
			    b[i__ + j * b_dim1] = temp * b[i__ + j * b_dim1];
/* L250: */
			}
		    }
/* L260: */
		}
	    }
	} else {

/*           Form  B := alpha*B*inv( A' ). */

	    if (upper) {
		for (k = *n; k >= 1; --k) {
		    if (nounit) {
			temp = 1. / a[k + k * a_dim1];
			i__1 = *m;
			for (i__ = 1; i__ <= i__1; ++i__) {
			    b[i__ + k * b_dim1] = temp * b[i__ + k * b_dim1];
/* L270: */
			}
		    }
		    i__1 = k - 1;
		    for (j = 1; j <= i__1; ++j) {
			if (a[j + k * a_dim1] != 0.) {
			    temp = a[j + k * a_dim1];
			    i__2 = *m;
			    for (i__ = 1; i__ <= i__2; ++i__) {
				b[i__ + j * b_dim1] -= temp * b[i__ + k *
					b_dim1];
/* L280: */
			    }
			}
/* L290: */
		    }
		    if (*alpha != 1.) {
			i__1 = *m;
			for (i__ = 1; i__ <= i__1; ++i__) {
			    b[i__ + k * b_dim1] = *alpha * b[i__ + k * b_dim1]
				    ;
/* L300: */
			}
		    }
/* L310: */
		}
	    } else {
		i__1 = *n;
		for (k = 1; k <= i__1; ++k) {
		    if (nounit) {
			temp = 1. / a[k + k * a_dim1];
			i__2 = *m;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    b[i__ + k * b_dim1] = temp * b[i__ + k * b_dim1];
/* L320: */
			}
		    }
		    i__2 = *n;
		    for (j = k + 1; j <= i__2; ++j) {
			if (a[j + k * a_dim1] != 0.) {
			    temp = a[j + k * a_dim1];
			    i__3 = *m;
			    for (i__ = 1; i__ <= i__3; ++i__) {
				b[i__ + j * b_dim1] -= temp * b[i__ + k *
					b_dim1];
/* L330: */
			    }
			}
/* L340: */
		    }
		    if (*alpha != 1.) {
			i__2 = *m;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    b[i__ + k * b_dim1] = *alpha * b[i__ + k * b_dim1]
				    ;
/* L350: */
			}
		    }
/* L360: */
		}
	    }
	}
    }

    return 0;

/*     End of DTRSM . */

} /* dtrsm_ */

/* Subroutine */ int dtrsv_(char *uplo, char *trans, char *diag, integer *n,
	doublereal *a, integer *lda, doublereal *x, integer *incx)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;

    /* Local variables */
    static integer info;
    static doublereal temp;
    static integer i__, j;
    extern logical lsame_(char *, char *);
    static integer ix, jx, kx;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    static logical nounit;


/*
    Purpose
    =======

    DTRSV  solves one of the systems of equations

       A*x = b,   or   A'*x = b,

    where b and x are n element vectors and A is an n by n unit, or
    non-unit, upper or lower triangular matrix.

    No test for singularity or near-singularity is included in this
    routine. Such tests must be performed before calling this routine.

    Arguments
    ==========

    UPLO   - CHARACTER*1.
             On entry, UPLO specifies whether the matrix is an upper or
             lower triangular matrix as follows:

                UPLO = 'U' or 'u'   A is an upper triangular matrix.

                UPLO = 'L' or 'l'   A is a lower triangular matrix.

             Unchanged on exit.

    TRANS  - CHARACTER*1.
             On entry, TRANS specifies the equations to be solved as
             follows:

                TRANS = 'N' or 'n'   A*x = b.

                TRANS = 'T' or 't'   A'*x = b.

                TRANS = 'C' or 'c'   A'*x = b.

             Unchanged on exit.

    DIAG   - CHARACTER*1.
             On entry, DIAG specifies whether or not A is unit
             triangular as follows:

                DIAG = 'U' or 'u'   A is assumed to be unit triangular.

                DIAG = 'N' or 'n'   A is not assumed to be unit
                                    triangular.

             Unchanged on exit.

    N      - INTEGER.
             On entry, N specifies the order of the matrix A.
             N must be at least zero.
             Unchanged on exit.

    A      - DOUBLE PRECISION array of DIMENSION ( LDA, n ).
             Before entry with  UPLO = 'U' or 'u', the leading n by n
             upper triangular part of the array A must contain the upper
             triangular matrix and the strictly lower triangular part of
             A is not referenced.
             Before entry with UPLO = 'L' or 'l', the leading n by n
             lower triangular part of the array A must contain the lower
             triangular matrix and the strictly upper triangular part of
             A is not referenced.
             Note that when  DIAG = 'U' or 'u', the diagonal elements of
             A are not referenced either, but are assumed to be unity.
             Unchanged on exit.

    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in the calling (sub) program. LDA must be at least
             max( 1, n ).
             Unchanged on exit.

    X      - DOUBLE PRECISION array of dimension at least
             ( 1 + ( n - 1 )*abs( INCX ) ).
             Before entry, the incremented array X must contain the n
             element right-hand side vector b. On exit, X is overwritten
             with the solution vector x.

    INCX   - INTEGER.
             On entry, INCX specifies the increment for the elements of
             X. INCX must not be zero.
             Unchanged on exit.


    Level 2 Blas routine.

    -- Written on 22-October-1986.
       Jack Dongarra, Argonne National Lab.
       Jeremy Du Croz, Nag Central Office.
       Sven Hammarling, Nag Central Office.
       Richard Hanson, Sandia National Labs.


       Test the input parameters.
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;
    --x;

    /* Function Body */
    info = 0;
    if (! lsame_(uplo, "U") && ! lsame_(uplo, "L")) {
	info = 1;
    } else if (! lsame_(trans, "N") && ! lsame_(trans,
	    "T") && ! lsame_(trans, "C")) {
	info = 2;
    } else if (! lsame_(diag, "U") && ! lsame_(diag,
	    "N")) {
	info = 3;
    } else if (*n < 0) {
	info = 4;
    } else if (*lda < max(1,*n)) {
	info = 6;
    } else if (*incx == 0) {
	info = 8;
    }
    if (info != 0) {
	xerbla_("DTRSV ", &info);
	return 0;
    }

/*     Quick return if possible. */

    if (*n == 0) {
	return 0;
    }

    nounit = lsame_(diag, "N");

/*
       Set up the start point in X if the increment is not unity. This
       will be  ( N - 1 )*INCX  too small for descending loops.
*/

    if (*incx <= 0) {
	kx = 1 - (*n - 1) * *incx;
    } else if (*incx != 1) {
	kx = 1;
    }

/*
       Start the operations. In this version the elements of A are
       accessed sequentially with one pass through A.
*/

    if (lsame_(trans, "N")) {

/*        Form  x := inv( A )*x. */

	if (lsame_(uplo, "U")) {
	    if (*incx == 1) {
		for (j = *n; j >= 1; --j) {
		    if (x[j] != 0.) {
			if (nounit) {
			    x[j] /= a[j + j * a_dim1];
			}
			temp = x[j];
			for (i__ = j - 1; i__ >= 1; --i__) {
			    x[i__] -= temp * a[i__ + j * a_dim1];
/* L10: */
			}
		    }
/* L20: */
		}
	    } else {
		jx = kx + (*n - 1) * *incx;
		for (j = *n; j >= 1; --j) {
		    if (x[jx] != 0.) {
			if (nounit) {
			    x[jx] /= a[j + j * a_dim1];
			}
			temp = x[jx];
			ix = jx;
			for (i__ = j - 1; i__ >= 1; --i__) {
			    ix -= *incx;
			    x[ix] -= temp * a[i__ + j * a_dim1];
/* L30: */
			}
		    }
		    jx -= *incx;
/* L40: */
		}
	    }
	} else {
	    if (*incx == 1) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    if (x[j] != 0.) {
			if (nounit) {
			    x[j] /= a[j + j * a_dim1];
			}
			temp = x[j];
			i__2 = *n;
			for (i__ = j + 1; i__ <= i__2; ++i__) {
			    x[i__] -= temp * a[i__ + j * a_dim1];
/* L50: */
			}
		    }
/* L60: */
		}
	    } else {
		jx = kx;
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    if (x[jx] != 0.) {
			if (nounit) {
			    x[jx] /= a[j + j * a_dim1];
			}
			temp = x[jx];
			ix = jx;
			i__2 = *n;
			for (i__ = j + 1; i__ <= i__2; ++i__) {
			    ix += *incx;
			    x[ix] -= temp * a[i__ + j * a_dim1];
/* L70: */
			}
		    }
		    jx += *incx;
/* L80: */
		}
	    }
	}
    } else {

/*        Form  x := inv( A' )*x. */

	if (lsame_(uplo, "U")) {
	    if (*incx == 1) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    temp = x[j];
		    i__2 = j - 1;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			temp -= a[i__ + j * a_dim1] * x[i__];
/* L90: */
		    }
		    if (nounit) {
			temp /= a[j + j * a_dim1];
		    }
		    x[j] = temp;
/* L100: */
		}
	    } else {
		jx = kx;
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    temp = x[jx];
		    ix = kx;
		    i__2 = j - 1;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			temp -= a[i__ + j * a_dim1] * x[ix];
			ix += *incx;
/* L110: */
		    }
		    if (nounit) {
			temp /= a[j + j * a_dim1];
		    }
		    x[jx] = temp;
		    jx += *incx;
/* L120: */
		}
	    }
	} else {
	    if (*incx == 1) {
		for (j = *n; j >= 1; --j) {
		    temp = x[j];
		    i__1 = j + 1;
		    for (i__ = *n; i__ >= i__1; --i__) {
			temp -= a[i__ + j * a_dim1] * x[i__];
/* L130: */
		    }
		    if (nounit) {
			temp /= a[j + j * a_dim1];
		    }
		    x[j] = temp;
/* L140: */
		}
	    } else {
		kx += (*n - 1) * *incx;
		jx = kx;
		for (j = *n; j >= 1; --j) {
		    temp = x[jx];
		    ix = kx;
		    i__1 = j + 1;
		    for (i__ = *n; i__ >= i__1; --i__) {
			temp -= a[i__ + j * a_dim1] * x[ix];
			ix -= *incx;
/* L150: */
		    }
		    if (nounit) {
			temp /= a[j + j * a_dim1];
		    }
		    x[jx] = temp;
		    jx -= *incx;
/* L160: */
		}
	    }
	}
    }

    return 0;

/*     End of DTRSV . */

} /* dtrsv_ */

integer idamax_(integer *n, doublereal *dx, integer *incx)
{
    /* System generated locals */
    integer ret_val, i__1;
    doublereal d__1;

    /* Local variables */
    static doublereal dmax__;
    static integer i__, ix;


/*
    Purpose
    =======

       finds the index of element having max. absolute value.
       jack dongarra, linpack, 3/11/78.
       modified 3/93 to return if incx .le. 0.
       modified 12/3/93, array(1) declarations changed to array(*)
*/


    /* Parameter adjustments */
    --dx;

    /* Function Body */
    ret_val = 0;
    if (*n < 1 || *incx <= 0) {
	return ret_val;
    }
    ret_val = 1;
    if (*n == 1) {
	return ret_val;
    }
    if (*incx == 1) {
	goto L20;
    }

/*        code for increment not equal to 1 */

    ix = 1;
    dmax__ = abs(dx[1]);
    ix += *incx;
    i__1 = *n;
    for (i__ = 2; i__ <= i__1; ++i__) {
	if ((d__1 = dx[ix], abs(d__1)) <= dmax__) {
	    goto L5;
	}
	ret_val = i__;
	dmax__ = (d__1 = dx[ix], abs(d__1));
L5:
	ix += *incx;
/* L10: */
    }
    return ret_val;

/*        code for increment equal to 1 */

L20:
    dmax__ = abs(dx[1]);
    i__1 = *n;
    for (i__ = 2; i__ <= i__1; ++i__) {
	if ((d__1 = dx[i__], abs(d__1)) <= dmax__) {
	    goto L30;
	}
	ret_val = i__;
	dmax__ = (d__1 = dx[i__], abs(d__1));
L30:
	;
    }
    return ret_val;
} /* idamax_ */

logical lsame_(char *ca, char *cb)
{
    /* System generated locals */
    logical ret_val;

    /* Local variables */
    static integer inta, intb, zcode;


/*
    -- LAPACK auxiliary routine (version 3.1) --
       Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
       November 2006


    Purpose
    =======

    LSAME returns .TRUE. if CA is the same letter as CB regardless of
    case.

    Arguments
    =========

    CA      (input) CHARACTER*1

    CB      (input) CHARACTER*1
            CA and CB specify the single characters to be compared.

   =====================================================================


       Test if the characters are equal
*/

    ret_val = *(unsigned char *)ca == *(unsigned char *)cb;
    if (ret_val) {
	return ret_val;
    }

/*     Now test for equivalence if both characters are alphabetic. */

    zcode = 'Z';

/*
       Use 'Z' rather than 'A' so that ASCII can be detected on Prime
       machines, on which ICHAR returns a value with bit 8 set.
       ICHAR('A') on Prime machines returns 193 which is the same as
       ICHAR('A') on an EBCDIC machine.
*/

    inta = *(unsigned char *)ca;
    intb = *(unsigned char *)cb;

    if (zcode == 90 || zcode == 122) {

/*
          ASCII is assumed - ZCODE is the ASCII code of either lower or
          upper case 'Z'.
*/

	if (inta >= 97 && inta <= 122) {
	    inta += -32;
	}
	if (intb >= 97 && intb <= 122) {
	    intb += -32;
	}

    } else if (zcode == 233 || zcode == 169) {

/*
          EBCDIC is assumed - ZCODE is the EBCDIC code of either lower or
          upper case 'Z'.
*/

	if (inta >= 129 && inta <= 137 || inta >= 145 && inta <= 153 || inta
		>= 162 && inta <= 169) {
	    inta += 64;
	}
	if (intb >= 129 && intb <= 137 || intb >= 145 && intb <= 153 || intb
		>= 162 && intb <= 169) {
	    intb += 64;
	}

    } else if (zcode == 218 || zcode == 250) {

/*
          ASCII is assumed, on Prime machines - ZCODE is the ASCII code
          plus 128 of either lower or upper case 'Z'.
*/

	if (inta >= 225 && inta <= 250) {
	    inta += -32;
	}
	if (intb >= 225 && intb <= 250) {
	    intb += -32;
	}
    }
    ret_val = inta == intb;

/*
       RETURN

       End of LSAME
*/

    return ret_val;
} /* lsame_ */

/* Subroutine */ int xerbla_(char *srname, integer *info)
{
    /* Format strings */
    static char fmt_9999[] = "(\002 ** On entry to \002,a6,\002 parameter nu"
	    "mber \002,i2,\002 had \002,\002an illegal value\002)";

    /* Builtin functions */
    integer s_wsfe(cilist *), do_fio(integer *, char *, ftnlen), e_wsfe(void);
    /* Subroutine */ int s_stop(char *, ftnlen);

    /* Fortran I/O blocks */
    static cilist io___197 = { 0, 6, 0, fmt_9999, 0 };


/*
    -- LAPACK auxiliary routine (preliminary version) --
       Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
       November 2006


    Purpose
    =======

    XERBLA  is an error handler for the LAPACK routines.
    It is called by an LAPACK routine if an input parameter has an
    invalid value.  A message is printed and execution stops.

    Installers may consider modifying the STOP statement in order to
    call system-specific exception-handling facilities.

    Arguments
    =========

    SRNAME  (input) CHARACTER*6
            The name of the routine which called XERBLA.

    INFO    (input) INTEGER
            The position of the invalid parameter in the parameter list
            of the calling routine.
*/


    s_wsfe(&io___197);
    do_fio(&c__1, srname, (ftnlen)6);
    do_fio(&c__1, (char *)&(*info), (ftnlen)sizeof(integer));
    e_wsfe();

    s_stop("", (ftnlen)0);


/*     End of XERBLA */

    return 0;
} /* xerbla_ */

