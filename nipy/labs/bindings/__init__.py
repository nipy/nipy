# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from .linalg import (blas_dnrm2, blas_dasum, blas_ddot, blas_daxpy, blas_dscal,
                     blas_dgemm, blas_dsymm, blas_dtrmm, blas_dtrsm, blas_dsyrk,
                     blas_dsyr2k, matrix_add, matrix_get, matrix_transpose,
                     vector_get, vector_set, vector_add, vector_sub, vector_mul,
                     vector_div, vector_sum)
from .array import (array_get, array_get_block, array_add, array_sub, array_mul,
                    array_div)
from .wrapper import (c_types, fff_type, npy_type, copy_vector, pass_matrix,
                      pass_vector, pass_array, pass_vector_via_iterator,
                      sum_via_iterators, copy_via_iterators)

from warnings import warn

warn('Module nipy.labs.bindings deprecated, will be removed',
     FutureWarning,
     stacklevel=2)

from nipy.testing import Tester
test = Tester().test
bench = Tester().bench
