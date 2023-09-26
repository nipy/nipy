# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from warnings import warn

from .array import (
    array_add,
    array_div,
    array_get,
    array_get_block,
    array_mul,
    array_sub,
)
from .linalg import (
    blas_dasum,
    blas_daxpy,
    blas_ddot,
    blas_dgemm,
    blas_dnrm2,
    blas_dscal,
    blas_dsymm,
    blas_dsyr2k,
    blas_dsyrk,
    blas_dtrmm,
    blas_dtrsm,
    matrix_add,
    matrix_get,
    matrix_transpose,
    vector_add,
    vector_div,
    vector_get,
    vector_mul,
    vector_set,
    vector_sub,
    vector_sum,
)
from .wrapper import (
    c_types,
    copy_vector,
    copy_via_iterators,
    fff_type,
    npy_type,
    pass_array,
    pass_matrix,
    pass_vector,
    pass_vector_via_iterator,
    sum_via_iterators,
)

warn('Module nipy.labs.bindings deprecated, will be removed',
     FutureWarning,
     stacklevel=2)
