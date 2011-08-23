# Includes
from numpy cimport import_array, ndarray, flatiter

cdef extern from "permutations.h":
    void permutations_import_array()


# Initialize numpy
permutations_import_array()
import_array()
import numpy as np

