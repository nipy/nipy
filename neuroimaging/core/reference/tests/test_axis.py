import types
import numpy as np
import nose.tools
from neuroimaging.testing import *

from neuroimaging.core.reference.axis import Axis

def test_eq():
        ax1 = Axis(name='xspace')
        ax2 = Axis(name='yspace')
        ax3 = Axis(name='xspace', dtype=np.complex)
        nose.tools.assert_true(ax1 == ax1)
        nose.tools.assert_false(ax1 == ax2)
        nose.tools.assert_false(ax1 == ax3)

def test_dtype():
	ax = Axis(name='x', dtype=np.int32)
        nose.tools.assert_equal(ax.dtype, np.dtype([('x', np.int32)]))
        nose.tools.assert_equal(ax.builtin, np.dtype(np.int32))

	ax.dtype = np.float
        nose.tools.assert_equal(ax.dtype, np.dtype([('x', np.float)]))
        nose.tools.assert_equal(ax.builtin, np.dtype(np.float))




