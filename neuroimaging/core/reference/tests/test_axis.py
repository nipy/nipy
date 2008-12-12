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

def test_len():
        ax1 = Axis(name='xspace', length=20)
        ax2 = Axis(name='yspace')
        nose.tools.assert_raises(ValueError, len, ax2)
        nose.tools.assert_equal(len(ax1), 20)




