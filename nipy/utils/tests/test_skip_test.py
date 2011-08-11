"""
Test the test skip utilities.
"""

import nose
from nipy.utils.skip_test import skip_if_running_nose

# First we must check that during test loading time, our skip_test does
# fire
try:
    skip_if_running_nose()
    skip_test_raised = False
except nose.SkipTest:
    skip_test_raised = True

def test_raise_at_load_time():
    """ Check that SkipTest was raised at load time
    """
    nose.tools.assert_true(skip_test_raised)

def test_not_raise_at_run_time():
    """ Check that SkipTest is not raised at run time 
    """
    try:
        skip_if_running_nose()
    except nose.SkipTest:
        # We need to raise another exception, as nose will capture this
        # one
        raise AssertionError

